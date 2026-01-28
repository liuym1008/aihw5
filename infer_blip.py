import os
import csv
import argparse
from typing import Dict, List
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from dataset import MultiModalDataset, ID2LABEL, LABEL2ID
from blip_model import BlipSentimentModel, BlipConfigLocal, MLPHead

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def macro_f1_from_conf(conf):
    C = len(conf)
    f1s = []
    for k in range(C):
        tp = conf[k][k]
        fp = sum(conf[i][k] for i in range(C)) - tp
        fn = sum(conf[k][j] for j in range(C)) - tp
        denom = (2 * tp + fp + fn)
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / C

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def infer_is_test_txt(path: str) -> bool:
    """Same heuristic as infer.py"""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            header = f.readline()
            if not header:
                return True
            comma = header.count(",")
            tab = header.count("\t")
            delim = "\t" if tab > comma else ","

            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                if delim in raw:
                    parts = raw.split(delim)
                elif "\t" in raw:
                    parts = raw.split("\t")
                elif "," in raw:
                    parts = raw.split(",")
                else:
                    parts = raw.split()
                if len(parts) < 2:
                    return True
                y = str(parts[1]).strip().lower()
                if y in ("", "null", "none", "nan"):
                    return True
                if y in ("neg", "neu", "pos"):
                    return False
                if y in LABEL2ID:
                    return False
                return True
    except Exception:
        return True

def write_submit_txt(src_txt: str, out_txt: str, guid2pred: Dict[str, str]):
    """Replace 2nd column with predicted label."""
    _ensure_dir(out_txt)

    with open(src_txt, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    if not lines:
        raise RuntimeError("empty input txt")

    header = lines[0].rstrip("\n")
    out_lines = [header + "\n"]

    for line in lines[1:]:
        raw = line.rstrip("\n")
        if not raw.strip():
            continue

        if "," in raw:
            parts = raw.split(",")
            sep = ","
        elif "\t" in raw:
            parts = raw.split("\t")
            sep = "\t"
        else:
            parts = raw.split()
            sep = " "

        guid = str(parts[0]).strip()
        pred = guid2pred.get(guid, "neutral")
        if len(parts) >= 2:
            parts[1] = pred

        out_lines.append(sep.join(parts) + "\n")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.writelines(out_lines)

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--image_root", type=str, required=True)
    p.add_argument("--bs", type=int, default=32)

    p.add_argument("--out_csv", type=str, default="runs_blip/preds.csv")
    p.add_argument("--out_bad_csv", type=str, default="runs_blip/badcases.csv")
    p.add_argument("--max_bad", type=int, default=200)

    # submit file
    p.add_argument("--out_submit_txt", type=str, default=None)
    p.add_argument("--is_test", action="store_true", help="Force treat data_path as test_without_label")

    args = p.parse_args()
    device = get_device()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    cfg_dict = ckpt.get("cfg", {})

    tokenizer_name = ckpt_args.get("tokenizer_name", "bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    max_len = int(ckpt_args.get("max_len", 128))
    image_size = int(ckpt_args.get("image_size", 224))
    text_clean = ckpt_args.get("text_clean", "basic")

    auto_is_test = infer_is_test_txt(args.data_path)
    is_test = args.is_test or auto_is_test

    ds = MultiModalDataset(
        data_path=args.data_path,
        image_root=args.image_root,
        tokenizer=tokenizer,
        max_len=max_len,
        image_size=image_size,
        is_test=is_test,
        text_clean=text_clean,
        img_aug="none",
        train=False,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=0)

    cfg = BlipConfigLocal(**cfg_dict) if cfg_dict else BlipConfigLocal()

    # 1) 先在 CPU 上建模型 + head（避免 head 留在 CPU / 模型在 MPS 的混用）
    model = BlipSentimentModel(cfg)
    sd = ckpt["model_state"]
    if getattr(model, "head", None) is None and ("head.net.0.weight" in sd):
        in_dim = sd["head.net.0.weight"].shape[0]
        model.head = MLPHead(in_dim, cfg.num_classes, cfg.dropout)

    # 2) 在 CPU 上加载权重
    model.load_state_dict(sd, strict=True)

    # 3) 最后整体搬到 device（MPS）
    model = model.to(device)
    model.eval()

    _ensure_dir(args.out_csv)
    _ensure_dir(args.out_bad_csv)

    rows: List[Dict] = []
    bad_rows: List[Dict] = []
    conf = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    total, correct = 0, 0
    guid2pred: Dict[str, str] = {}

    for batch in loader:
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        img = batch["image"].to(device)

        guid = batch["guid"]
        y = batch.get("label", None)

        logits = model(ids, attn, img)
        prob = F.softmax(logits, dim=-1)
        pred = prob.argmax(dim=-1)

        for i in range(len(pred)):
            pr = int(pred[i].item())
            g = str(guid[i])
            guid2pred[g] = ID2LABEL[pr]

            if y is not None:
                gt = int(y[i].item())
                total += 1
                correct += int(gt == pr)
                conf[gt][pr] += 1

                row = {
                    "guid": g,
                    "gt": ID2LABEL[gt],
                    "pred": ID2LABEL[pr],
                    "p_neg": float(prob[i, 0].item()),
                    "p_neu": float(prob[i, 1].item()),
                    "p_pos": float(prob[i, 2].item()),
                }
                rows.append(row)
                if gt != pr and len(bad_rows) < args.max_bad:
                    bad_rows.append(row)
            else:
                total += 1
                row = {
                    "guid": g,
                    "pred": ID2LABEL[pr],
                    "p_neg": float(prob[i, 0].item()),
                    "p_neu": float(prob[i, 1].item()),
                    "p_pos": float(prob[i, 2].item()),
                }
                rows.append(row)

    # metrics if labeled
    if rows and ("gt" in rows[0]):
        acc = correct / max(1, total)
        mf1 = macro_f1_from_conf(conf)
        print(f"[infer_blip] total={total} acc={acc:.4f} macro_f1={mf1:.4f}")
    else:
        print(f"[infer_blip] total={total} (no GT)")

    # write csv
    if rows:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[infer_blip] saved -> {args.out_csv}")

    if bad_rows:
        with open(args.out_bad_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(bad_rows[0].keys()))
            w.writeheader()
            w.writerows(bad_rows)
        print(f"[infer_blip] saved badcases -> {args.out_bad_csv}")

    # submit txt
    if args.out_submit_txt is not None:
        write_submit_txt(args.data_path, args.out_submit_txt, guid2pred)
        print(f"[infer_blip] saved submit txt -> {args.out_submit_txt}")

if __name__ == "__main__":
    main()