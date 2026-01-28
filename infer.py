import argparse
import csv
import os
from typing import List, Dict, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from dataset import MultiModalDataset, ID2LABEL, LABEL2ID
from model import MultiModalSentimentModel, ModelConfig

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def macro_f1_from_confmat(conf):
    # conf: [C][C]  rows=gt, cols=pred
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
    """
    Heuristic: read the first non-empty data row, look at the 2nd column (label/tag).
    - If it's 'null' / empty / unknown label -> treat as test_without_label
    - Else treat as labeled (val/train)
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            header = f.readline()
            if not header:
                return True
            # detect delimiter similarly to dataset.py
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
                # unknown label token -> likely test_without_label or corrupted
                return True
    except Exception:
        return True

def write_submit_txt(src_txt: str, out_txt: str, guid2pred: Dict[str, str]):
    """
    把 test_without_label.txt 中的 label(null) 替换成预测标签
    兼容常见格式：guid,tag,post 或 guid\\ttag\\tpost 或 guid tag post
    默认把“第2列”替换成预测标签。
    """
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--bs", type=int, default=16)

    # 输出文件（不改原 txt）
    parser.add_argument("--out_csv", type=str, default="runs/val_pred_with_gt.csv")
    parser.add_argument("--out_bad_csv", type=str, default="runs/val_bad_cases.csv")
    parser.add_argument("--max_bad", type=int, default=100)   # bad case最多保存多少条
    parser.add_argument("--max_print", type=int, default=30)  # 终端最多打印多少条

    # 生成提交文件：把 test_without_label.txt 的 null 替换成预测标签
    parser.add_argument("--out_submit_txt", type=str, default=None)

    parser.add_argument("--is_test", action="store_true", help="Force treat data_path as test_without_label")

    args = parser.parse_args()
    device = get_device()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    cfg = ModelConfig(**ckpt["cfg"])

    # 推理必须复用训练时的 tokenizer/backbone
    use_clip = bool(ckpt_args.get("use_clip", False))
    clip_name = ckpt_args.get("clip_name", "openai/clip-vit-base-patch32")
    text_model = ckpt.get("text_model", "bert-base-multilingual-cased")

    tokenizer_name = clip_name if use_clip else text_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if use_clip:
        tokenizer.model_max_length = min(getattr(tokenizer, "model_max_length", 77), 77)

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
        train=False,
        text_clean=text_clean,
        img_aug="none",
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=args.bs, shuffle=False)

    text_backbone = clip_name if use_clip else text_model
    model = MultiModalSentimentModel(text_backbone, cfg).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    logit_adjust = ckpt.get("logit_adjust", None)
    if logit_adjust is not None:
        logit_adjust = logit_adjust.to(device)

    _ensure_dir(args.out_csv)
    _ensure_dir(args.out_bad_csv)

    all_rows: List[Dict] = []
    bad_rows: List[Dict] = []

    # 统计（仅在有 GT 时）
    total = 0
    correct = 0
    conf = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]  # 3类

    guid2pred_label: Dict[str, str] = {}

    for batch in loader:
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        img = batch["image"].to(device)

        guids = batch.get("guid", None)
        gts = batch.get("label", None)  # if is_test=True -> None

        logits = model(input_ids=ids, attention_mask=attn, images=img)
        if logit_adjust is not None:
            logits = logits + logit_adjust

        prob = F.softmax(logits, dim=-1)
        pred = prob.argmax(dim=-1).detach().cpu().tolist()
        prob_cpu = prob.detach().cpu().tolist()

        bs = len(pred)
        for i in range(bs):
            p = int(pred[i])
            probs = prob_cpu[i]
            guid = str(guids[i]) if guids is not None else str(total + i)

            guid2pred_label[guid] = ID2LABEL[p]

            # 如果有 gt（val_split.txt 是有标签的），就算指标/找 badcase
            if gts is not None:
                gt = int(gts[i])
                is_ok = (p == gt)
                total += 1
                correct += int(is_ok)
                conf[gt][p] += 1

                row = {
                    "guid": guid,
                    "gt_id": gt,
                    "gt": ID2LABEL[gt],
                    "pred_id": p,
                    "pred": ID2LABEL[p],
                    "correct": int(is_ok),
                    "p0_negative": probs[0],
                    "p1_neutral": probs[1],
                    "p2_positive": probs[2],
                }
                all_rows.append(row)

                if (not is_ok) and (len(bad_rows) < args.max_bad):
                    bad_rows.append(row)
            else:
                # 没有 gt（test_without_label.txt 这种），只输出预测
                row = {
                    "guid": guid,
                    "pred_id": p,
                    "pred": ID2LABEL[p],
                    "p0_negative": probs[0],
                    "p1_neutral": probs[1],
                    "p2_positive": probs[2],
                }
                all_rows.append(row)
                total += 1

    # 写出预测 CSV
    if all_rows:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    # 写出 bad case（仅有 GT 时才会有）
    if bad_rows:
        with open(args.out_bad_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(bad_rows[0].keys()))
            writer.writeheader()
            writer.writerows(bad_rows)

    # 打印指标
    if all_rows and ("gt" in all_rows[0]):  # 有标签
        acc = correct / max(1, total)
        mf1 = macro_f1_from_confmat(conf)
        print(f"[infer] total={total} acc={acc:.4f} macro_f1={mf1:.4f}")
        print(f"[infer] saved full preds -> {args.out_csv}")
        print(f"[infer] saved bad cases -> {args.out_bad_csv} (top {len(bad_rows)})")
    else:
        print(f"[infer] total={total} (no GT) saved preds -> {args.out_csv}")

    # 终端少量预览
    if all_rows:
        print(f"[infer] preview first {min(args.max_print, len(all_rows))} rows:")
        for r in all_rows[: args.max_print]:
            if "gt" in r:
                print(r["guid"], "gt=", r["gt"], "pred=", r["pred"], "ok=", r["correct"])
            else:
                print(r["guid"], "pred=", r["pred"])

    # 生成提交 txt
    if args.out_submit_txt is not None:
        write_submit_txt(args.data_path, args.out_submit_txt, guid2pred_label)
        print(f"[infer] saved submit txt -> {args.out_submit_txt}")

if __name__ == "__main__":
    main()