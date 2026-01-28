import os
import csv
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from dataset import MultiModalDataset, ID2LABEL
from blip_model import BlipSentimentModel, BlipConfigLocal

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

    ds = MultiModalDataset(
        data_path=args.data_path,
        image_root=args.image_root,
        tokenizer=tokenizer,
        max_len=max_len,
        image_size=image_size,
        is_test=False,
        text_clean=text_clean,
        img_aug="none",
        train=False,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=0)

    cfg = BlipConfigLocal(**cfg_dict) if cfg_dict else BlipConfigLocal()
    model = BlipSentimentModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    rows = []
    bad_rows = []
    conf = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    total, correct = 0, 0

    for batch in loader:
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        img = batch["image"].to(device)
        y = batch["label"].to(device)
        guid = batch["guid"]

        logits = model(ids, attn, img)
        prob = F.softmax(logits, dim=-1)
        pred = prob.argmax(dim=-1)

        for i in range(len(pred)):
            gt = int(y[i].item())
            pr = int(pred[i].item())
            total += 1
            correct += int(gt == pr)
            conf[gt][pr] += 1

            row = {
                "guid": str(guid[i]),
                "gt": ID2LABEL[gt],
                "pred": ID2LABEL[pr],
                "p_neg": float(prob[i, 0].item()),
                "p_neu": float(prob[i, 1].item()),
                "p_pos": float(prob[i, 2].item()),
            }
            rows.append(row)
            if gt != pr and len(bad_rows) < args.max_bad:
                bad_rows.append(row)

    acc = correct / max(1, total)
    mf1 = macro_f1_from_conf(conf)
    print(f"[infer_blip] total={total} acc={acc:.4f} macro_f1={mf1:.4f}")

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

if __name__ == "__main__":
    main()