import argparse
import csv
import os
from typing import List, Dict
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from dataset import MultiModalDataset, ID2LABEL
from visualbert_model import VisualBertSentimentModel, VisualBertConfigLocal

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def macro_f1_from_confmat(conf):
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

def write_submit_txt(src_txt: str, out_txt: str, guid2pred: Dict[int, str]):
    """
    把 test_without_label.txt 中的 label(null) 替换成预测标签
    常见格式：guid,tag,post  或 guid tag post（dataset.py 也支持）
    这里按“行内第2列是标签”处理：把 null 替换成 pred
    """
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)

    with open(src_txt, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        raise RuntimeError("empty input txt")

    header = lines[0].rstrip("\n")
    out_lines = [header + "\n"]

    for line in lines[1:]:
        raw = line.rstrip("\n")
        if not raw.strip():
            continue

        # 尝试逗号分隔；否则按 tab/空格
        if "," in raw:
            parts = raw.split(",")
            sep = ","
        elif "\t" in raw:
            parts = raw.split("\t")
            sep = "\t"
        else:
            parts = raw.split()
            sep = " "

        # guid 在第1列
        guid = int(parts[0])
        pred = guid2pred.get(guid, "neutral")

        # label 在第2列（tag/label/sentiment）
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
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--bs", type=int, default=16)

    parser.add_argument("--out_csv", type=str, default="runs_visualbert/preds.csv")
    parser.add_argument("--out_bad_csv", type=str, default="runs_visualbert/bad_cases.csv")
    parser.add_argument("--max_bad", type=int, default=100)
    parser.add_argument("--max_print", type=int, default=30)

    # 生成提交文件：把 test_without_label.txt 的 null 替换成预测标签
    parser.add_argument("--out_submit_txt", type=str, default=None)

    args = parser.parse_args()
    device = get_device()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
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
        train=False,
        text_clean=text_clean,
        img_aug="none",
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=args.bs, shuffle=False)

    # mode 推理：用 ckpt 的 cfg
    cfg_dict = ckpt.get("cfg", {})
    cfg = VisualBertConfigLocal(**cfg_dict)
    model = VisualBertSentimentModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    logit_adjust = ckpt.get("logit_adjust", None)
    if logit_adjust is not None:
        logit_adjust = logit_adjust.to(device)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_bad_csv), exist_ok=True)

    all_rows: List[Dict] = []
    bad_rows: List[Dict] = []

    total = 0
    correct = 0
    conf = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    guid2pred_label = {}

    for batch in loader:
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        img = batch["image"].to(device)

        guids = batch.get("guid", None)
        gts = batch.get("label", None)

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
            guid = int(guids[i]) if guids is not None else (total + i)

            guid2pred_label[guid] = ID2LABEL[p]

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

    # 写 CSV
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    if len(bad_rows) > 0:
        with open(args.out_bad_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(bad_rows[0].keys()))
            writer.writeheader()
            writer.writerows(bad_rows)

    # 打印指标
    if "gt" in all_rows[0]:
        acc = correct / max(1, total)
        mf1 = macro_f1_from_confmat(conf)
        print(f"[infer_visualbert] total={total} acc={acc:.4f} macro_f1={mf1:.4f}")
        print(f"[infer_visualbert] saved full preds -> {args.out_csv}")
        print(f"[infer_visualbert] saved bad cases -> {args.out_bad_csv} (top {len(bad_rows)})")
    else:
        print(f"[infer_visualbert] total={total} (no GT) saved preds -> {args.out_csv}")

    print(f"[infer_visualbert] preview first {min(args.max_print, len(all_rows))} rows:")
    for r in all_rows[: args.max_print]:
        if "gt" in r:
            print(r["guid"], "gt=", r["gt"], "pred=", r["pred"], "ok=", r["correct"])
        else:
            print(r["guid"], "pred=", r["pred"])

    # 生成提交 txt
    if args.out_submit_txt is not None:
        write_submit_txt(args.data_path, args.out_submit_txt, guid2pred_label)
        print(f"[infer_visualbert] saved submit txt -> {args.out_submit_txt}")

if __name__ == "__main__":
    main()