import os
import argparse
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset import MMEmoDataset, collate_fn, ID2LABEL
from model import MultiModalSentiment

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="path to best_*.pt")
    parser.add_argument("--mode", type=str, default="multimodal", choices=["multimodal", "text_only", "image_only"])
    parser.add_argument("--text_model", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--out", type=str, default=None)

    parser.add_argument("--clean_text", action="store_true", help="force enable text cleaning")
    parser.add_argument("--no_clean_text", action="store_true", help="force disable text cleaning")
    args = parser.parse_args()

    device = get_device()
    code_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(code_dir, "..", "project5"))  # project5/

    ckpt = torch.load(args.ckpt, map_location="cpu")

    # 用 ckpt 里记录的 clean_text（避免不一致）
    clean_text_flag = bool(ckpt.get("clean_text", True))
    if args.clean_text:
        clean_text_flag = True
    if args.no_clean_text:
        clean_text_flag = False

    test_path = os.path.join(root, "test_without_label.txt")
    df = pd.read_csv(test_path)

    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    test_set = MMEmoDataset(
        root, df, tokenizer,
        max_len=args.max_len,
        image_size=args.image_size,
        is_train=False,
        clean_text_flag=clean_text_flag,
    )
    test_loader = DataLoader(test_set, batch_size=args.bs, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = MultiModalSentiment(text_model_name=args.text_model, mode=args.mode, num_classes=3).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    preds = []
    guids_all = []

    for batch in tqdm(test_loader, desc="infer"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        image = batch["image"].to(device)

        logits = model(input_ids, attention_mask, image)
        pred_ids = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        preds.extend([ID2LABEL[i] for i in pred_ids])
        guids_all.extend(batch["guid"])

    out_df = pd.DataFrame({"guid": guids_all, "tag": preds})

    os.makedirs(os.path.join(code_dir, "outputs"), exist_ok=True)
    if args.out is None:
        args.out = os.path.join(code_dir, "outputs", f"test_pred_{args.mode}.txt")
    out_df.to_csv(args.out, index=False)

    print("Saved:", args.out)

if __name__ == "__main__":
    main()