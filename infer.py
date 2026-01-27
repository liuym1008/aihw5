import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from dataset import MultiModalDataset, ID2LABEL
from model import MultiModalSentimentModel, ModelConfig

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--bs", type=int, default=16)
    args = parser.parse_args()

    device = get_device()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    text_model = ckpt["text_model"]
    cfg_dict = ckpt["cfg"]
    cfg = ModelConfig(**cfg_dict)

    tokenizer = AutoTokenizer.from_pretrained(text_model)
    ds = MultiModalDataset(
        data_path=args.data_path,
        image_root=args.image_root,
        tokenizer=tokenizer,
        max_len=ckpt["args"].get("max_len", 128),
        image_size=ckpt["args"].get("image_size", 224),
        train=False,
        text_clean=ckpt["args"].get("text_clean", "basic"),
        img_aug="none",
    )

    loader = torch.utils.data.DataLoader(ds, batch_size=args.bs, shuffle=False)

    model = MultiModalSentimentModel(text_model, cfg).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    all_preds = []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        img = batch["image"].to(device)

        logits = model(input_ids=ids, attention_mask=attn, images=img)
        prob = F.softmax(logits, dim=-1)
        pred = prob.argmax(dim=-1).detach().cpu().tolist()
        all_preds.extend(pred)

    for i, p in enumerate(all_preds[:50]):
        print(i, ID2LABEL[p])

if __name__ == "__main__":
    main()