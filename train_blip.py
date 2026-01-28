import os
import json
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from dataset import MultiModalDataset, LABEL2ID
from blip_model import BlipSentimentModel, BlipConfigLocal

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def compute_class_counts(ds: MultiModalDataset):
    counts = np.zeros(3, dtype=np.int64)
    for s in ds.samples:
        y = getattr(s, "label", None)
        if y is None:
            continue
        if isinstance(y, int):
            counts[int(y)] += 1
        else:
            counts[LABEL2ID[str(y).strip().lower()]] += 1
    return counts

def inv_class_weight(counts: np.ndarray, power: float = 0.5):
    w = (counts.sum() / np.maximum(counts, 1)) ** power
    w = w / w.mean()
    return w.astype(np.float32)

# Loss / EMA
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=-1)
        p = torch.exp(logp)
        pt = p.gather(1, target.unsqueeze(1)).squeeze(1)
        focal = (1 - pt).pow(self.gamma)
        ce = F.nll_loss(logp, target, weight=self.weight, reduction="none")
        return (focal * ce).mean()

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone()

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply_shadow(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}

# 指标：acc + macro_f1
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        img = batch["image"].to(device)
        y = batch["label"].to(device)

        logits = model(ids, attn, img)
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    y = torch.cat(all_y, dim=0)
    pred = logits.argmax(dim=-1)

    for c in range(3):
        print("pred_count", c, int((pred==c).sum()))

    acc = (pred == y).float().mean().item()

    f1s = []
    for c in range(3):
        tp = ((pred == c) & (y == c)).sum().item()
        fp = ((pred == c) & (y != c)).sum().item()
        fn = ((pred != c) & (y == c)).sum().item()
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        f1s.append(f1)

    return acc, float(np.mean(f1s))

# 三分组优化器：text/vision/head
def build_optimizer(model, base_lr, wd, head_lr_mul=10.0, vision_lr_mul=0.5):
    head_params, vision_params, other_params = [], [], []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "head" in n:
            head_params.append(p)
        elif "blip.vision_model" in n:
            vision_params.append(p)
        else:
            other_params.append(p)

    return torch.optim.AdamW(
        [
            {"params": other_params, "lr": base_lr, "weight_decay": wd},
            {"params": vision_params, "lr": base_lr * vision_lr_mul, "weight_decay": wd},
            {"params": head_params, "lr": base_lr * head_lr_mul, "weight_decay": wd},
        ]
    )

def train_one(args):
    device = get_device()
    print("device:", device)

    # tokenizer（英文）
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    print(f"[blip] using blip_name={args.blip_name}")

    # dataset：全部关键字传参，避免顺序坑
    train_set = MultiModalDataset(
        data_path=args.train_path,
        image_root=args.image_root,
        tokenizer=tokenizer,
        max_len=args.max_len,
        image_size=args.image_size,
        is_test=False,
        text_clean=args.text_clean,
        img_aug=args.img_aug,
        train=True,
    )
    val_set = MultiModalDataset(
        data_path=args.val_path,
        image_root=args.image_root,
        tokenizer=tokenizer,
        max_len=args.max_len,
        image_size=args.image_size,
        is_test=False,
        text_clean=args.text_clean,
        img_aug="none",
        train=False,
    )

    counts = compute_class_counts(train_set)
    inv_w = inv_class_weight(counts, power=args.class_weight_power)
    print(f"[blip] class_count={counts.tolist()}, inv_class_weight={inv_w.tolist()}")

    train_loader = DataLoader(
        train_set,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # model
    cfg = BlipConfigLocal(
        blip_name=args.blip_name,
        num_classes=3,
        dropout=args.dropout,
        pool=args.pool,
        use_safetensors=True,
    )
    model = BlipSentimentModel(cfg).to(device)

    # 冻结 vision 前几轮
    model.freeze_vision(True)

    # class weight
    class_weight = torch.tensor(inv_w, dtype=torch.float32).to(device) if args.use_class_weight else None

    # loss
    if args.use_focal:
        criterion = FocalLoss(gamma=args.focal_gamma, weight=class_weight)
    else:
        criterion = None  # 用 CE

    # optim / sched
    optimizer = build_optimizer(
        model,
        base_lr=args.lr,
        wd=args.wd,
        head_lr_mul=args.head_lr_mul,
        vision_lr_mul=args.vision_lr_mul,
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # EMA
    ema = EMA(model, decay=args.ema_decay) if args.ema else None

    os.makedirs(args.run_dir, exist_ok=True)
    best_path = os.path.join(args.run_dir, "best_blip.pt")
    hist_path = os.path.join(args.run_dir, "history_blip.json")

    best_mf1 = -1.0
    no_improve = 0
    history = []

    for ep in range(1, args.epochs + 1):
        model.train()

        # 解冻 vision
        if ep == args.freeze_vision_epochs + 1:
            model.freeze_vision(False)
            print(f"[blip] unfreeze vision at epoch {ep}")

        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            img = batch["image"].to(device)
            y = batch["label"].to(device)

            logits = model(ids, attn, img)

            if criterion is not None:
                loss = criterion(logits, y)
            else:
                loss = F.cross_entropy(logits, y, weight=class_weight)

            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(model)

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))

        # eval（用 EMA 权重更稳）
        if ema is not None:
            ema.apply_shadow(model)
        val_acc, val_mf1 = evaluate(model, val_loader, device)
        if ema is not None:
            ema.restore(model)

        print(f"[blip] epoch {ep}/{args.epochs}: loss={avg_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_mf1:.4f}")

        history.append({
            "epoch": ep,
            "train_loss": float(avg_loss),
            "val_acc": float(val_acc),
            "val_macro_f1": float(val_mf1),
        })

        # save best
        if val_mf1 > best_mf1:
            best_mf1 = val_mf1
            no_improve = 0
            torch.save(
                {
                    "arch": "blip",
                    "model_state": model.state_dict(),
                    "cfg": model.export_config(),
                    "args": vars(args),
                    "best_val_macro_f1": float(best_mf1),
                },
                best_path,
            )
            print(f"[blip] saved best -> {best_path}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"[blip] early stop at epoch {ep}, best_macro_f1={best_mf1:.4f}")
                break

        # save history each epoch
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    # final summary
    with open(os.path.join(args.run_dir, "train_result.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_macro_f1": float(best_mf1),
                "best_ckpt": best_path,
                "history": hist_path,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

def main():
    p = argparse.ArgumentParser()

    p.add_argument("--train_path", type=str, required=True)
    p.add_argument("--val_path", type=str, required=True)
    p.add_argument("--image_root", type=str, required=True)
    p.add_argument("--run_dir", type=str, default="runs_blip")

    p.add_argument("--blip_name", type=str, default="Salesforce/blip-itm-base-coco")
    p.add_argument("--tokenizer_name", type=str, default="bert-base-uncased")

    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--bs", type=int, default=16)

    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--head_lr_mul", type=float, default=10.0)
    p.add_argument("--vision_lr_mul", type=float, default=0.5)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.06)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--pool", type=str, default="cls", choices=["cls", "mean"])

    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--image_size", type=int, default=224)
    # Mac/MPS 稳一点：默认 0
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--text_clean", type=str, default="basic", choices=["none", "basic", "aggressive"])
    p.add_argument("--img_aug", type=str, default="strong", choices=["none", "weak", "strong"])

    p.add_argument("--use_class_weight", action="store_true")
    p.add_argument("--class_weight_power", type=float, default=0.5)

    p.add_argument("--use_focal", action="store_true")
    p.add_argument("--focal_gamma", type=float, default=2.0)

    p.add_argument("--ema", action="store_true")
    p.add_argument("--ema_decay", type=float, default=0.999)

    p.add_argument("--freeze_vision_epochs", type=int, default=2)

    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    seed_everything(args.seed)
    os.makedirs(args.run_dir, exist_ok=True)

    t0 = time.time()
    train_one(args)
    print(f"Done. time={time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()