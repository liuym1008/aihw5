import os
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from dataset import MultiModalDataset, LABEL2ID
from visualbert_model import VisualBertSentimentModel, VisualBertConfigLocal

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

def compute_class_stats(dataset: MultiModalDataset):
    counts = np.zeros(3, dtype=np.int64)
    for s in dataset.samples:
        y = getattr(s, "label", None)
        if y is None:
            continue
        if isinstance(y, int):
            counts[int(y)] += 1
        else:
            counts[LABEL2ID[str(y).strip().lower()]] += 1
    return counts

def make_inv_class_weight(counts: np.ndarray, power: float = 0.5):
    w = (counts.sum() / np.maximum(counts, 1)) ** power
    w = w / w.mean()
    return w.astype(np.float32)

def label_smoothing_ce(logits, target, smoothing=0.1, weight=None):
    n = logits.size(-1)
    logp = F.log_softmax(logits, dim=-1)
    with torch.no_grad():
        true_dist = torch.zeros_like(logp)
        true_dist.fill_(smoothing / (n - 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - smoothing)
    if weight is not None:
        w = weight.unsqueeze(0)
        loss = -(true_dist * logp) * w
        return loss.sum(dim=-1).mean()
    return -(true_dist * logp).sum(dim=-1).mean()

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

@torch.no_grad()
def evaluate(model, loader, device, logit_adjust=None):
    model.eval()
    all_logits, all_y = [], []

    for batch in loader:
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        img = batch["image"].to(device)
        y = batch["label"].to(device)

        logits = model(input_ids=ids, attention_mask=attn, images=img)
        if logit_adjust is not None:
            logits = logits + logit_adjust

        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    y = torch.cat(all_y, dim=0)
    pred = logits.argmax(dim=-1)

    acc = (pred == y).float().mean().item()

    f1s = []
    for c in range(3):
        tp = ((pred == c) & (y == c)).sum().item()
        fp = ((pred == c) & (y != c)).sum().item()
        fn = ((pred != c) & (y == c)).sum().item()
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1s.append(2 * prec * rec / (prec + rec + 1e-12))

    return acc, float(np.mean(f1s))

def build_optimizer(model, base_lr, wd, head_lr_mul=12.0, img_lr_mul=0.5):
    head_params, vb_params, img_params = [], [], []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "head" in n:
            head_params.append(p)
        elif "image_encoder" in n:
            img_params.append(p)
        else:
            vb_params.append(p)

    return torch.optim.AdamW(
        [
            {"params": vb_params, "lr": base_lr, "weight_decay": wd},
            {"params": img_params, "lr": base_lr * img_lr_mul, "weight_decay": wd},
            {"params": head_params, "lr": base_lr * head_lr_mul, "weight_decay": wd},
        ]
    )

def train_one_mode(args, mode: str):
    device = get_device()
    print("device:", device)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    # 全部用关键字传参，避免把 image_transform 误传成 int
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

    counts = compute_class_stats(train_set)
    inv_w = make_inv_class_weight(counts, args.class_weight_power)
    print(f"[{mode}] class_count={counts.tolist()}, inv_class_weight={inv_w.tolist()}")

    sampler = None
    if args.sampler:
        sample_w = []
        for s in train_set.samples:
            yy = int(s.label) if isinstance(s.label, int) else LABEL2ID[str(s.label).strip().lower()]
            sample_w.append(float(inv_w[yy]))
        sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
        print(f"[{mode}] sampler enabled.")

    train_loader = DataLoader(
        train_set,
        batch_size=args.bs,
        shuffle=(sampler is None),
        sampler=sampler,
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

    logit_adjust = None
    if args.logit_adjust:
        priors = counts / counts.sum()
        logit_adjust = (-args.adjust_tau * torch.log(torch.tensor(priors + 1e-12, dtype=torch.float32))).to(device)

    cfg = VisualBertConfigLocal(
        num_classes=3,
        dropout=args.dropout,
        mode=mode,
        freeze_image=(args.freeze_image_epochs > 0),
        visualbert_name=args.visualbert_name,
    )
    model = VisualBertSentimentModel(cfg).to(device)

    freeze_until = args.freeze_image_epochs

    class_weight = torch.tensor(inv_w, dtype=torch.float32).to(device) if args.use_class_weight else None
    optimizer = build_optimizer(model, args.lr, args.wd, args.head_lr_mul, args.img_lr_mul)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    criterion = None
    if args.use_focal:
        criterion = FocalLoss(gamma=args.focal_gamma, weight=class_weight)

    ema = EMA(model, args.ema_decay) if args.ema else None

    best_mf1 = -1.0
    best_path = os.path.join(args.run_dir, f"best_{mode}.pt")
    history = []

    for ep in range(1, args.epochs + 1):
        model.train()

        if freeze_until > 0 and ep > freeze_until:
            for p in model.image_encoder.parameters():
                p.requires_grad = True
            freeze_until = 0
            print(f"[{mode}] unfreeze image_encoder at epoch {ep}.")

        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            img = batch["image"].to(device)
            y = batch["label"].to(device)

            logits = model(input_ids=ids, attention_mask=attn, images=img)
            if logit_adjust is not None:
                logits = logits + logit_adjust

            if criterion is not None:
                loss = criterion(logits, y)
            elif args.label_smoothing > 0:
                loss = label_smoothing_ce(logits, y, smoothing=args.label_smoothing, weight=class_weight)
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

        if ema is not None:
            ema.apply_shadow(model)
        val_acc, val_mf1 = evaluate(model, val_loader, device, logit_adjust)
        if ema is not None:
            ema.restore(model)

        print(f"[{mode}] epoch {ep}/{args.epochs}: loss={avg_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_mf1:.4f}")
        history.append({"epoch": ep, "train_loss": float(avg_loss), "val_acc": float(val_acc), "val_macro_f1": float(val_mf1)})

        if val_mf1 > best_mf1:
            best_mf1 = val_mf1
            os.makedirs(args.run_dir, exist_ok=True)
            torch.save(
                {
                    "arch": "visualbert",
                    "model_state": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "args": vars(args),
                    "logit_adjust": (logit_adjust.detach().cpu() if logit_adjust is not None else None),
                },
                best_path,
            )
            print(f"[{mode}] saved best -> {best_path}")

    os.makedirs(args.run_dir, exist_ok=True)
    with open(os.path.join(args.run_dir, f"history_{mode}.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(history, f, ensure_ascii=False, indent=2)

    return best_path

def main():
    p = argparse.ArgumentParser()

    p.add_argument("--train_path", type=str, required=True)
    p.add_argument("--val_path", type=str, required=True)
    p.add_argument("--image_root", type=str, required=True)
    p.add_argument("--run_dir", type=str, default="runs_visualbert")

    p.add_argument("--visualbert_name", type=str, default="uclanlp/visualbert-vqa-coco-pre")
    p.add_argument("--tokenizer_name", type=str, default="bert-base-uncased")

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--head_lr_mul", type=float, default=12.0)
    p.add_argument("--img_lr_mul", type=float, default=0.5)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.06)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--text_clean", type=str, default="basic")
    p.add_argument("--img_aug", type=str, default="strong")

    p.add_argument("--sampler", action="store_true")
    p.add_argument("--use_class_weight", action="store_true")
    p.add_argument("--class_weight_power", type=float, default=0.5)
    p.add_argument("--label_smoothing", type=float, default=0.0)

    p.add_argument("--logit_adjust", action="store_true")
    p.add_argument("--adjust_tau", type=float, default=0.3)

    p.add_argument("--freeze_image_epochs", type=int, default=3)

    p.add_argument("--use_focal", action="store_true")
    p.add_argument("--focal_gamma", type=float, default=2.0)

    p.add_argument("--ema", action="store_true")
    p.add_argument("--ema_decay", type=float, default=0.999)

    args = p.parse_args()
    seed_everything(42)
    os.makedirs(args.run_dir, exist_ok=True)

    train_one_mode(args, "multimodal")

if __name__ == "__main__":
    main()