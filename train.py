import os
import json
import time
import math
import random
import argparse
from dataclasses import asdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from dataset import MultiModalDataset, LABEL2ID
from model import MultiModalSentimentModel, ModelConfig

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Mac
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 1.5, weight=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=-1)
        p = torch.exp(logp)
        # 取目标类的概率
        pt = p.gather(1, target.unsqueeze(1)).squeeze(1)
        logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.weight is not None:
            w = self.weight.gather(0, target)
            loss = loss * w
        return loss.mean()

def label_smoothing_ce(logits, target, smoothing=0.1, weight=None):
    n = logits.size(-1)
    logp = F.log_softmax(logits, dim=-1)
    with torch.no_grad():
        true_dist = torch.zeros_like(logp)
        true_dist.fill_(smoothing / (n - 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - smoothing)
    if weight is not None:
        w = weight.unsqueeze(0)  # (1,C)
        loss = -(true_dist * logp) * w
        return loss.sum(dim=-1).mean()
    return -(true_dist * logp).sum(dim=-1).mean()

def compute_class_stats(dataset: MultiModalDataset):
    counts = np.zeros(3, dtype=np.int64)
    for s in dataset.samples:
        # dataset.samples 里是 Sample(guid, text, label, image_key)
        y = getattr(s, "label", None)
        if y is None:
            continue
        if isinstance(y, int):
            counts[int(y)] += 1
        else:
            counts[LABEL2ID[str(y).strip().lower()]] += 1
    return counts

def make_sampler_from_counts(counts: np.ndarray, power: float = 0.5):
    w = (counts.sum() / np.maximum(counts, 1)) ** power
    w = w / w.mean()
    return w.astype(np.float32)

@torch.no_grad()
def evaluate(model, loader, device, logit_adjust=None):
    model.eval()
    all_logits = []
    all_y = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        img = batch["image"].to(device)
        y = batch["label"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attn, images=img)
        if logit_adjust is not None:
            logits = logits + logit_adjust

        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    y = torch.cat(all_y, dim=0)

    pred = logits.argmax(dim=-1)
    acc = (pred == y).float().mean().item()

    # macro f1
    f1s = []
    for c in range(3):
        tp = ((pred == c) & (y == c)).sum().item()
        fp = ((pred == c) & (y != c)).sum().item()
        fn = ((pred != c) & (y == c)).sum().item()
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))
    return acc, macro_f1

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        self.backup = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = p.detach().clone()
                p.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module):
        for name, p in model.named_parameters():
            if name in getattr(self, "backup", {}):
                p.copy_(self.backup[name])
        self.backup = {}

def build_optimizer(model, base_lr, wd, head_lr_mul=5.0):
    """
    训练策略：encoder 用 base_lr，head 用更大 lr（通常能显著抬升 acc）
    """
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in n for k in ["bias", "LayerNorm.weight", "layer_norm.weight"]):
            no_decay.append(p)
        else:
            decay.append(p)

    # 找 head / proj / fusion 参数给更大学习率
    head_params, base_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in n for k in ["head", "text_proj", "img_proj", "gate", "cross"]):
            head_params.append(p)
        else:
            base_params.append(p)

    param_groups = [
        {"params": base_params, "lr": base_lr, "weight_decay": wd},
        {"params": head_params, "lr": base_lr * head_lr_mul, "weight_decay": wd},
    ]
    return torch.optim.AdamW(param_groups)

def train_one_mode(args, mode: str):
    device = get_device()
    print("device:", device)

    tokenizer_name = args.clip_name if args.use_clip else args.text_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if args.use_clip:
        tokenizer.model_max_length = min(tokenizer.model_max_length, 77)

    train_set = MultiModalDataset(
        data_path=args.train_path,
        image_root=args.image_root,
        tokenizer=tokenizer,
        max_len=args.max_len,
        image_size=args.image_size,
        train=True,
        text_clean=args.text_clean,
        img_aug=args.img_aug,
    )
    val_set = MultiModalDataset(
        data_path=args.val_path,
        image_root=args.image_root,
        tokenizer=tokenizer,
        max_len=args.max_len,
        image_size=args.image_size,
        train=False,
        text_clean=args.text_clean,
        img_aug="none",
    )

    counts = compute_class_stats(train_set)
    inv_w = make_sampler_from_counts(counts, power=args.class_weight_power)
    print(f"[{mode}] class_count={counts.tolist()}, inv_class_weight={inv_w.tolist()}")

    sampler = None
    if args.sampler:
        # 按样本标签给权重
        sample_w = []
        for s in train_set.samples:
            y = getattr(s, "label", None)
            if y is None:
                continue
            if isinstance(y, int):
                yy = int(y)
            else:
                yy = LABEL2ID[str(y).strip().lower()]
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

    # logit adjust
    logit_adjust = None
    if args.logit_adjust:
        priors = counts / counts.sum()
        log_prior = torch.log(torch.tensor(priors + 1e-12, dtype=torch.float32))
        logit_adjust = (-args.adjust_tau * log_prior).to(device)
        print(f"[{mode}] logit_adjust enabled. priors={priors.tolist()} tau={args.adjust_tau}")

    cfg = ModelConfig(
        num_classes=3,
        dropout=args.dropout,
        proj_dim=args.proj_dim,
        fusion=args.fusion,
        mode=mode,
        use_clip=args.use_clip,
        clip_name=args.clip_name,
        freeze_image=(args.freeze_image_epochs > 0),
    )

    text_backbone = args.clip_name if args.use_clip else args.text_model
    model = MultiModalSentimentModel(text_backbone, cfg).to(device)

    # freeze/unfreeze image encoder
    freeze_until = args.freeze_image_epochs if mode in ["multimodal", "image_only"] else 0

    # loss
    class_weight = torch.tensor(inv_w, dtype=torch.float32).to(device) if args.use_class_weight else None

    if args.use_focal:
        criterion = FocalLoss(gamma=args.focal_gamma, weight=class_weight)
        print(f"[{mode}] use FocalLoss(gamma={args.focal_gamma}).")
    else:
        criterion = None  # use CE / LS later

    optimizer = build_optimizer(model, base_lr=args.lr, wd=args.wd, head_lr_mul=args.head_lr_mul)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * args.warmup)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = None
    ema = EMA(model, decay=args.ema_decay) if args.ema else None

    best_mf1 = -1.0
    best_acc = -1.0
    best_path = os.path.join(args.run_dir, f"best_{mode}.pt")
    history = []  # 记录每个 epoch 的指标，最后写成 json
    patience = args.patience
    no_improve = 0

    for ep in range(1, args.epochs + 1):
        model.train()

        # unfreeze at epoch
        if freeze_until > 0 and ep > freeze_until:
            # only if using resnet (not clip)
            if not args.use_clip and hasattr(model, "image_encoder"):
                for p in model.image_encoder.parameters():
                    p.requires_grad = True
                freeze_until = 0
                print(f"[{mode}] unfreeze image_encoder at epoch {ep}.")

        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            img = batch["image"].to(device)
            y = batch["label"].to(device)
            
            logits = model(input_ids=input_ids, attention_mask=attn, images=img)
            if logit_adjust is not None:
                logits = logits + logit_adjust

            if args.use_focal:
                loss = criterion(logits, y)
            else:
                if args.label_smoothing > 0:
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

        # eval (EMA weights if enabled)
        if ema is not None:
            ema.apply_to(model)
        val_acc, val_mf1 = evaluate(model, val_loader, device, logit_adjust=logit_adjust)
        if ema is not None:
            ema.restore(model)

        print(f"[{mode}] epoch {ep}/{args.epochs}: loss={avg_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_mf1:.4f}")

        # 记录每个 epoch 的结果
        history.append({
            "epoch": int(ep),
            "train_loss": float(avg_loss),
            "val_acc": float(val_acc),
            "val_macro_f1": float(val_mf1),
            "lr_base": float(optimizer.param_groups[0]["lr"]),
            "lr_head": float(optimizer.param_groups[1]["lr"]) if len(optimizer.param_groups) > 1 else float(optimizer.param_groups[0]["lr"]),
        })


        if val_mf1 > best_mf1:
            best_mf1 = val_mf1
            best_acc = val_acc
            no_improve = 0

            if ema is not None:
                ema.apply_to(model)

            save_obj = {
                "model_state": model.state_dict(),
                "cfg": asdict(cfg),
                "text_model": args.text_model,
                "args": vars(args),
            }
            # 把 logit_adjust 存进去
            if logit_adjust is not None:
                save_obj["logit_adjust"] = logit_adjust.detach().cpu()

            torch.save(save_obj, best_path)

            print(f"[{mode}] saved best -> {best_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[{mode}] Early stopping at epoch {ep}, best mf1={best_mf1:.4f}")
                break

    # 保存每个 epoch 的指标历史
    hist_path = os.path.join(args.run_dir, f"history_{mode}.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"[{mode}] saved history -> {hist_path}")

    return {
        "mode": mode,
        "best_macro_f1": float(best_mf1),
        "best_acc": float(best_acc),
        "best_ckpt": best_path,
    }

def main():
    parser = argparse.ArgumentParser()

    # paths
    parser.add_argument("--train_path", type=str, default="train.jsonl")
    parser.add_argument("--val_path", type=str, default="val.jsonl")
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--run_dir", type=str, default="runs")

    # backbone
    parser.add_argument("--text_model", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--use_clip", action="store_true")
    parser.add_argument("--clip_name", type=str, default="openai/clip-vit-base-patch32")

    # training
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head_lr_mul", type=float, default=6.0)  # 重点：head 更大学习率，通常显著提 acc
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--amp", action="store_true")  # 仅 cuda 生效

    # data innovation
    parser.add_argument("--text_clean", type=str, default="basic", choices=["none", "basic", "aggressive"])
    parser.add_argument("--img_aug", type=str, default="strong", choices=["none", "weak", "strong"])

    # fusion innovation
    parser.add_argument("--fusion", type=str, default="gated", choices=["concat", "gated", "cross_attn", "sum"])

    # loss / threshold innovation
    parser.add_argument("--sampler", action="store_true", help="WeightedRandomSampler")
    parser.add_argument("--use_class_weight", action="store_true", help="use sqrt-balanced class weights in loss")
    parser.add_argument("--class_weight_power", type=float, default=0.5)
    parser.add_argument("--use_focal", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=1.2)
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    parser.add_argument("--logit_adjust", action="store_true")
    parser.add_argument("--adjust_tau", type=float, default=1.0)

    # schedule innovation
    parser.add_argument("--freeze_image_epochs", type=int, default=1)

    # EMA
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)

    # ablation
    parser.add_argument("--ablation", action="store_true", help="train text_only + image_only + multimodal")

    args = parser.parse_args()
    seed_everything(args.seed)
    os.makedirs(args.run_dir, exist_ok=True)

    t0 = time.time()
    if args.ablation:
        results = []
        for mode in ["text_only", "image_only", "multimodal"]:
            results.append(train_one_mode(args, mode))
        with open(os.path.join(args.run_dir, "ablation_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("Saved:", os.path.join(args.run_dir, "ablation_results.json"))
    else:
        res = train_one_mode(args, "multimodal")
        with open(os.path.join(args.run_dir, "train_result.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print("Saved:", os.path.join(args.run_dir, "train_result.json"))

    print(f"Done. time={(time.time()-t0):.1f}s")

if __name__ == "__main__":
    main()