import os
import json
import time
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from dataset import MMEmoDataset, collate_fn
from model import MultiModalSentiment
import matplotlib.pyplot as plt
import seaborn as sns

def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _filter_existing_samples(root: str, df: pd.DataFrame) -> pd.DataFrame:
    data_dir = os.path.join(root, "data")
    txt_ok = df["guid"].apply(lambda g: os.path.exists(os.path.join(data_dir, f"{g}.txt")))
    img_ok = df["guid"].apply(lambda g: os.path.exists(os.path.join(data_dir, f"{g}.jpg")))
    ok = txt_ok & img_ok

    missing = df.loc[~ok, "guid"].tolist()
    if len(missing) > 0:
        print(f"[data-check] drop {len(missing)} samples due to missing files (txt/jpg). Example: {missing[:5]}")
    return df.loc[ok].reset_index(drop=True)

@torch.no_grad()
def evaluate(model, loader, device, args, priors: torch.Tensor):
    """
    Route B: logit adjustment / prior correction
      logits' = logits - tau * log(prior)
    priors: shape [3], order = (neg, neu, pos)
    """
    model.eval()
    y_true, y_pred = [], []

    # 避免 log(0)
    priors = torch.clamp(priors, min=1e-12).to(device)
    log_priors = torch.log(priors)

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        image = batch["image"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask, image)

        # logit adjustment
        if args.logit_adjust:
            # broadcast: [B,3] - [3]
            logits = logits - (args.adjust_tau * log_priors)

        pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        y_pred.extend(pred)
        y_true.extend(labels.cpu().numpy().tolist())

    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")

    report = classification_report(
        y_true,
        y_pred,
        target_names=["negative", "neutral", "positive"],
        digits=4
    )
    cm = confusion_matrix(y_true, y_pred)
    return acc, mf1, report, cm

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma: float = 2.0):
        super().__init__()
        self.weight = weight
        self.gamma = float(gamma)

    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()

def freeze_module(m: nn.Module, freeze: bool):
    for p in m.parameters():
        p.requires_grad = not freeze

def build_optimizer(args, model: nn.Module, mode: str):
    """
    分组学习率：
    - text_encoder: args.lr_text
    - image_encoder: args.lr_img
    - 其余 head: args.lr_head
    """
    params_text, params_img, params_head = [], [], []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("text_encoder."):
            params_text.append(p)
        elif n.startswith("image_encoder."):
            params_img.append(p)
        else:
            params_head.append(p)

    groups = []
    if mode in ["multimodal", "text_only"] and len(params_text) > 0:
        groups.append({"params": params_text, "lr": args.lr_text})
    if mode in ["multimodal", "image_only"] and len(params_img) > 0:
        groups.append({"params": params_img, "lr": args.lr_img})
    if len(params_head) > 0:
        groups.append({"params": params_head, "lr": args.lr_head})

    optimizer = torch.optim.AdamW(groups, weight_decay=args.wd)
    return optimizer

def build_sampler(train_df: pd.DataFrame):
    """
    WeightedRandomSampler：按类别反比概率抽样（用于 neutral 少数类）
    """
    label2idx = {"negative": 0, "neutral": 1, "positive": 2}
    y = train_df["tag"].map(label2idx).values
    class_count = np.bincount(y, minlength=3)
    class_w = 1.0 / np.sqrt(np.maximum(class_count, 1))
    sample_w = class_w[y]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_w, dtype=torch.double),
        num_samples=len(sample_w),
        replacement=True
    )
    return sampler, class_count.tolist(), class_w.tolist()

def _compute_priors_from_train_df(train_df: pd.DataFrame, device):
    """
    返回 priors: torch.Tensor([p_neg, p_neu, p_pos])  (neg, neu, pos)
    用真实训练集分布（未被 sampler 改过的 train_df）计算。
    """
    counts = train_df["tag"].value_counts()
    n = len(train_df)
    p_neg = counts.get("negative", 0) / max(n, 1)
    p_neu = counts.get("neutral", 0) / max(n, 1)
    p_pos = counts.get("positive", 0) / max(n, 1)
    priors = torch.tensor([p_neg, p_neu, p_pos], dtype=torch.float, device=device)
    return priors, counts

def train_one_mode(args, mode: str):
    seed_all(args.seed)
    device = get_device()

    root = os.path.abspath(os.path.join(args.code_dir, "..", "project5"))  # project5/

    # 1) 严格以 train.txt 为准
    train_txt_path = os.path.join(root, "train.txt")
    df = pd.read_csv(train_txt_path, dtype={"guid": str, "tag": str})
    df["guid"] = df["guid"].astype(str).str.strip()
    df["tag"] = df["tag"].astype(str).str.strip().str.lower()

    # 2) 去重 guid（冗余）
    dup_cnt = df.duplicated(subset=["guid"]).sum()
    if dup_cnt > 0:
        print(f"[label-check] train.txt has {dup_cnt} duplicated guid rows -> keep first occurrence.")
    df = df.drop_duplicates(subset=["guid"], keep="first").reset_index(drop=True)

    # 3) 标签合法性检查
    valid = {"negative", "neutral", "positive"}
    bad = df.loc[~df["tag"].isin(valid)]
    if len(bad) > 0:
        print("[label-check] invalid tags examples:\n", bad.head())
        df = df[df["tag"].isin(valid)].reset_index(drop=True)

    # 4) 严格只用 train.txt 的 guid，同时保证 data/ 文件存在
    df = _filter_existing_samples(root, df)

    # stratified split
    train_df, val_df = train_test_split(
        df,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=df["tag"],
    )

    # --- priors（用于 logit adjust） ---
    priors, counts = _compute_priors_from_train_df(train_df, device)
    if args.logit_adjust:
        print(f"[{mode}] logit_adjust enabled. priors(neg,neu,pos)={priors.detach().cpu().numpy().round(4).tolist()} tau={args.adjust_tau}")

    tokenizer = AutoTokenizer.from_pretrained(args.text_model)

    train_set = MMEmoDataset(
        root, train_df, tokenizer,
        max_len=args.max_len,
        image_size=args.image_size,
        is_train=True,
        clean_text_flag=args.clean_text,
    )
    val_set = MMEmoDataset(
        root, val_df, tokenizer,
        max_len=args.max_len,
        image_size=args.image_size,
        is_train=False,
        clean_text_flag=args.clean_text,
    )

    # sampler（训练集）
    if args.use_sampler:
        sampler, class_count, class_w = build_sampler(train_df)
        print(f"[{mode}] sampler enabled. class_count={class_count}, inv_class_weight={class_w}")
        train_loader = DataLoader(
            train_set,
            batch_size=args.bs,
            sampler=sampler,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
            prefetch_factor=2,
            collate_fn=collate_fn
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=args.bs,
            shuffle=True,
            num_workers=2,
            persistent_workers=True,
            prefetch_factor=2,
            collate_fn=collate_fn
        )

    val_loader = DataLoader(
        val_set,
        batch_size=args.bs,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate_fn
    )

    model = MultiModalSentiment(
        text_model_name=args.text_model,
        mode=mode,
        dropout=args.dropout,
        num_classes=3,
    ).to(device)

    # 训练前冻结 image_encoder 若指定
    if mode in ["multimodal", "image_only"] and args.freeze_img_epochs > 0:
        freeze_module(model.image_encoder, freeze=True)
        print(f"[{mode}] freeze image_encoder for first {args.freeze_img_epochs} epoch(s).")

    # 类别权重（仍保留，用于 CE/Focal）
    total = counts.sum()
    neg = counts.get("negative", 1)
    neu = counts.get("neutral", 1)
    pos = counts.get("positive", 1)

    # sqrt-balanced
    w_neg = (total / (3.0 * neg)) ** 0.5
    w_neu = (total / (3.0 * neu)) ** 0.5
    w_pos = (total / (3.0 * pos)) ** 0.5
    class_weights = torch.tensor([w_neg, w_neu, w_pos], dtype=torch.float, device=device)

    print(f"[{mode}] class counts: {dict(counts)}")
    print(f"[{mode}] class weights (sqrt-balanced): {class_weights.tolist()}  (neg, neu, pos)")

    # loss
    if args.use_focal:
        criterion = FocalLoss(weight=class_weights, gamma=args.focal_gamma)
        print(f"[{mode}] use FocalLoss(gamma={args.focal_gamma}).")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"[{mode}] use CrossEntropyLoss(weighted).")

    # optimizer（分组学习率）
    optimizer = build_optimizer(args, model, mode)

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup),
        num_training_steps=total_steps
    )

    os.makedirs(args.run_dir, exist_ok=True)
    best_mf1 = -1.0
    best_path = os.path.join(args.run_dir, f"best_{mode}.pt")

    patience = args.patience
    no_improve = 0
    best_ep = -1

    for ep in range(1, args.epochs + 1):
        # 到点解冻 image_encoder
        if mode in ["multimodal", "image_only"] and args.freeze_img_epochs > 0 and ep == args.freeze_img_epochs + 1:
            freeze_module(model.image_encoder, freeze=False)
            print(f"[{mode}] unfreeze image_encoder at epoch {ep}.")

            optimizer = build_optimizer(args, model, mode)
            total_steps = len(train_loader) * (args.epochs - ep + 1)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(total_steps * args.warmup),
                num_training_steps=total_steps
            )

        model.train()
        pbar = tqdm(train_loader, desc=f"[{mode}] epoch {ep}/{args.epochs}")
        losses = []

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            image = batch["image"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, attention_mask, image)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())
            pbar.set_postfix(loss=float(np.mean(losses)))

        acc, mf1, report, cm = evaluate(model, val_loader, device, args, priors)
        tag = " (logit_adjust)" if args.logit_adjust else ""
        print(f"[{mode}] epoch {ep}: val_acc={acc:.4f} val_macro_f1={mf1:.4f}{tag}")
        print(report)

        if mf1 > best_mf1:
            best_mf1 = mf1
            no_improve = 0
            best_ep = ep

            torch.save(
                {
                    "model": model.state_dict(),
                    "mode": mode,
                    "text_model": args.text_model,
                    "clean_text": bool(args.clean_text),
                    "logit_adjust": bool(args.logit_adjust),
                    "adjust_tau": float(args.adjust_tau),
                    "priors_neg_neu_pos": priors.detach().cpu().tolist(),
                },
                best_path
            )
            print(f"[{mode}] saved best -> {best_path}")

            labels_name = ["negative", "neutral", "positive"]
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=labels_name, yticklabels=labels_name)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"Confusion Matrix ({mode})")
            fig_path = os.path.join(args.run_dir, f"confusion_matrix_{mode}.png")
            plt.tight_layout()
            plt.savefig(fig_path)
            plt.close()
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[{mode}] Early stopping at epoch {ep}, best mf1={best_mf1:.4f}")
                break

    return {
        "mode": mode,
        "best_macro_f1": float(best_mf1),
        "best_epoch": best_ep,
        "best_ckpt": best_path,
        "clean_text": bool(args.clean_text),
        "use_sampler": bool(args.use_sampler),
        "use_focal": bool(args.use_focal),
        "logit_adjust": bool(args.logit_adjust),
        "adjust_tau": float(args.adjust_tau),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code_dir", type=str, default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--text_model", type=str, default="bert-base-multilingual-cased")

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--bs", type=int, default=16)

    # 分组学习率
    parser.add_argument("--lr_text", type=float, default=1e-5)
    parser.add_argument("--lr_img", type=float, default=1e-5)
    parser.add_argument("--lr_head", type=float, default=5e-5)

    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=224)

    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--run_dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"))

    # 数据&loss&采样
    parser.add_argument("--clean_text", action="store_true", help="enable text cleaning")
    parser.add_argument("--no_clean_text", action="store_true", help="disable text cleaning (override)")

    parser.add_argument("--use_sampler", action="store_true", help="enable WeightedRandomSampler for train")
    parser.add_argument("--no_sampler", action="store_true", help="disable sampler (override)")

    parser.add_argument("--use_focal", action="store_true", help="enable FocalLoss")
    parser.add_argument("--focal_gamma", type=float, default=2.0)

    parser.add_argument("--patience", type=int, default=4, help="early stopping patience")
    parser.add_argument("--freeze_img_epochs", type=int, default=1, help="freeze image encoder for N epochs")

    # --- Route B: logit adjustment ---
    parser.add_argument("--logit_adjust", action="store_true", help="apply prior correction on logits during eval")
    parser.add_argument("--adjust_tau", type=float, default=1.0, help="strength for logit adjustment")

    parser.add_argument("--ablation", action="store_true", help="train text_only + image_only + multimodal")
    args = parser.parse_args()

    # 处理 override
    if args.no_clean_text:
        args.clean_text = False
    else:
        # 默认开启清洗
        args.clean_text = True

    if args.no_sampler:
        args.use_sampler = False
    else:
        args.use_sampler = True

    t0 = time.time()
    os.makedirs(args.run_dir, exist_ok=True)

    if args.ablation:
        results = []
        for mode in ["text_only", "image_only", "multimodal"]:
            results.append(train_one_mode(args, mode))
        with open(os.path.join(args.run_dir, "ablation_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("\nSaved:", os.path.join(args.run_dir, "ablation_results.json"))
    else:
        res = train_one_mode(args, "multimodal")
        with open(os.path.join(args.run_dir, "train_result.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print("\nSaved:", os.path.join(args.run_dir, "train_result.json"))

    print(f"\nDone. time={(time.time() - t0):.1f}s")

if __name__ == "__main__":
    main()