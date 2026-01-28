import json
import argparse
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=str, required=True, help="path to history json")
    parser.add_argument("--title", type=str, default="Training Curves")
    args = parser.parse_args()

    with open(args.history, "r") as f:
        hist = json.load(f)

    # 1) dict: {"train_loss": [...], "val_acc": [...], ...}
    # 2) list: [{"epoch": 1, "train_loss": x, "val_acc": y, ...}, ...]
    if isinstance(hist, list):
        train_loss = [h["train_loss"] for h in hist]
        val_loss = [h.get("val_loss") for h in hist]  # 允许不存在
        val_acc = [h["val_acc"] for h in hist]
        val_mf1 = [h["val_macro_f1"] for h in hist]
    else:
        train_loss = hist.get("train_loss")
        val_loss = hist.get("val_loss")
        val_acc = hist.get("val_acc")
        val_mf1 = hist.get("val_macro_f1")

    if train_loss is None or val_acc is None or val_mf1 is None:
        raise ValueError("history json missing required fields")

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_loss, label="Train Loss")
    if val_loss is not None:
        plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, val_acc, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Val Acc")

    plt.subplot(1, 3, 3)
    plt.plot(epochs, val_mf1, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Val Macro-F1")

    plt.suptitle(args.title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()