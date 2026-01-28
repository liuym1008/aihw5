import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="path to val pred csv with gt")
    parser.add_argument("--title", type=str, default="Confusion Matrix")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    if "gt" not in df.columns or "pred" not in df.columns:
        raise ValueError(f"CSV must contain columns: gt, pred. found={list(df.columns)}")

    y_true = df["gt"]
    y_pred = df["pred"]

    # 兼容两种格式：
    # 1) gt/pred 是字符串：negative/neutral/positive
    # 2) gt/pred 是数字：0/1/2
    label_names = ["negative", "neutral", "positive"]

    if y_true.dtype == object or y_pred.dtype == object:
        # 字符串标签
        labels = label_names
    else:
        # 数字标签
        labels = [0, 1, 2]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.xticks(range(len(labels)), label_names, rotation=20)
    plt.yticks(range(len(labels)), label_names)
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title(args.title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()