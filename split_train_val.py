import argparse
import csv
import random
import os

def guess_delimiter(first_line: str) -> str:
    return "\t" if first_line.count("\t") > first_line.count(",") else ","

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="../project5/train.txt", help="path to original train.txt")
    parser.add_argument("--train_out", type=str, default="../project5/train_split.txt", help="output train split path")
    parser.add_argument("--val_out", type=str, default="../project5/val_split.txt", help="output val split path")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="validation ratio")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--shuffle", action="store_true", help="shuffle before split (recommended)")
    args = parser.parse_args()

    random.seed(args.seed)

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"input file not found: {args.input}")

    # read all
    with open(args.input, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
        if not first_line:
            raise ValueError("Empty input file.")
        delim = guess_delimiter(first_line)
        f.seek(0)

        reader = csv.reader(f, delimiter=delim)
        header = next(reader, None)
        if header is None:
            raise ValueError("No header found in input file.")

        rows = [row for row in reader if row and any(cell.strip() for cell in row)]

    if args.shuffle:
        random.shuffle(rows)

    n = len(rows)
    n_val = int(n * args.val_ratio)
    n_train = n - n_val

    train_rows = rows[:n_train]
    val_rows = rows[n_train:]

    # write (keep same delimiter and header)
    os.makedirs(os.path.dirname(os.path.abspath(args.train_out)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.val_out)), exist_ok=True)

    with open(args.train_out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=delim)
        w.writerow(header)
        w.writerows(train_rows)

    with open(args.val_out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=delim)
        w.writerow(header)
        w.writerows(val_rows)

    print(f"Done split: total={n}, train={len(train_rows)}, val={len(val_rows)}")
    print(f"train_out={args.train_out}")
    print(f"val_out={args.val_out}")

if __name__ == "__main__":
    main()