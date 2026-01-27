import os
import json
import csv
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

# Label mapping
LABEL2ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# allow some alias
LABEL_ALIASES = {
    "neg": "negative",
    "neu": "neutral",
    "pos": "positive",
}

HEADER_TOKENS = {"guid", "id", "tag", "label", "sentiment", "text", "post", "image", "img", "img_path", "image_path"}

def _strip_bom(s: str) -> str:
    return s.lstrip("\ufeff").strip()

def _guess_delimiter(first_line: str) -> str:
    # crude but effective: choose delimiter with more hits in header
    comma = first_line.count(",")
    tab = first_line.count("\t")
    if tab > comma:
        return "\t"
    return ","

def _safe_lower(x: Any) -> str:
    return str(x).strip().lower()

def _normalize_label(y: Any) -> str:
    s = _safe_lower(y)
    s = _strip_bom(s)
    if s in LABEL_ALIASES:
        s = LABEL_ALIASES[s]
    return s

@dataclass
class Sample:
    guid: str
    text: str
    label: Optional[str]  # can be None for test_without_label
    image_key: str        # used to build image path

class MultiModalDataset(Dataset):
    """
    Supports:
      - .json / .jsonl: existing behavior (expects fields compatible with your code)
      - .txt / .csv / .tsv:
          must have header, and at least:
            guid (or id) column
            tag/label/sentiment column (optional for test set)
          and optionally:
            post/text column
            image/img/image_path column
        Common format you showed: guid,tag,post
    """

    def __init__(
        self,
        data_path: str,
        image_root: str,
        tokenizer=None,
        image_transform=None,
        max_len: int = 128,
        image_size: int | None = None,
        is_test: bool = False,
        text_clean: str = "none",
        **kwargs,
    ):
        self.data_path = data_path
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_len = max_len
        self.image_size = image_size
        self.is_test = is_test
        self.text_clean = text_clean
        self.train = bool(kwargs.get("train", not is_test))
        self.img_aug = str(kwargs.get("img_aug", "none"))

        if self.image_transform is None:
            size = self.image_size or 224
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)

            if self.train and self.img_aug in ["weak", "strong"]:
                # 训练：带增强
                aug = [
                    T.RandomResizedCrop(size, scale=(0.8, 1.0)),
                    T.RandomHorizontalFlip(p=0.5),
                ]
                if self.img_aug == "strong":
                    aug += [
                        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                        T.RandomGrayscale(p=0.1),
                    ]
                self.image_transform = T.Compose(aug + [
                    T.ToTensor(),
                    T.Normalize(mean=mean, std=std),
                    # strong 时可再加一点随机擦除
                    T.RandomErasing(p=0.25 if self.img_aug == "strong" else 0.1, scale=(0.02, 0.15))
                ])
            else:
                # 验证/测试：确定性
                self.image_transform = T.Compose([
                    T.Resize((size, size)),
                    T.ToTensor(),
                    T.Normalize(mean=mean, std=std),
                ])

        self.samples: List[Sample] = self._load(data_path)

    # Text cleaning 
    def _clean_text(self, s: str) -> str:
        if s is None:
            return ""
        s = str(s)
        if self.text_clean == "none":
            return s
        s = s.strip()
        if self.text_clean == "basic":
            # remove extra whitespace
            s = re.sub(r"\s+", " ", s)
            return s
        if self.text_clean == "aggressive":
            s = re.sub(r"http\S+", "", s)        # remove urls
            s = re.sub(r"@\w+", "", s)           # remove @mentions
            s = re.sub(r"\s+", " ", s).strip()
            return s
        return s
    
    # Label parsing
    def _get_label_id(self, y: Any) -> int:
        if y is None:
            raise ValueError("Label is None but is_test=False. Check your val/train files.")
        s = _normalize_label(y)
        if s in LABEL2ID:
            return LABEL2ID[s]
        raise ValueError(f"Bad label: {y}")

    # Image path building
    def _build_image_path(self, image_key: str) -> str:
        """
        If image_key already looks like a path, join with root if relative.
        Otherwise, treat it as an id and try id.jpg/png/jpeg.
        """
        k = str(image_key).strip()
        if not k:
            # fallback
            k = "0"

        # if already has extension, treat as filename/path
        if os.path.splitext(k)[1].lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            p = k
            if not os.path.isabs(p):
                p = os.path.join(self.image_root, p)
            return p

        # otherwise try common extensions
        for ext in [".jpg", ".png", ".jpeg", ".webp"]:
            p = os.path.join(self.image_root, k + ext)
            if os.path.exists(p):
                return p

        # fallback
        return os.path.join(self.image_root, k + ".jpg")

    # TXT/CSV loader
    def _load_txt_or_csv(self, p: str) -> List[Sample]:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline()
            if not first_line:
                return []
            delim = _guess_delimiter(first_line)
            f.seek(0)

            reader = csv.reader(f, delimiter=delim)
            raw_header = next(reader, None)
            if raw_header is None:
                return []

            header = [_safe_lower(_strip_bom(h)) for h in raw_header]

            # map columns
            def find_col(*candidates: str) -> Optional[int]:
                for c in candidates:
                    c = c.lower()
                    if c in header:
                        return header.index(c)
                return None

            guid_i = find_col("guid", "id")
            label_i = find_col("tag", "label", "sentiment")  # may be None for test
            text_i = find_col("post", "text", "sentence", "content")
            img_i = find_col("image", "img", "image_path", "img_path")

            if guid_i is None:
                raise ValueError(
                    f"[dataset] Cannot find guid/id column in {p}. header={raw_header}"
                )

            samples: List[Sample] = []
            for row in reader:
                if not row:
                    continue
                # pad if short
                if len(row) < len(header):
                    row = row + [""] * (len(header) - len(row))

                guid = _strip_bom(row[guid_i])
                if not guid:
                    continue

                label = None
                if label_i is not None:
                    label = _strip_bom(row[label_i])
                    # skip accidental header-like rows
                    if _safe_lower(label) in HEADER_TOKENS and _safe_lower(guid) in HEADER_TOKENS:
                        continue

                text = ""
                if text_i is not None:
                    text = row[text_i]
                text = self._clean_text(text)

                image_key = guid
                if img_i is not None and row[img_i].strip():
                    image_key = row[img_i].strip()

                samples.append(Sample(guid=guid, text=text, label=label, image_key=image_key))

        return samples

    # JSON/JSONL loader
    def _load_json_or_jsonl(self, p: str) -> List[Sample]:
        ext = os.path.splitext(p)[1].lower()
        samples: List[Sample] = []
        if ext == ".jsonl":
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    guid = str(obj.get("guid", obj.get("id", ""))).strip()
                    if not guid:
                        continue
                    label = obj.get("label", obj.get("tag", obj.get("sentiment", None)))
                    text = obj.get("text", obj.get("post", ""))
                    image_key = obj.get("image", obj.get("img", obj.get("image_path", guid)))
                    samples.append(Sample(guid=guid, text=self._clean_text(text), label=label, image_key=str(image_key)))
        else:  # .json
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            # allow list[dict] or dict with 'data'
            if isinstance(data, dict) and "data" in data:
                data = data["data"]
            if not isinstance(data, list):
                raise ValueError(f"[dataset] json must be list/dict-with-data, got {type(data)}")
            for obj in data:
                guid = str(obj.get("guid", obj.get("id", ""))).strip()
                if not guid:
                    continue
                label = obj.get("label", obj.get("tag", obj.get("sentiment", None)))
                text = obj.get("text", obj.get("post", ""))
                image_key = obj.get("image", obj.get("img", obj.get("image_path", guid)))
                samples.append(Sample(guid=guid, text=self._clean_text(text), label=label, image_key=str(image_key)))
        return samples

    def _load(self, data_path: str) -> List[Sample]:
        ext = os.path.splitext(data_path)[1].lower()
        if ext in [".json", ".jsonl"]:
            return self._load_json_or_jsonl(data_path)
        if ext in [".txt", ".csv", ".tsv"]:
            return self._load_txt_or_csv(data_path)
        raise ValueError("data_path must be .json/.jsonl/.txt/.csv/.tsv")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]

        # label
        y = None
        if not self.is_test:
            y = self._get_label_id(s.label)

        # text tokenize
        if self.tokenizer is not None:
            enc = self.tokenizer(
                s.text,
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
        else:
            input_ids = torch.zeros(self.max_len, dtype=torch.long)
            attention_mask = torch.zeros(self.max_len, dtype=torch.long)

        # image
        img_path = self._build_image_path(s.image_key)
        img = Image.open(img_path).convert("RGB")
        if self.image_transform is not None:
            img = self.image_transform(img)

        out = {
            "guid": s.guid,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": img,
        }
        if not self.is_test:
            out["label"] = torch.tensor(y, dtype=torch.long)
        return out