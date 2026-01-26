import os
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def clean_text(s: str) -> str:
    """
    轻量文本清洗：适合情感分类，不会过度破坏语义。
    - URL -> <URL>
    - @user -> <USER>
    - #tag -> 'hashtag tag'
    - 多空白折叠
    """
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\n", " ").replace("\t", " ")
    s = re.sub(r"http\S+|www\.\S+", " <URL> ", s)
    s = re.sub(r"@\w+", " <USER> ", s)
    s = re.sub(r"#(\w+)", r" hashtag \1 ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

@dataclass
class Sample:
    guid: str
    text: str
    label: Optional[int]  # None for test

class MMEmoDataset(Dataset):
    def __init__(
        self,
        root_dir: str,             # project5/
        dataframe: pd.DataFrame,   # columns: guid, tag (tag can be None)
        tokenizer,
        max_len: int = 128,
        image_size: int = 224,
        is_train: bool = False,
        clean_text_flag: bool = True,
    ):
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, "data")
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.image_size = image_size
        self.clean_text_flag = bool(clean_text_flag)

        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        val_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.img_tf = train_tf if is_train else val_tf

    def __len__(self):
        return len(self.df)

    def _read_text(self, guid: str) -> str:
        path = os.path.join(self.data_dir, f"{guid}.txt")
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read().strip()
        except FileNotFoundError:
            txt = ""
        return clean_text(txt) if self.clean_text_flag else txt

    def _read_image(self, guid: str) -> torch.Tensor:
        path = os.path.join(self.data_dir, f"{guid}.jpg")
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.image_size, self.image_size))
        return self.img_tf(img)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        guid = str(row["guid"])

        text = self._read_text(guid)
        image = self._read_image(guid)

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            "guid": guid,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "image": image,
        }

        tag = row.get("tag", None)
        if tag is None or (isinstance(tag, str) and tag.lower() == "null"):
            item["label"] = None
        else:
            tag_str = str(tag).strip().lower()
            item["label"] = int(LABEL2ID[tag_str])

        return item

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    guids = [b["guid"] for b in batch]
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    images = torch.stack([b["image"] for b in batch], dim=0)

    labels = [b["label"] for b in batch]
    if all(x is None for x in labels):
        labels_t = None
    else:
        labels_t = torch.tensor([int(x) for x in labels], dtype=torch.long)

    return {
        "guid": guids,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "image": images,
        "labels": labels_t,
    }