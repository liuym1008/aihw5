from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
from transformers import BlipForImageTextRetrieval

@dataclass
class BlipConfigLocal:
    blip_name: str = "Salesforce/blip-itm-base-coco"
    num_classes: int = 3
    dropout: float = 0.1
    pool: str = "cls"  # "cls" | "mean"
    use_safetensors: bool = True

class MLPHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)

class BlipSentimentModel(nn.Module):
    """
    BLIP Retrieval/ITM backbone -> (text_feat + image_feat) -> MLP head
    兼容不同 transformers 版本返回字段名差异
    """
    def __init__(self, cfg: BlipConfigLocal):
        super().__init__()
        self.cfg = cfg

        self.blip = BlipForImageTextRetrieval.from_pretrained(
            cfg.blip_name,
            use_safetensors=cfg.use_safetensors,
        )

        # 先不固定输入维度：第一次 forward 看到 feat.shape[-1] 再创建 head
        self.head = None

    def freeze_vision(self, freeze: bool = True):
        # 不同版本命名可能略有不同，这里做两层兼容
        vision = getattr(self.blip, "vision_model", None)
        if vision is None and hasattr(self.blip, "blip"):
            vision = getattr(self.blip.blip, "vision_model", None)
        if vision is not None:
            for p in vision.parameters():
                p.requires_grad = not freeze

    def _pool_text(self, hs, attention_mask):
        if self.cfg.pool == "cls":
            return hs[:, 0]
        mask = attention_mask.unsqueeze(-1).to(hs.dtype)
        return (hs * mask).sum(dim=1) / (mask.sum(dim=1).clamp_min(1.0))

    def forward(self, input_ids, attention_mask, images):
        out = self.blip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=images,
            return_dict=True,
        )

        # transformers 版本返回 keys: ['itm_score', 'last_hidden_state', 'question_embeds']
        txt = out.question_embeds          
        itm = out.itm_score                

        # 把 txt pool 成 (B, D)
        if txt.dim() == 3:
            if self.cfg.pool == "cls":
                txt = txt[:, 0, :]  # (B, D) 取第一个 token
            else:
                # mean pooling（用 attention_mask 去 padding）
                mask = attention_mask.unsqueeze(-1).to(txt.dtype)  # (B, L, 1)
                txt = (txt * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)  # (B, D)

        feat = torch.cat([txt, itm], dim=-1)  # (B, D+2)

        if self.head is None:
            self.head = MLPHead(feat.size(-1), self.cfg.num_classes, self.cfg.dropout).to(feat.device)

        return self.head(feat)

    def export_config(self):
        return asdict(self.cfg)