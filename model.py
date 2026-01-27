from dataclasses import dataclass
from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from transformers import AutoModel, CLIPModel
import math

@dataclass
class ModelConfig:
    num_classes: int = 3
    dropout: float = 0.2
    proj_dim: int = 256
    fusion: str = "gated"     # "concat" | "gated" | "cross_attn" | "sum"
    mode: str = "multimodal"  # "text_only" | "image_only" | "multimodal"
    use_clip: bool = False
    clip_name: str = "openai/clip-vit-base-patch32"
    freeze_image: bool = False

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

class CrossAttnFusion(nn.Module):
    """
    MPS 稳定版 Cross-Attn（不使用 nn.MultiheadAttention，避免其 backward 里 view/stride 报错）
    用点积得到一个 gate（alpha），再在 text/img 间做加权融合。
    """
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, text_feat: torch.Tensor, img_feat: torch.Tensor):
        # text_feat, img_feat: (B, D)
        q = self.q(text_feat)
        k = self.k(img_feat)
        v = self.v(img_feat)

        # 点积得到一个标量 gate：alpha in (0,1)，shape (B,1)
        score = (q * k).sum(dim=-1, keepdim=True) / math.sqrt(q.size(-1))
        alpha = torch.sigmoid(score)

        # 融合：alpha 更像“图像相关性”权重
        out = alpha * v + (1.0 - alpha) * q
        out = self.out(out)
        out = self.drop(out)

        out = out + self.ffn(out)
        return out

class MultiModalSentimentModel(nn.Module):
    def __init__(self, text_model_name: str, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.mode = cfg.mode

        self.use_clip = cfg.use_clip
        self.clip_name = cfg.clip_name

        if self.use_clip:
            self.clip = CLIPModel.from_pretrained(self.clip_name)

            # 冻结 CLIP，避免 MPS 下 backward 进入 CLIP 触发 view/stride 报错
            for p in self.clip.parameters():
                p.requires_grad = False
            self.clip.eval()

            # CLIP 的 get_text_features/get_image_features 输出维度是 projection_dim
            clip_dim = self.clip.config.projection_dim
            text_dim = clip_dim
            img_dim = clip_dim
        else:
            self.text_encoder = AutoModel.from_pretrained(text_model_name)
            text_dim = self.text_encoder.config.hidden_size

            # resnet50
            backbone = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
            self.image_encoder = nn.Sequential(*list(backbone.children())[:-1])  # (B,2048,1,1)
            img_dim = 2048

            if cfg.freeze_image:
                for p in self.image_encoder.parameters():
                    p.requires_grad = False

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, cfg.proj_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, cfg.proj_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        # fusion
        self.fusion = cfg.fusion
        if self.fusion == "concat":
            head_in = cfg.proj_dim * 2
            self.head = MLPHead(head_in, cfg.num_classes, cfg.dropout)
        elif self.fusion == "sum":
            head_in = cfg.proj_dim
            self.head = MLPHead(head_in, cfg.num_classes, cfg.dropout)
        elif self.fusion == "gated":
            self.gate = nn.Sequential(
                nn.Linear(cfg.proj_dim * 2, cfg.proj_dim),
                nn.GELU(),
                nn.Linear(cfg.proj_dim, cfg.proj_dim),
                nn.Sigmoid(),
            )
            head_in = cfg.proj_dim
            self.head = MLPHead(head_in, cfg.num_classes, cfg.dropout)
        elif self.fusion == "cross_attn":
            self.cross = CrossAttnFusion(cfg.proj_dim, dropout=cfg.dropout)
            head_in = cfg.proj_dim
            self.head = MLPHead(head_in, cfg.num_classes, cfg.dropout)
        else:
            raise ValueError(f"Unknown fusion: {self.fusion}")

    def encode_text(self, input_ids, attention_mask):
        if self.use_clip:
            self.clip.eval()
            with torch.no_grad():
                out = self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
            return out.detach()

        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # CLS pooled
        return out.last_hidden_state[:, 0]

    def encode_image(self, images):
        if self.use_clip:
            # 冻结 CLIP 的反向传播
            self.clip.eval()
            with torch.no_grad():
                out = self.clip.get_image_features(pixel_values=images)
            return out.detach()
        feat = self.image_encoder(images).flatten(1)  # (B,2048)
        return feat

    def forward(self, input_ids=None, attention_mask=None, images=None):
        mode = self.mode

        text_feat = None
        img_feat = None

        if mode in ["multimodal", "text_only"]:
            t = self.encode_text(input_ids, attention_mask)
            text_feat = self.text_proj(t)

        if mode in ["multimodal", "image_only"]:
            v = self.encode_image(images)
            img_feat = self.img_proj(v)

        if mode == "text_only":
            fused = text_feat
        elif mode == "image_only":
            fused = img_feat
        else:
            if self.fusion == "concat":
                fused = torch.cat([text_feat, img_feat], dim=-1)
            elif self.fusion == "sum":
                fused = text_feat + img_feat
            elif self.fusion == "gated":
                g = self.gate(torch.cat([text_feat, img_feat], dim=-1))
                fused = g * text_feat + (1 - g) * img_feat
            elif self.fusion == "cross_attn":
                fused = self.cross(text_feat, img_feat)
            else:
                raise ValueError(f"Unknown fusion: {self.fusion}")

        logits = self.head(fused)
        return logits