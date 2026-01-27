from dataclasses import dataclass
import torch
import torch.nn as nn
import torchvision.models as tvm
from transformers import VisualBertModel

@dataclass
class VisualBertConfigLocal:
    num_classes: int = 3
    dropout: float = 0.2
    mode: str = "multimodal"  # "text_only" | "image_only" | "multimodal"
    freeze_image: bool = False
    visualbert_name: str = "uclanlp/visualbert-vqa-coco-pre"

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

class VisualBertSentimentModel(nn.Module):
    """
    VisualBERT 多模态情感分类：
      - 输入：input_ids, attention_mask, images (B,3,H,W)
      - 输出：logits (B,3)

    图像 -> ResNet50 conv feature map (B,2048,7,7)
         -> flatten 成 49 个 region: (B,49,2048)
         -> 作为 visual_embeds 喂给 VisualBERT
    """
    def __init__(self, cfg: VisualBertConfigLocal):
        super().__init__()
        self.cfg = cfg
        self.mode = cfg.mode

        # 1) VisualBERT backbone
        self.vb = VisualBertModel.from_pretrained(cfg.visualbert_name, use_safetensors=True)
        hidden = self.vb.config.hidden_size

        # 2) image encoder: ResNet50 conv features => 49 regions
        backbone = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
        # 去掉 avgpool + fc，保留到 layer4 输出 (B,2048,7,7)
        self.image_encoder = nn.Sequential(*list(backbone.children())[:-2])

        if cfg.freeze_image:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        # 3) 分类头：用 pooled_output / CLS
        self.head = MLPHead(hidden, cfg.num_classes, cfg.dropout)

    def _encode_image_to_regions(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B,3,H,W)
        return visual_embeds: (B,49,2048)
        """
        feat = self.image_encoder(images)  # (B,2048,7,7)
        b, c, h, w = feat.shape
        feat = feat.flatten(2).transpose(1, 2).contiguous()  # (B,49,2048)
        return feat

    def forward(self, input_ids=None, attention_mask=None, images=None):
        assert input_ids is not None and attention_mask is not None, "need input_ids & attention_mask"
        assert images is not None, "need images"

        B = input_ids.size(0)
        device = input_ids.device

        # image -> region embeds
        if self.mode in ["multimodal", "image_only"]:
            visual_embeds = self._encode_image_to_regions(images)
            visual_attention_mask = torch.ones(
                (B, visual_embeds.size(1)), dtype=torch.long, device=device
            )
        else:
            # text_only：给一个占位 visual token，mask=0 表示不使用视觉信息
            visual_embeds = torch.zeros((B, 1, 2048), dtype=torch.float32, device=device)
            visual_attention_mask = torch.zeros((B, 1), dtype=torch.long, device=device)

        # text handling for ablation
        if self.mode == "image_only":
            # 让文本“不可用”：把 attention_mask 置 0，但保留第一个 token 为 1，避免全 0 造成异常
            attn = torch.zeros_like(attention_mask)
            attn[:, 0] = 1
            attention_mask = attn

        # VisualBERT forward
        out = self.vb(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_attention_mask,
            return_dict=True,
        )

        # pooled_output 优先；没有就用 CLS
        pooled = out.pooler_output
        if pooled is None:
            pooled = out.last_hidden_state[:, 0]

        logits = self.head(pooled)
        return logits