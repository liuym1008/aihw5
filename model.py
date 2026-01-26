import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision.models import resnet50, ResNet50_Weights

class FusionHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)

class MultiModalSentiment(nn.Module):
    """
    mode:
      - "multimodal": text + image gated fusion
      - "text_only":  only text
      - "image_only": only image
    """
    def __init__(
        self,
        text_model_name: str = "bert-base-multilingual-cased",
        mode: str = "multimodal",
        num_classes: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        assert mode in ["multimodal", "text_only", "image_only"]
        self.mode = mode

        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_encoder.config.hidden_size

        # Image encoder (ResNet50)
        self.image_encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
        img_dim = self.image_encoder.fc.in_features
        self.image_encoder.fc = nn.Identity()

        if mode == "multimodal":
            h = 512
            self.proj_t = nn.Sequential(
                nn.Linear(text_dim, h),
                nn.Dropout(dropout),
            )
            self.proj_i = nn.Sequential(
                nn.Linear(img_dim, h),
                nn.Dropout(dropout),
            )
            self.gate = nn.Sequential(
                nn.Linear(2 * h, h),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(h, h),
                nn.Sigmoid(),
            )
            self.classifier = FusionHead(h, num_classes=num_classes, dropout=dropout)
        elif mode == "text_only":
            self.classifier = FusionHead(text_dim, num_classes=num_classes, dropout=dropout)
        else:  # image_only
            self.classifier = FusionHead(img_dim, num_classes=num_classes, dropout=dropout)

    def forward(self, input_ids, attention_mask, image):
        feats = []

        if self.mode in ["multimodal", "text_only"]:
            out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_feat = out.last_hidden_state[:, 0, :]  # CLS
            feats.append(text_feat)

        if self.mode in ["multimodal", "image_only"]:
            img_feat = self.image_encoder(image)
            feats.append(img_feat)

        if self.mode == "multimodal":
            text_feat, img_feat = feats[0], feats[1]
            t = self.proj_t(text_feat)
            i = self.proj_i(img_feat)
            g = self.gate(torch.cat([t, i], dim=1))
            fused = g * t + (1.0 - g) * i
            logits = self.classifier(fused)
        else:
            logits = self.classifier(feats[0])

        return logits