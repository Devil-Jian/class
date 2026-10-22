"""Real dual-phase feature fusion precedes gated Attention-MIL patient pooling."""
from __future__ import annotations

import torch
from torch import nn
import timm


MODEL_NAMES = {
    "resnet18_mil": "resnet18",
    "deit_mil": "deit_small_patch16_224",
    "dinov2s_mil": "vit_small_patch14_dinov2.lvd142m",
}


class DualPhaseMIL(nn.Module):
    def __init__(self, model_name="dinov2s_mil", num_classes=3,
                 pretrained=False, embed_dim=256, phase_mode="dual",
                 freeze_encoder=False, encode_chunk=16):
        super().__init__()
        if model_name not in MODEL_NAMES:
            raise ValueError(f"Unknown model: {model_name}")
        if phase_mode not in {"dual", "ncct"}:
            raise ValueError("phase_mode must be dual or ncct")
        self.model_name = model_name
        self.phase_mode = phase_mode
        self.freeze_encoder = freeze_encoder
        self.encode_chunk = encode_chunk
        self.encoder = timm.create_model(
            MODEL_NAMES[model_name], pretrained=pretrained, num_classes=0,
        )
        feature_dim = self.encoder.num_features
        input_dim = feature_dim * (2 if phase_mode == "dual" else 1)
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, embed_dim), nn.ReLU(), nn.Dropout(0.1)
        )
        self.attn_v = nn.Linear(embed_dim, embed_dim // 2)
        self.attn_u = nn.Linear(embed_dim, embed_dim // 2)
        self.attn_w = nn.Linear(embed_dim // 2, 1, bias=False)
        self.classifier = nn.Linear(embed_dim, num_classes)
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def _encode(self, images):
        b, k, c, h, w = images.shape
        flat = images.reshape(b * k, c, h, w)
        outputs = []
        for chunk in flat.split(self.encode_chunk, dim=0):
            # Match the 3-channel ImageNet-pretrained feature extractor input.
            if chunk.shape[1] == 1:
                chunk = chunk.expand(-1, 3, -1, -1)
            if self.freeze_encoder:
                with torch.no_grad():
                    y = self.encoder(chunk)
            else:
                y = self.encoder(chunk)
            if y.ndim > 2:
                y = y.mean(tuple(range(2, y.ndim)))
            outputs.append(y)
        return torch.cat(outputs, dim=0).reshape(b, k, -1)

    def forward(self, ap, pvp, valid_mask=None):
        ap_features = self._encode(ap)
        if self.phase_mode == "dual":
            pvp_features = self._encode(pvp)
            features = torch.cat([ap_features, pvp_features], dim=-1)
        else:
            features = ap_features
        fused = self.fusion(features)
        logits = self.attn_w(torch.tanh(self.attn_v(fused)) *
                             torch.sigmoid(self.attn_u(fused))).squeeze(-1)
        if valid_mask is not None:
            if not torch.all(valid_mask.any(dim=1)):
                raise ValueError("Empty MIL bag")
            logits = logits.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1)
        pooled = (fused * weights.unsqueeze(-1)).sum(dim=1)
        return {"logits": self.classifier(pooled), "attention_scores": weights,
                "patient_features": pooled}
