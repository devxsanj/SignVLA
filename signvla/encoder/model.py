"""Lightweight temporal sign encoder: Bi-GRU over landmark sequences."""
import torch
import torch.nn as nn

from signvla import config


class GestureGRU(nn.Module):
    """(B, T, 126) landmark sequence -> (B, num_classes) logits."""

    def __init__(self, num_classes, input_dim=config.FEATURES_PER_FRAME,
                 hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                          bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, num_classes),
        )

    def forward(self, x):
        out, _ = self.gru(self.input_layer(x))
        return self.head(out.mean(dim=1))  # temporal average pooling


def load_checkpoint(path=config.MODEL_PATH, device="cpu"):
    """-> (model in eval mode, gestures). Refuses checkpoints from the old feature space."""
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    if ckpt.get("feature_version") != config.FEATURE_VERSION:
        raise RuntimeError(
            f"{path} was trained on an older feature space (feature_version="
            f"{ckpt.get('feature_version')}, need {config.FEATURE_VERSION}). "
            "Retrain: python -m scripts.train")
    model = GestureGRU(ckpt["num_classes"], hidden_dim=ckpt["hidden_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), ckpt["gestures"]
