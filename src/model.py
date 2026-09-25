"""
PyTorch Neural Network Models for ISL Gesture Sequence Recognition
Supports:
1. GestureGRU (Bi-directional GRU with LayerNorm and Linear Classifier)
2. GestureTCN (Temporal 1D-CNN + Global Pooling)
"""
import torch
import torch.nn as nn


class GestureGRU(nn.Module):
    """
    Recurrent sequence classifier designed for temporal landmark series.
    Input shape: (batch_size, seq_len=30, features=126)
    Output shape: (batch_size, num_classes)
    """
    def __init__(self, input_dim=126, hidden_dim=128, num_layers=2, num_classes=6, dropout=0.3):
        super(GestureGRU, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, T, D)
        features = self.input_layer(x)          # (B, T, hidden_dim)
        gru_out, _ = self.gru(features)         # (B, T, hidden_dim * 2)
        # Take temporal average pooling over sequence length
        pooled = torch.mean(gru_out, dim=1)     # (B, hidden_dim * 2)
        logits = self.fc(pooled)                # (B, num_classes)
        return logits


def get_model(num_classes, input_dim=126, hidden_dim=128):
    return GestureGRU(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
