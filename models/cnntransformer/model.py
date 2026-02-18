import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLightTransformer(nn.Module):
    """
    CNN for local patterns + lightweight Transformer for temporal semantics
    Input : [B, T, C]
    Output: [B, num_classes]
    """
    def __init__(
        self,
        in_channels=6,
        num_classes=6,
        d_model=128,
        nhead=4,
        num_layers=2,
        dropout=0.3
    ):
        super().__init__()

        # CNN 特征提取
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.cls_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: [B, T, C] → [B, C, T]
        x = x.transpose(1, 2)
        x = self.cnn(x)              # [B, d_model, T]
        x = x.transpose(1, 2)        # [B, T, d_model]

        x = self.transformer(x)      # [B, T, d_model]
        x = x.mean(dim=1)            # temporal pooling

        return self.cls_head(x)