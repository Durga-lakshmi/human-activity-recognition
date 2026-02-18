import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalCNN(nn.Module):
    """
    Input : [B, T, C]  (e.g. T=128, C=6)
    Output: [B, num_classes]
    """
    def __init__(self, in_channels=6, num_classes=6, dropout=0.3):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: [B, T, C] → [B, C, T]
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)