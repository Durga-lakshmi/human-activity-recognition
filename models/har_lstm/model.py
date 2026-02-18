import torch
import torch.nn as nn

class PositionEncoder6(nn.Module):
    def __init__(self, input_channels, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, hidden_dim, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [B, T, C] → [B, C, T]
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)  # [B, hidden_dim]
        return x


class HARLSTM(nn.Module):
    def __init__(self, num_channels, num_positions, positions, num_classes, hidden_dim, dropout):
        super().__init__()
        self.num_positions = num_positions
        self.positions = positions
        self.encoder = PositionEncoder6(input_channels=num_channels,hidden_dim=hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_positions, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_dict):
        embeddings = []
        for pos in self.positions:
            emb = self.encoder(x_dict[pos])
            embeddings.append(emb)
        fused = torch.cat(embeddings, dim=1)  # [B, hidden_dim*num_positions]
        logits = self.fusion(fused)
        return logits
