import torch
import torch.nn as nn
from models.har_lstm.model import HARLSTM

fusion_model = model = HARLSTM(
            num_channels=6,
            num_positions=7,
            positions=["chest", "head", "shin", "thigh", "upperarm", "waist", "forearm"],
            num_classes=8,
            hidden_dim=64,
            dropout=0.5
        )



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


#fusion_model.load_state_dict(torch.load("../human_activity/artifacts/models/har_HARLSTM_bestmodel93.pth"))

#encoder = fusion_model.encoder 

class SingleHAR(nn.Module):
    def __init__(self, num_classes=8, hidden_dim=64):
        super().__init__()
        self.encoder = PositionEncoder6(hidden_dim=hidden_dim)
        #self.encoder = encoder
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        emb = self.encoder(x)        # [B, 64]
        logits = self.classifier(emb)
        return logits