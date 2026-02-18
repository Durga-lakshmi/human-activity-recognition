# import torch
# import torch.nn as nn

# class HAPTLSTM(nn.Module):
#     def __init__(self, input_dim=6, hidden_dim=128, num_classes=12):
#         super().__init__()

#         self.lstm = nn.LSTM(
#             input_size=input_dim,
#             hidden_size=hidden_dim,
#             num_layers=1,
#             batch_first=True
#         )

#         self.classifier = nn.Linear(hidden_dim, num_classes)

#     def forward(self, x):
#         # x: [B, T, 6]
#         out, (h_n, _) = self.lstm(x)

#         # h_n: [1, B, hidden_dim]
#         last_hidden = h_n[-1]          # [B, hidden_dim]

#         logits = self.classifier(last_hidden)
#         return logits

import torch
import torch.nn as nn
import torch.nn.functional as F

class HAPTLSTM(nn.Module):
    def __init__(self, input_dim=6, num_classes=12, window_size=128):
        super().__init__()
        # Input: (B, T, C) → (B, C, T)
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x: (B, T, C) → (B, C, T)
        x = x.permute(0,2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x).squeeze(-1)  # (B, 128)
        x = self.fc(x)
        return x
