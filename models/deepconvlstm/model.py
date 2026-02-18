import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepConvLSTM(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
        conv_channels: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()

        # ---- CNN feature extractor ----
        self.conv1 = nn.Conv1d(in_channels, conv_channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2)

        self.bn1 = nn.BatchNorm1d(conv_channels)
        self.bn2 = nn.BatchNorm1d(conv_channels)
        self.bn3 = nn.BatchNorm1d(conv_channels)
        self.bn4 = nn.BatchNorm1d(conv_channels)

        # ---- LSTM temporal modeling ----
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # ---- Classifier ----
        self.fc = nn.Linear(lstm_hidden, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, T, C)
        """
        # CNN expects (B, C, T)
        x = x.permute(0, 2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        # Back to (B, T, F)
        x = x.permute(0, 2, 1)

        # LSTM
        out, _ = self.lstm(x)  # (B, T, H)

        # Take last timestep
        out = out[:, -1, :]    # (B, H)
        out = self.dropout(out)

        logits = self.fc(out)
        return logits



class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, padding=2):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

        # 如果 in_ch != out_ch，用 1x1 conv 做通道对齐，保证可以残差相加
        self.proj = None
        if in_ch != out_ch:
            self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        # x: (B, C, T)
        residual = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)

        if self.proj is not None:
            residual = self.proj(residual)

        out = out + residual
        out = self.act(out)
        return out


class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.att_fc = nn.Linear(in_dim, in_dim)
        self.v = nn.Linear(in_dim, 1, bias=False)

    def forward(self, h):
        # h: (B, T, H)
        scores = torch.tanh(self.att_fc(h))          # (B, T, H)
        scores = self.v(scores).squeeze(-1)          # (B, T)
        alpha = torch.softmax(scores, dim=-1)        # (B, T)

        context = torch.sum(h * alpha.unsqueeze(-1), dim=1)  # (B, H)
        return context, alpha


class DeepConvBiLSTMAtt(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,    # HAPT: acc+gyro -> 6
        num_classes: int = 12,   # 12 fine classes
        conv_channels: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = True,
        use_attention: bool = True,
    ):
        super().__init__()

        # ---- CNN feature extractor (带残差) ----
        self.conv_block1 = ConvBlock1D(in_channels, conv_channels, kernel_size=5, padding=2)
        self.conv_block2 = ConvBlock1D(conv_channels, conv_channels, kernel_size=5, padding=2)
        self.conv_block3 = ConvBlock1D(conv_channels, conv_channels, kernel_size=5, padding=2)
        self.conv_block4 = ConvBlock1D(conv_channels, conv_channels, kernel_size=5, padding=2)

        self.conv_dropout = nn.Dropout(dropout)

        # ---- LSTM temporal modeling ----
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        # ---- Pooling over time (attention or mean) ----
        self.use_attention = use_attention
        if use_attention:
            self.pool = AttentionPooling(lstm_out_dim)
        else:
            self.pool = None  # 用简单 mean pooling

        # ---- Classifier ----
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x, return_attn: bool = False):
        """
        x: (B, T, C)
        """
        # 1) CNN expects (B, C, T)
        x = x.permute(0, 2, 1)  # (B, C, T)

        # 2) CNN feature extractor
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        x = self.conv_dropout(x)

        # 3) Back to (B, T, F) for LSTM
        x = x.permute(0, 2, 1)  # (B, T, F)

        # 4) LSTM temporal modeling
        out, _ = self.lstm(x)   # out: (B, T, H*dir)

        # 5) 时间聚合
        attn_weights = None
        if self.use_attention:
            # AttentionPooling 返回 (context, alpha)
            context, alpha = self.pool(out)   # context: (B, H*dir)
            feat = context
            attn_weights = alpha
        else:
            # 简单 mean pooling
            feat = out.mean(dim=1)            # (B, H*dir)

        # 6) Dropout + Classifier
        feat = self.dropout(feat)             # 这里 feat 是 Tensor，不是 tuple
        logits = self.fc(feat)                # (B, num_classes)

        if return_attn and attn_weights is not None:
            return logits, attn_weights

        return logits





class TwoHeadDeepConvLSTM(nn.Module):
    """
    Two-head HAR model:
    - Head 1: Activity State (num_states classes, 比如 6)
    - Head 2: Transition (binary/probability, 标量 ∈ R)
    输入:  x [B, T, C]
    输出: dict:
        {
            "state_logits": [B, num_states],
            "trans_logit":  [B]
        }
    """

    def __init__(
        self,
        in_channels: int = 6,     # HAPT: 6 维
        num_states: int = 6,      # WALKING, SITTING, ...
        conv_channels: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = False,
    ):
        super().__init__()

        # ---- CNN ----
        self.conv1 = nn.Conv1d(in_channels, conv_channels, 5, padding=2)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, 5, padding=2)
        self.conv3 = nn.Conv1d(conv_channels, conv_channels, 5, padding=2)
        self.conv4 = nn.Conv1d(conv_channels, conv_channels, 5, padding=2)

        self.bn1 = nn.BatchNorm1d(conv_channels)
        self.bn2 = nn.BatchNorm1d(conv_channels)
        self.bn3 = nn.BatchNorm1d(conv_channels)
        self.bn4 = nn.BatchNorm1d(conv_channels)

        self.dropout = nn.Dropout(dropout)

        # ---- LSTM ----
        self.bidirectional = bidirectional
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # ---- 两个头 ----
        self.state_head = nn.Linear(lstm_out_dim, num_states)
        self.trans_head = nn.Linear(lstm_out_dim, 1)

    def forward(self, x):
        """
        x: [B, T, C]
        """
        # CNN 期望 [B, C, T]
        x = x.transpose(1, 2)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = self.dropout(x)

        # 回到 [B, T, C]
        x = x.transpose(1, 2)

        lstm_out, _ = self.lstm(x)          # [B, T, H]

        # 时间维上做 mean pooling（稳定）
        feat = lstm_out.mean(dim=1)         # [B, H]

        state_logits = self.state_head(feat)         # [B, num_states]
        trans_logit = self.trans_head(feat).squeeze(-1)  # [B]

        return {
            "state_logits": state_logits,
            "trans_logit": trans_logit,
        }
