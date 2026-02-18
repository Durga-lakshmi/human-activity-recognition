import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMConv(nn.Module):
    """
    LSTM -> CNN -> classifier
    Input:  x (B, T, C)
    Output: logits (B, num_classes)
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        conv_channels: int = 64,
        kernel_size: int = 5,
        dropout: float = 0.5,
        bidirectional: bool = False,
        pool: str = "gap",   # "gap" or "last" or "attn"
    ):
        super().__init__()
        self.pool = pool
        self.bidirectional = bidirectional
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        # ---- LSTM first (temporal modeling) ----
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # ---- CNN on top of LSTM outputs ----
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(lstm_out_dim, conv_channels, kernel_size=kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(conv_channels)
        self.bn2 = nn.BatchNorm1d(conv_channels)

        # ---- Attention pooling projection (for "attn" mode) ----
        if self.pool == "attn":
            # 输入维度 = conv_channels (feature dim)，输出一个 score
            self.att_proj = nn.Linear(conv_channels, 1)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(conv_channels, num_classes)

    def forward(self, x):
        """
        x: (B, T, C)
        """
        # LSTM -> (B, T, H*)
        seq, _ = self.lstm(x)

        # CNN expects (B, F, T)
        z = seq.permute(0, 2, 1)  # (B, H*, T)

        z = F.relu(self.bn1(self.conv1(z)))
        z = F.relu(self.bn2(self.conv2(z)))  # (B, F, T)  F=conv_channels

        if self.pool == "gap":
            # Global average pooling over time -> (B, F)
            feat = z.mean(dim=-1)

        elif self.pool == "last":
            # take last time step in CNN feature map -> (B, F)
            feat = z[:, :, -1]

        elif self.pool == "attn":
            # ---- Attention pooling over time ----
            # z: (B, F, T) -> (B, T, F)
            z_t = z.permute(0, 2, 1)

            # score: (B, T, 1)
            # 可以先加一个非线性，比如 tanh
            scores = self.att_proj(torch.tanh(z_t))

            # 注意力权重: (B, T, 1)
            attn_weights = torch.softmax(scores, dim=1)

            # 加权求和: (B, T, F) * (B, T, 1) -> (B, T, F) -> sum over T -> (B, F)
            feat = (attn_weights * z_t).sum(dim=1)

        else:
            raise ValueError(f"Unknown pool: {self.pool}")

        feat = self.dropout(feat)
        logits = self.fc(feat)
        return logits


#--------
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMConv(nn.Module):
    """
    LSTM -> CNN -> classifier
    Input:  x (B, T, C)
    Output: logits (B, num_classes)
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        conv_channels: int = 64,
        kernel_size: int = 5,
        lstm_dropout: float = 0.0,   # 新：LSTM 层间 dropout
        head_dropout: float = 0.5,   # 新：classifier head dropout
        bidirectional: bool = False,
        pool: str = "gap",           # "gap" or "last" or "attn"
    ):
        super().__init__()
        self.pool = pool
        self.bidirectional = bidirectional
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        # ---- LSTM first (temporal modeling) ----
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # ---- CNN on top of LSTM outputs ----
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(lstm_out_dim, conv_channels, kernel_size=kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(conv_channels)
        self.bn2 = nn.BatchNorm1d(conv_channels)

        # ---- Attention pooling projection (for "attn" mode) ----
        if self.pool == "attn":
            # 输入维度 = conv_channels (feature dim)，输出一个 score
            self.att_proj = nn.Linear(conv_channels, 1)

        # 这里专门是 head dropout
        self.dropout = nn.Dropout(head_dropout)
        self.fc = nn.Linear(conv_channels, num_classes)

    def forward(self, x):
        """
        x: (B, T, C)
        """
        # LSTM -> (B, T, H*)
        seq, _ = self.lstm(x)

        # CNN expects (B, F, T)
        z = seq.permute(0, 2, 1)  # (B, H*, T)

        z = F.relu(self.bn1(self.conv1(z)))
        z = F.relu(self.bn2(self.conv2(z)))  # (B, F, T)  F=conv_channels

        if self.pool == "gap":
            # Global average pooling over time -> (B, F)
            feat = z.mean(dim=-1)

        elif self.pool == "last":
            # take last time step in CNN feature map -> (B, F)
            feat = z[:, :, -1]

        elif self.pool == "attn":
            # ---- Attention pooling over time ----
            # z: (B, F, T) -> (B, T, F)
            z_t = z.permute(0, 2, 1)

            # score: (B, T, 1)
            scores = self.att_proj(torch.tanh(z_t))

            # 注意力权重: (B, T, 1)
            attn_weights = torch.softmax(scores, dim=1)

            # 加权求和: (B, T, F) * (B, T, 1) -> (B, T, F) -> sum over T -> (B, F)
            feat = (attn_weights * z_t).sum(dim=1)

        else:
            raise ValueError(f"Unknown pool: {self.pool}")

        feat = self.dropout(feat)
        logits = self.fc(feat)
        return logits




#------
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMConv(nn.Module):
    """
    LSTM -> CNN -> classifier (+ 物理特征拼接)
    Input:  x (B, T, C), C >= 6 (加速度 + 角速度)
    Output: logits (B, num_classes)
    """
    def __init__(
        self,
        in_channels: int = 6,          # HAPT 用 6 维: acc(3) + gyro(3)
        num_classes: int = 12,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        conv_channels: int = 64,
        kernel_size: int = 5,
        lstm_dropout: float = 0.0,     # LSTM 层间 dropout
        head_dropout: float = 0.5,     # classifier head dropout
        bidirectional: bool = False,
        pool: str = "gap",             # "gap" or "last" or "attn"
    ):
        super().__init__()
        self.pool = pool
        self.bidirectional = bidirectional
        self.in_channels = in_channels

        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        # ---- LSTM (temporal modeling) ----
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # ---- CNN on top of LSTM outputs ----
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(lstm_out_dim, conv_channels, kernel_size=kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(conv_channels)
        self.bn2 = nn.BatchNorm1d(conv_channels)

        # ---- Attention pooling projection (for "attn" mode) ----
        if self.pool == "attn":
            self.att_proj = nn.Linear(conv_channels, 1)

        # head dropout
        self.dropout = nn.Dropout(head_dropout)

        # +6 是物理特征维度: g_dir(3) + energy + var + jerk_energy
        self.fc = nn.Linear(conv_channels + 6, num_classes)

    def forward(self, x):
        """
        x: (B, T, C), C >= 6
        """
        B, T, C = x.shape
        assert C >= 6, f"Expected in_channels >= 6 (acc+gyro), got {C}"

        # ---- LSTM ----
        seq, _ = self.lstm(x)          # (B, T, H*)

        # ---- CNN ----
        z = seq.permute(0, 2, 1)       # (B, H*, T)
        z = F.relu(self.bn1(self.conv1(z)))
        z = F.relu(self.bn2(self.conv2(z)))   # (B, F, T), F=conv_channels

        # ---- temporal pooling → feat: (B, F) ----
        if self.pool == "gap":
            feat = z.mean(dim=-1)      # (B, F)

        elif self.pool == "last":
            feat = z[:, :, -1]         # (B, F)

        elif self.pool == "attn":
            # z: (B, F, T) -> (B, T, F)
            z_t = z.permute(0, 2, 1)   # (B, T, F)
            scores = self.att_proj(torch.tanh(z_t))   # (B, T, 1)
            attn_weights = torch.softmax(scores, dim=1)  # (B, T, 1)
            feat = (attn_weights * z_t).sum(dim=1)    # (B, F)
        else:
            raise ValueError(f"Unknown pool: {self.pool}")

        # =====================================================
        #           物理特征 from 原始输入 x
        # =====================================================
        # 假定前 3 维是线加速度 acc，后面可以是 gyro 等
        acc = x[..., :3]                          # (B, T, 3)

        # ---- 重力方向（姿态核心） ----
        g_mean = acc.mean(dim=1)                  # (B, 3)
        g_dir = g_mean / (g_mean.norm(dim=-1, keepdim=True) + 1e-8)  # (B, 3)

        # ---- 能量 / 方差（动静强度） ----
        energy = (x ** 2).mean(dim=(1, 2))        # (B,)
        var = acc.var(dim=1, unbiased=False).mean(dim=1)  # (B,)

        # ---- jerk（高频变化，区分静/动/transition） ----
        if T > 1:
            jerk = acc[:, 1:, :] - acc[:, :-1, :]           # (B, T-1, 3)
            jerk_energy = (jerk ** 2).mean(dim=(1, 2))      # (B,)
        else:
            jerk_energy = torch.zeros(B, device=x.device, dtype=x.dtype)

        # ---- 拼成 6 维 extra 特征 ----
        extra = torch.cat(
            [
                g_dir,                             # (B, 3)
                energy.unsqueeze(-1),              # (B, 1)
                var.unsqueeze(-1),                 # (B, 1)
                jerk_energy.unsqueeze(-1),         # (B, 1)
            ],
            dim=-1,
        )                                          # (B, 6)

        # ---- 拼接 backbone 表示 + 物理特征 ----
        feat_cat = torch.cat([feat, extra], dim=-1)   # (B, F+6)
        feat_cat = self.dropout(feat_cat)

        logits = self.fc(feat_cat)                   # (B, num_classes)
        return logits



import torch
import torch.nn as nn
import torch.nn.functional as F

#---深层
class LSTMConv(nn.Module):
    """
    LSTM -> CNN -> classifier (+ 物理特征拼接)
    Input:  x (B, T, C), C >= 6 (加速度 + 角速度)
    Output:
        - logits (B, num_classes)
        - （可选）phys_loss: scalar
    """
    def __init__(
        self,
        in_channels: int = 6,          # HAPT: acc(3) + gyro(3)
        num_classes: int = 12,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        conv_channels: int = 64,
        kernel_size: int = 5,
        lstm_dropout: float = 0.0,
        head_dropout: float = 0.5,
        bidirectional: bool = False,
        pool: str = "gap",             # "gap" or "last" or "attn"
    ):
        super().__init__()
        self.pool = pool
        self.bidirectional = bidirectional
        self.in_channels = in_channels

        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        # ---- LSTM ----
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # ---- CNN ----
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(lstm_out_dim, conv_channels, kernel_size=kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(conv_channels)
        self.bn2 = nn.BatchNorm1d(conv_channels)

        if self.pool == "attn":
            self.att_proj = nn.Linear(conv_channels, 1)

        self.dropout = nn.Dropout(head_dropout)
        # conv_channels + 6: g_dir(3) + energy + var + jerk_energy
        self.fc = nn.Linear(conv_channels + 6, num_classes)

    def forward(self, x, return_phys_loss: bool = False):
        """
        x: (B, T, C)
        return:
            logits 或 (logits, phys_loss)
        """
        B, T, C = x.shape
        assert C >= 6, f"Expected in_channels >= 6 (acc+gyro), got {C}"

        # ---- LSTM ----
        seq, _ = self.lstm(x)               # (B, T, H*)

        # ---- CNN ----
        z = seq.permute(0, 2, 1)            # (B, H*, T)
        z = F.relu(self.bn1(self.conv1(z)))
        z = F.relu(self.bn2(self.conv2(z))) # (B, F, T)

        # ========== 物理 smoothness loss：特征随时间不要乱跳 ==========
        phys_loss = None
        if return_phys_loss and T > 1:
            # 在 conv 特征上做 temporal smoothness
            # z[:, :, 1:] - z[:, :, :-1] 形状: (B, F, T-1)
            diff = z[:, :, 1:] - z[:, :, :-1]
            phys_loss = (diff ** 2).mean()  # 标量

        # ---- pooling ----
        if self.pool == "gap":
            feat = z.mean(dim=-1)           # (B, F)

        elif self.pool == "last":
            feat = z[:, :, -1]              # (B, F)

        elif self.pool == "attn":
            z_t = z.permute(0, 2, 1)        # (B, T, F)
            scores = self.att_proj(torch.tanh(z_t))  # (B, T, 1)
            attn_weights = torch.softmax(scores, dim=1)
            feat = (attn_weights * z_t).sum(dim=1)   # (B, F)
        else:
            raise ValueError(f"Unknown pool: {self.pool}")

        # ========== 物理显式特征：g_dir + energy + var + jerk_energy ==========
        acc = x[..., :3]                    # (B, T, 3)

        g_mean = acc.mean(dim=1)            # (B, 3)
        g_dir = g_mean / (g_mean.norm(dim=-1, keepdim=True) + 1e-8)  # (B, 3)

        energy = (x ** 2).mean(dim=(1, 2))  # (B,)
        var = acc.var(dim=1, unbiased=False).mean(dim=1)  # (B,)

        if T > 1:
            jerk = acc[:, 1:, :] - acc[:, :-1, :]        # (B, T-1, 3)
            jerk_energy = (jerk ** 2).mean(dim=(1, 2))   # (B,)
        else:
            jerk_energy = torch.zeros(B, device=x.device, dtype=x.dtype)

        extra = torch.cat(
            [
                g_dir,                         # (B, 3)
                energy.unsqueeze(-1),          # (B, 1)
                var.unsqueeze(-1),             # (B, 1)
                jerk_energy.unsqueeze(-1),     # (B, 1)
            ],
            dim=-1,
        )                                      # (B, 6)

        feat_cat = torch.cat([feat, extra], dim=-1)   # (B, F+6)
        feat_cat = self.dropout(feat_cat)

        logits = self.fc(feat_cat)             # (B, num_classes)

        if return_phys_loss:
            # 如果 batch_size=1 或 T=1 时 phys_loss 可能为 None，这里兜个底
            if phys_loss is None:
                phys_loss = logits.new_tensor(0.0)
            return logits, phys_loss

        return logits



#-----浅层


import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMConv(nn.Module):
    """
    简化版：LSTM(1层) -> 单层 CNN -> classifier (+ 物理特征拼接)
    Input:  x (B, T, C), C >= 6 (加速度 + 角速度)
    Output:
        - logits (B, num_classes)
        - （可选）phys_loss: scalar
    """
    def __init__(
        self,
        in_channels: int = 6,          # HAPT: acc(3) + gyro(3)
        num_classes: int = 12,
        lstm_hidden: int = 64,         # 默认更小
        lstm_layers: int = 1,          # 强制 1 层
        conv_channels: int = 64,
        kernel_size: int = 5,
        lstm_dropout: float = 0.0,     # 基本不会用到（1层时忽略）
        head_dropout: float = 0.1,     # 默认小一点
        bidirectional: bool = False,
        pool: str = "gap",             # "gap" or "last"（"attn" 退化为 "gap"）
    ):
        super().__init__()
        self.pool = pool
        self.bidirectional = bidirectional
        self.in_channels = in_channels

        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        # ---- LSTM（固定 1 层，更浅）----
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=lstm_hidden,
            num_layers=1,              # 不管传什么，都强制 1 层
            batch_first=True,
            dropout=0.0,               # 单层 LSTM dropout 无效，直接关掉
            bidirectional=bidirectional,
        )

        # ---- CNN：单层卷积 + BN（替代原来的两层 conv）----
        pad = kernel_size // 2
        self.conv = nn.Conv1d(lstm_out_dim, conv_channels, kernel_size=kernel_size, padding=pad)
        self.bn = nn.BatchNorm1d(conv_channels)

        # 注意：不再单独实现 attn pooling，如果 pool="attn"，退化为 GAP
        self.dropout = nn.Dropout(head_dropout)

        # conv_channels + 6: g_dir(3) + energy + var + jerk_energy
        self.fc = nn.Linear(conv_channels + 6, num_classes)

    def forward(self, x, return_phys_loss: bool = False):
        """
        x: (B, T, C)
        return:
            logits 或 (logits, phys_loss)
        """
        B, T, C = x.shape
        assert C >= 6, f"Expected in_channels >= 6 (acc+gyro), got {C}"

        # ---- LSTM ----
        seq, _ = self.lstm(x)               # (B, T, H*)

        # ---- CNN ----
        z = seq.permute(0, 2, 1)            # (B, H*, T)
        z = F.relu(self.bn(self.conv(z)))   # (B, F, T)

        # ========== 物理 smoothness loss：特征随时间不要乱跳 ==========
        phys_loss = None
        if return_phys_loss and T > 1:
            # 在 conv 特征上做 temporal smoothness
            # z[:, :, 1:] - z[:, :, :-1] 形状: (B, F, T-1)
            diff = z[:, :, 1:] - z[:, :, :-1]
            phys_loss = (diff ** 2).mean()  # 标量

        # ---- pooling ----
        if self.pool in ["gap", "attn"]:    # attn 退化为 GAP
            feat = z.mean(dim=-1)           # (B, F)

        elif self.pool == "last":
            feat = z[:, :, -1]              # (B, F)

        else:
            raise ValueError(f"Unknown pool: {self.pool}")

        # ========== 物理显式特征：g_dir + energy + var + jerk_energy ==========
        acc = x[..., :3]                    # (B, T, 3)

        g_mean = acc.mean(dim=1)            # (B, 3)
        g_dir = g_mean / (g_mean.norm(dim=-1, keepdim=True) + 1e-8)  # (B, 3)

        energy = (x ** 2).mean(dim=(1, 2))  # (B,)
        var = acc.var(dim=1, unbiased=False).mean(dim=1)  # (B,)

        if T > 1:
            jerk = acc[:, 1:, :] - acc[:, :-1, :]        # (B, T-1, 3)
            jerk_energy = (jerk ** 2).mean(dim=(1, 2))   # (B,)
        else:
            jerk_energy = torch.zeros(B, device=x.device, dtype=x.dtype)

        extra = torch.cat(
            [
                g_dir,                         # (B, 3)
                energy.unsqueeze(-1),          # (B, 1)
                var.unsqueeze(-1),             # (B, 1)
                jerk_energy.unsqueeze(-1),     # (B, 1)
            ],
            dim=-1,
        )                                      # (B, 6)

        feat_cat = torch.cat([feat, extra], dim=-1)   # (B, F+6)
        feat_cat = self.dropout(feat_cat)

        logits = self.fc(feat_cat)             # (B, num_classes)

        if return_phys_loss:
            if phys_loss is None:
                phys_loss = logits.new_tensor(0.0)
            return logits, phys_loss

        return logits


#-----


import torch
import torch.nn as nn
import torch.nn.functional as F

#---深层
class LSTMConv(nn.Module):
    """
    LSTM -> CNN -> classifier (+ 物理特征拼接)
    Input:  x (B, T, C), C >= 6 (加速度 + 角速度)
    Output:
        - logits (B, num_classes)
        - （可选）phys_loss: scalar
    """
    def __init__(
        self,
        in_channels: int = 6,          # HAPT: acc(3) + gyro(3)
        num_classes: int = 12,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        conv_channels: int = 64,
        kernel_size: int = 5,
        lstm_dropout: float = 0.0,
        head_dropout: float = 0.5,
        bidirectional: bool = False,
        pool: str = "gap",             # "gap" or "last" or "attn"
    ):
        super().__init__()
        self.pool = pool
        self.bidirectional = bidirectional
        self.in_channels = in_channels

        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        # ---- LSTM ----
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # ---- CNN ----
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(lstm_out_dim, conv_channels, kernel_size=kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(conv_channels)
        self.bn2 = nn.BatchNorm1d(conv_channels)

        if self.pool == "attn":
            self.att_proj = nn.Linear(conv_channels, 1)

        self.dropout = nn.Dropout(head_dropout)
        # conv_channels + 6: g_dir(3) + energy + var + jerk_energy
        self.fc = nn.Linear(conv_channels + 6, num_classes)

    def forward(self, x, return_phys_loss: bool = False):
        """
        x: (B, T, C)
        return:
            logits 或 (logits, phys_loss)
        """
        B, T, C = x.shape
        assert C >= 6, f"Expected in_channels >= 6 (acc+gyro), got {C}"

        # ---- LSTM ----
        seq, _ = self.lstm(x)               # (B, T, H*)

        # ---- CNN ----
        z = seq.permute(0, 2, 1)            # (B, H*, T)
        z = F.relu(self.bn1(self.conv1(z)))
        z = F.relu(self.bn2(self.conv2(z))) # (B, F, T)

        # ========== 物理 smoothness loss：特征随时间不要乱跳 ==========
        phys_loss = None
        if return_phys_loss and T > 1:
            # 在 conv 特征上做 temporal smoothness
            # z[:, :, 1:] - z[:, :, :-1] 形状: (B, F, T-1)
            diff = z[:, :, 1:] - z[:, :, :-1]
            phys_loss = (diff ** 2).mean()  # 标量

        # ---- pooling ----
        if self.pool == "gap":
            feat = z.mean(dim=-1)           # (B, F)

        elif self.pool == "last":
            feat = z[:, :, -1]              # (B, F)

        elif self.pool == "attn":
            z_t = z.permute(0, 2, 1)        # (B, T, F)
            scores = self.att_proj(torch.tanh(z_t))  # (B, T, 1)
            attn_weights = torch.softmax(scores, dim=1)
            feat = (attn_weights * z_t).sum(dim=1)   # (B, F)
        else:
            raise ValueError(f"Unknown pool: {self.pool}")

        # ========== 物理显式特征：g_dir + energy + var + jerk_energy ==========
        acc = x[..., :3]                    # (B, T, 3)

        g_mean = acc.mean(dim=1)            # (B, 3)
        g_dir = g_mean / (g_mean.norm(dim=-1, keepdim=True) + 1e-8)  # (B, 3)

        energy = (x ** 2).mean(dim=(1, 2))  # (B,)
        var = acc.var(dim=1, unbiased=False).mean(dim=1)  # (B,)

        if T > 1:
            jerk = acc[:, 1:, :] - acc[:, :-1, :]        # (B, T-1, 3)
            jerk_energy = (jerk ** 2).mean(dim=(1, 2))   # (B,)
        else:
            jerk_energy = torch.zeros(B, device=x.device, dtype=x.dtype)

        extra = torch.cat(
            [
                g_dir,                         # (B, 3)
                energy.unsqueeze(-1),          # (B, 1)
                var.unsqueeze(-1),             # (B, 1)
                jerk_energy.unsqueeze(-1),     # (B, 1)
            ],
            dim=-1,
        )                                      # (B, 6)

        feat_cat = torch.cat([feat, extra], dim=-1)   # (B, F+6)
        feat_cat = self.dropout(feat_cat)

        logits = self.fc(feat_cat)             # (B, num_classes)

        if return_phys_loss:
            # 如果 batch_size=1 或 T=1 时 phys_loss 可能为 None，这里兜个底
            if phys_loss is None:
                phys_loss = logits.new_tensor(0.0)
            return logits, phys_loss

        return logits




