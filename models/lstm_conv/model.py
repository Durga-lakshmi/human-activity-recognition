
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


