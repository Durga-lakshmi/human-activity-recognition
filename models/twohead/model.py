import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class HarBackbone(nn.Module):
    """
    Backbone: Conv1d -> Conv1d -> LSTM -> last hidden
    Supports optional mask for temporal-crop training.
    Input x:   [B, T, C]
    Input mask:[B, T] bool (True=valid, False=padding)
    """
    def __init__(
        self,
        in_ch: int = 6,
        feat: int = 128,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        bidir: bool = False,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.proj = nn.Conv1d(128, feat, kernel_size=1)

        self.bidir = bidir
        self.rnn = nn.LSTM(
            input_size=feat,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidir,
        )
        base = lstm_hidden * (2 if bidir else 1)
        self.out_dim = base * 2   # mean+max 拼接 => 2倍

        # two MaxPool1d(2) => time length / 4
        self.time_downsample = 4

    def forward(self, x, mask=None):
        """
        x:    [B, T, C]
        mask: [B, T] (True = valid, False = padded/zeroed)
        """
        B, T, C = x.shape
        x = x.transpose(1, 2)           # [B, C, T]
        x = self.conv(x)                # [B, 128, T']
        x = self.proj(x)                # [B, feat, T']
        x = x.transpose(1, 2)           # [B, T', feat]

        if mask is None:
            #h, _ = self.rnn(x)          # [B, T', H]
            #return h[:, -1, :]          # [B, H]
            h, _ = self.rnn(x)                       # [B, T', H]
            h_mean = h.mean(dim=1)                   # [B, H]
            h_max = h.max(dim=1).values              # [B, H]
            return torch.cat([h_mean, h_max], dim=-1)  # [B, 2H]

        # ---- mask 下采样到 T' ----
        mask_ds = mask[:, ::self.time_downsample]          # [B, T']
        mask_ds = mask_ds[:, :x.size(1)]                   # 对齐长度
        h, _ = self.rnn(x)                                 # [B, T', H]

        # masked mean
        m = mask_ds.unsqueeze(-1).float()                  # [B, T', 1]
        h_sum = (h * m).sum(dim=1)                         # [B, H]
        den = m.sum(dim=1).clamp(min=1.0)                  # [B, 1]
        h_mean = h_sum / den                               # [B, H]

        # masked max（把无效位置设成 -inf）
        h_masked = h.masked_fill(~mask_ds.unsqueeze(-1), float("-inf"))
        h_max = h_masked.max(dim=1).values
        h_max = torch.where(torch.isfinite(h_max), h_max, torch.zeros_like(h_max))

        return torch.cat([h_mean, h_max], dim=-1)          # [B, 2H]


class TwoHeadHAR(nn.Module):
    """
    New Two-head (correct for your new pipeline):
      - state head: 6-class base activity (0..5)
      - transition head: 1 logit (sigmoid -> prob/ratio)
    Output is dict:
      {
        "state_logits": [B, num_states],
        "trans_logit":  [B]
      }
    """
    def __init__(
        self,
        in_ch: int = 6,
        num_states: int = 6,
        dropout: float = 0.2,
        bidir: bool = False,
    ):
        super().__init__()
        self.num_states = num_states

        self.backbone = HarBackbone(
            in_ch=in_ch,
            feat=128,
            lstm_hidden=128,
            lstm_layers=1,
            bidir=bidir,
        )

        #self.head_state = nn.Sequential(
        #    nn.Dropout(dropout),
        #    nn.Linear(self.backbone.out_dim, num_states),
        #)
        self.head_state = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.out_dim + 6, num_states),
        )

        # transition: scalar logit
        self.head_trans = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.out_dim, 1),
        )

    def forward(self, x, mask=None):
        """
        x: [B, T, C], C>=6 (acc+gyro)
        """
        feat = self.backbone(x, mask)  # [B, H]

        # ---------- 物理特征 ----------
        # 1) 加速度（假设前 3 维）
        acc = x[..., :3]                              # [B, T, 3]

        # ---- 重力方向（姿态核心）----
        g_mean = acc.mean(dim=1)                      # [B, 3]
        g_dir = g_mean / (g_mean.norm(dim=-1, keepdim=True) + 1e-8)  # [B, 3]

        # ---- 能量 / 方差（动静强度）----
        energy = (x ** 2).mean(dim=(1, 2))            # [B]
        var    = acc.var(dim=1).mean(dim=1)           # [B]

        # ---- jerk（高频变化，区分静/动/transition）----
        jerk = acc[:, 1:, :] - acc[:, :-1, :]         # [B, T-1, 3]
        jerk_energy = (jerk ** 2).mean(dim=(1, 2))    # [B]

        # ---------- 拼接 ----------
        extra = torch.cat(
            [
                g_dir,                                # [B, 3] 姿态方向
                energy.unsqueeze(-1),                 # [B, 1]
                var.unsqueeze(-1),                    # [B, 1]
                jerk_energy.unsqueeze(-1),            # [B, 1]
            ],
            dim=-1
        )                                             # [B, 6]

        feat2 = torch.cat([feat, extra], dim=-1)     # [B, H+6]

        # ---------- heads ----------
        state_logits = self.head_state(feat2)         # [B, num_states]
        trans_logit  = self.head_trans(feat).squeeze(-1)  # [B]

        return {
            "state_logits": state_logits,
            "trans_logit": trans_logit,
        }


# ---------------- optional helpers (keep if you still use crop) ----------------

def temporal_random_crop_with_mask(x, crop_len):
    T, C = x.shape
    if crop_len >= T:
        mask = torch.ones(T, dtype=torch.bool, device=x.device)
        return x, mask

    t0 = torch.randint(0, T - crop_len + 1, (1,), device=x.device).item()
    mask = torch.zeros(T, dtype=torch.bool, device=x.device)
    mask[t0:t0 + crop_len] = True

    x_masked = x.clone()
    x_masked[~mask] = 0.0
    return x_masked, mask


def apply_stage2_crop_batch(x, crop_len=64):
    B, T, C = x.shape
    device = x.device

    x_out = x.clone()
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)

    if crop_len >= T:
        mask[:] = True
        return x_out, mask

    for i in range(B):
        t0 = torch.randint(0, T - crop_len + 1, (1,), device=device).item()
        mask[i, t0:t0 + crop_len] = True
        x_out[i, ~mask[i]] = 0.0

    return x_out, mask
