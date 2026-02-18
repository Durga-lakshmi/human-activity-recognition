import torch
import torch.nn as nn


class TCNBlock(nn.Module):
    """
    单个 TCN block:
    Conv1D (dilated) + BN + ReLU + Dropout + Residual
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()

        # same padding，保持时间长度不变
        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        # residual 对齐通道
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        """
        x: [B, C, T]
        """
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)

        return out + self.residual(x)


class TCN(nn.Module):
    """
    TCN for HAR (baseline)

    Input : [B, T, C]
    Output: [B, num_classes]
    """
    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 6,
        hidden_channels: int = 64,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        c_in = in_channels

        for i in range(num_layers):
            layers.append(
                TCNBlock(
                    in_channels=c_in,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=2 ** i,   # 1, 2, 4, 8
                    dropout=dropout,
                )
            )
            c_in = hidden_channels

        self.backbone = nn.Sequential(*layers)

        # 只做最干净的分类头
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x):
        """
        x: [B, T, C]
        """
        x = x.transpose(1, 2)          # [B, C, T]
        x = self.backbone(x)           # [B, hidden, T]
        x = x.mean(dim=-1)             # Global Average Pooling over time
        logits = self.classifier(x)    # [B, num_classes]
        return logits
