import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Union, List


from ..pose_features import GravityPoseFeatures, PostureSitStandFeatures


# =========================
# TCN Basic Block
# =========================
class TemporalConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_ch, out_ch,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)

        self.residual = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        # x: [B, C, T]
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)

        res = self.residual(x)
        return F.relu(out + res)


# =========================
# CNN + TCN Main Model（supports use_pose + phys_head + posture）
# =========================
class CNN_TCN_HAR(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 12,
        cnn_channels: int = 64,
        tcn_channels: int = 128,
        dropout: float = 0.3,

        # hyperparameters for CNN/TCN
        cnn_kernel_size: int = 5,
        tcn_kernel_size: int = 3,
        tcn_dilations: Union[Sequence[int], tuple] = (1, 2),
        pooling: str = "mean",        # "mean" or "mean_max"

        # Posture/ Physical Constraints 
        use_pose: bool = False,       
        pose_kernel_size: int = 25,
        pose_detach_gravity: bool = True,
        use_phys_head: bool = False,  

        # Window-level sitting/standing posture characteristics
        use_posture: bool = False,
    ):
        super().__init__()

        if isinstance(tcn_dilations, tuple):
            tcn_dilations = list(tcn_dilations)
        assert pooling in ["mean", "mean_max"], f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.use_pose = use_pose
        self.use_phys_head = use_phys_head
        self.use_posture = use_posture

        # ======================================
        # 1) Pose Module & Valid Input Channels
        # ======================================
        if self.use_pose:
            assert in_channels >= 6, "When `use_pose=True`, the first 6 dimensions must be acc+gyro."

            # pose only consumes the first 6 dimensions, outputs 11 dimensions: [acc_body, gyro_body, acc_world, pitch, roll]
            self.pose = GravityPoseFeatures(
                kernel_size=pose_kernel_size,
                detach_gravity=pose_detach_gravity,
            )
            pose_out_dim = 11
            extra_scalar_dim = in_channels - 6  # 例如 acc_norm / gyro_norm / jerk...
            eff_in_channels = pose_out_dim + extra_scalar_dim
        else:
            eff_in_channels = in_channels

        # ==========================================
        # 2) CNN Component (Local Temporal Features)
        # ==========================================
        pad = cnn_kernel_size // 2
        self.cnn = nn.Sequential(
            nn.Conv1d(eff_in_channels, cnn_channels, kernel_size=cnn_kernel_size, padding=pad),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),

            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=cnn_kernel_size, padding=pad),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
        )

        # =========================================
        # 3) TCN Component (Long-Term Dependencies)
        # =========================================
        tcn_blocks = []
        in_ch = cnn_channels
        for d in tcn_dilations:
            tcn_blocks.append(
                TemporalConvBlock(
                    in_ch, tcn_channels,
                    kernel_size=tcn_kernel_size,
                    dilation=d,
                    dropout=dropout,
                )
            )
            in_ch = tcn_channels
        self.tcn = nn.Sequential(*tcn_blocks)

        # ========================
        # 4) Global pooling + Head
        # ========================
        if self.pooling == "mean":
            base_feat_dim = tcn_channels       
        else:  # "mean_max"
            base_feat_dim = tcn_channels * 2   

        
        if self.use_posture:
            self.posture = PostureSitStandFeatures()
            feat_dim = base_feat_dim + 7
        else:
            feat_dim = base_feat_dim

        
        # >> Explicitly define the numbering for major categories / minor categories
        major_classes = [0, 1, 2, 3, 4, 5]        
        minor_classes = [6, 7, 8, 9, 10, 11]      

        # ---- Main category head (12 classes) uses ClassAwareHead  ----
        # >>> Modified: Originally nn.Sequential, now changed to ClassAwareHead
        self.classifier = ClassAwareHead(         
            feat_dim=feat_dim,
            num_classes=num_classes,
            major_classes=major_classes,
            minor_classes=minor_classes,
            minor_hidden_dim=128,
            minor_dropout=dropout,
        )
        
        # ---- main head（12 classes）----
        #self.classifier = nn.Sequential(
        #    nn.Linear(feat_dim, 128),
        #    nn.ReLU(),
        #    nn.Dropout(dropout),
        #    nn.Linear(128, num_classes),
        #)

        # >>> Transition aux head（6–11 classes）
        self.head_trans = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 6),   # class 6..11 → index 0..5
        )

        # 3/4 binary head（0-based 3/4 → original label: 4/5）
        self.head_34 = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),   # index 0 → class 3；index 1 → class 4
        )

        # physical head (6 physical targets)
        if self.use_phys_head:
            self.head_phys = nn.Linear(feat_dim, 6)
            self.mse = nn.MSELoss()

        # >>> 可选：保存 feat_dim 供外部调试
        self.feat_dim = feat_dim

        # =============================================================================
        # Explicitly define the backbone to facilitate unified parameter freezing later
        # Includes: pose (if enabled) + CNN + TCN + posture (if enabled)
        # =============================================================================
        backbone_modules = []

        if self.use_pose:
            backbone_modules.append(self.pose)

        # Primary Feature Extraction: CNN + TCN
        backbone_modules.append(self.cnn)
        backbone_modules.append(self.tcn)


        if self.use_posture:
            backbone_modules.append(self.posture)

        # 注册成 ModuleList，之后可以 model.backbone.parameters() 冻结
        #self.backbone = nn.ModuleList(backbone_modules)
        # >>> 修改这里：不要再用 ModuleList，否则 state_dict 里会多出 backbone.* 的 key
        # 原来：self.backbone = nn.ModuleList(backbone_modules)
        self.backbone = backbone_modules  # A regular Python list used to collect modules that need to be frozen.

    # ======================================================
    # Physical target calculation (on the original input x)
    # ======================================================
    def _compute_phys_targets(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C>=6] 
        return: [B, 6]
          0:3 = g_dir
          3   = log_energy
          4   = log_variance
          5   = log_jerk_energy
        """
        x = x.detach()
        B, T, C = x.shape
        eps = 1e-6

        acc = x[..., :3]                  # [B,T,3]
        g_vec = acc.mean(dim=1)           # [B,3]
        g_dir = g_vec / (torch.norm(g_vec, dim=1, keepdim=True) + eps)

        energy = torch.log1p((x ** 2).mean(dim=(1, 2)))

        var_time = x.var(dim=1)           # [B,C]
        variance = torch.log1p(var_time.mean(dim=1))

        dx = x[:, 1:, :] - x[:, :-1, :]   # [B,T-1,C]
        jerk_energy = torch.log1p((dx ** 2).mean(dim=(1, 2)))

        targets = torch.zeros(B, 6, device=x.device, dtype=x.dtype)
        targets[:, 0:3] = g_dir
        targets[:, 3] = energy
        targets[:, 4] = variance
        targets[:, 5] = jerk_energy

        return targets

    # ========================
    # forward
    # ========================
    def forward(
        self,
        x: torch.Tensor,
        return_phys_loss: bool = False,
        return_feat: bool = False,
    ):
        x_input = x  # Keep one original input for calculating phys_targets.

        # ----- Pose -----
        if self.use_pose:
            x_raw = x[..., :6]          # acc+gyro
            x_pose, _ = self.pose(x_raw)  # [B,T,11]

            if x.shape[-1] > 6:
                x_scalar = x[..., 6:]   # Additional scalar physical characteristics
                x = torch.cat([x_pose, x_scalar], dim=-1)
            else:
                x = x_pose

        # [B, C, T]
        x = x.permute(0, 2, 1)

        # CNN
        x = self.cnn(x)

        # TCN
        x = self.tcn(x)

        # Temporal pooling → x_feat
        if self.pooling == "mean":
            x_feat = x.mean(dim=-1)   # [B, C]
        else:  # "mean_max"
            mean = x.mean(dim=-1)
            maxv = x.max(dim=-1).values
            x_feat = torch.cat([mean, maxv], dim=-1)  # [B, 2C]

        # Window-level Attitude Features
        if self.use_posture:
            # x_input: [B, T, C>=6]，
            x_raw = x_input[..., :6]                 # [B, T, 6]
            f_posture = self.posture(x_raw)          # [B, 7]
            x_feat = torch.cat([x_feat, f_posture], dim=1)  # [B, base_feat_dim+7]

        # ---- main head → logits_main（12 classes）----
        logits = self.classifier(x_feat)

        # >>> Transition aux head → logits_trans（6 classes -> class 6..11）
        logits_trans = self.head_trans(x_feat)

        # >>> 3/4 aux head → logits_34（2 classes -> 3 vs 4）
        logits_34 = self.head_34(x_feat)

        # ========== Scenario 1: Neither phys_loss nor feat (default path) ==========
        # >>> The default return value is a dictionary containing the main logits and two auxiliary heads.
        if not return_phys_loss and not return_feat:
            return {
                "logits": logits,              
                "logits_trans": logits_trans,  
                "logits_34": logits_34,        
            }

        phys_loss = None

        # ========== Scenario 2: Requires phys_loss ==========
        if return_phys_loss and self.use_phys_head:
            phys_pred = self.head_phys(x_feat)              # [B,6]
            phys_tgt = self._compute_phys_targets(x_input)  # [B,6]
            phys_loss = self.mse(phys_pred, phys_tgt)

        # 2.1 phys_loss + feat
        if return_phys_loss and return_feat:
            return {
                "logits": logits,
                "logits_trans": logits_trans,
                "logits_34": logits_34,
                "phys_loss": phys_loss,
                "feat": x_feat,
            }

        # 2.2 only phys_loss
        if return_phys_loss and not return_feat:
            return {
                "logits": logits,
                "logits_trans": logits_trans,
                "logits_34": logits_34,
                "phys_loss": phys_loss,
            }

        # 2.3 only feat（to t-SNE）
        # >>> returns(logits, x_feat)
        if return_feat and not return_phys_loss:
            return logits, x_feat

        return {
            "logits": logits,
            "logits_trans": logits_trans,
            "logits_34": logits_34,
        }




class ClassAwareHead(nn.Module):
    """
    Input:
        feat: [B, D]  Features from CNN/TCN/LSTM backbone

    Design:
        - Major classes: Use a simple Linear layer
        - Minor classes: Use a more complex MLP (stronger fitting capability, allows higher dropout)

    Example:
        major_classes = [0,1,2,3,4,5]
        minor_classes = [6,7,8,9,10,11]
    """
    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        major_classes: List[int],
        minor_classes: List[int],
        minor_hidden_dim: int = 128,
        minor_dropout: float = 0.3,
    ):
        super().__init__()
        assert len(major_classes) + len(minor_classes) == num_classes

        self.num_classes = num_classes

        # Store the class ID in a buffer that automatically follows the device's movement
        self.register_buffer("major_idx", torch.tensor(major_classes, dtype=torch.long))
        self.register_buffer("minor_idx", torch.tensor(minor_classes, dtype=torch.long))

        # Major category head: Simple linear, preventing overfitting
        self.major_head = nn.Linear(feat_dim, len(major_classes))

        # Minor head: Use a single-layer MLP + dropout to enhance the subclass's expressive power
        self.minor_head = nn.Sequential(
            nn.Linear(feat_dim, minor_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=minor_dropout),
            nn.Linear(minor_hidden_dim, len(minor_classes)),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        feat: [B, D]
        返回:
            logits: [B, num_classes]
        """
        B, D = feat.shape

        logits = feat.new_zeros(B, self.num_classes)

        
        logits_major = self.major_head(feat)      # [B, |major|]
        logits_minor = self.minor_head(feat)      # [B, |minor|]


        logits[:, self.major_idx] = logits_major
        logits[:, self.minor_idx] = logits_minor

        return logits