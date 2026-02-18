import torch
import torch.nn as nn
import torch.nn.functional as F


class GravityPoseFeatures(nn.Module):
    """
    Input: x_raw, shape [B, T, 6] (ax, ay, az, gx, gy, gz)
    Output:
      - x_feat: [B, T, 11] = [acc_body(3), gyro_body(3), acc_world(3), pitch(1), roll(1)]
      - extra: dict storing intermediate physical quantities for visualization or auxiliary loss calculations
    """
    def __init__(self, kernel_size: int = 25, detach_gravity: bool = True):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size  best single -> padding='same'"
        self.kernel_size = kernel_size
        self.detach_gravity = detach_gravity

        # Apply fixed convolution kernel for smoothing (mean filtering), group=3 filters each axis separately
        weight = torch.ones(3, 1, kernel_size, dtype=torch.float32)
        weight = weight / kernel_size
        self.register_buffer("smooth_kernel", weight, persistent=False)

    def estimate_gravity(self, acc: torch.Tensor) -> torch.Tensor:
        """
        acc: [B, T, 3]
        Returns: 
        g_hat: [B, T, 3], unit vector, approximate direction of gravity
        """
        B, T, _ = acc.shape
        x = acc.permute(0, 2, 1)  # [B, 3, T]

        pad = (self.kernel_size - 1) // 2
        x_padded = F.pad(x, (pad, pad), mode="replicate")

        g = F.conv1d(x_padded, self.smooth_kernel, groups=3)  # [B, 3, T]
        g = g.permute(0, 2, 1)  # [B, T, 3]
        g = F.normalize(g, dim=-1, eps=1e-6)

        return g

    def build_world_frame(self, g: torch.Tensor):
        """
        Construct the world coordinate system basis vectors (x_world, y_world, z_world) based on the gravitational direction g
        g: [B, T, 3], already normalized
        Returns:
        x_w, y_w, z_w: [B, T, 3]
        """
        B, T, _ = g.shape
        device = g.device

        z_w = g  # Z-axis aligned with gravity direction

        # Select reference vectors to avoid parallelism with z_w
        ref = torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 1, 3).expand(B, T, 3)
        cos_sim = (z_w * ref).sum(dim=-1).abs()  # [B, T]
        mask = cos_sim > 0.9

        if mask.any():
            alt_ref = torch.tensor([0.0, 1.0, 0.0], device=device).view(1, 1, 3)
            ref = torch.where(mask[..., None], alt_ref.expand_as(ref), ref)

        x_w = torch.cross(ref, z_w, dim=-1)
        x_w = F.normalize(x_w, dim=-1, eps=1e-6)

        y_w = torch.cross(z_w, x_w, dim=-1)
        y_w = F.normalize(y_w, dim=-1, eps=1e-6)

        return x_w, y_w, z_w

    def body_to_world(self, acc_body: torch.Tensor, x_w, y_w, z_w):
        """
        acc_body: [B, T, 3]
        x_w, y_w, z_w: [B, T, 3]
        Returns acc_world: [B, T, 3]
        """
        acc_x = (acc_body * x_w).sum(dim=-1)  # [B, T]
        acc_y = (acc_body * y_w).sum(dim=-1)
        acc_z = (acc_body * z_w).sum(dim=-1)

        acc_world = torch.stack([acc_x, acc_y, acc_z], dim=-1)  # [B, T, 3]
        return acc_world

    def compute_angles(self, g: torch.Tensor):
        """
        Calculate pitch/roll using gravity direction g
        Common definitions:
        pitch = atan2(-gx, sqrt(gy^2 + gz^2))
        roll = atan2(gy, gz)
        g: [B, T, 3]
        Return pitch, roll: [B, T]
        """
        gx = g[..., 0]
        gy = g[..., 1]
        gz = g[..., 2]

        pitch = torch.atan2(-gx, torch.sqrt(gy * gy + gz * gz + 1e-6))
        roll = torch.atan2(gy, gz + 1e-6)

        return pitch, roll

    def forward(self, x_raw: torch.Tensor):
        """
        x_raw: [B, T, 6]  (ax, ay, az, gx, gy, gz)
        Returns:
          x_feat: [B, T, 11]
          extra: dict
        """
        assert x_raw.size(-1) == 6, "The last dimension should be 6 = 3 accelerations + 3 angular velocities."

        acc_body = x_raw[..., 0:3]
        gyro_body = x_raw[..., 3:6]

        # 1) Gravity estimation
        g_hat = self.estimate_gravity(acc_body)  # [B, T, 3]
        if self.detach_gravity:
            g_hat = g_hat.detach()

        # 2) Construct the world coordinate system
        x_w, y_w, z_w = self.build_world_frame(g_hat)

        # 3) Acceleration projected onto world coordinates
        acc_world = self.body_to_world(acc_body, x_w, y_w, z_w)  # [B, T, 3]

        # 4) Calculate posture angles
        pitch, roll = self.compute_angles(g_hat)  # [B, T]

        # 5) Spliced into new features
        pitch_feat = pitch.unsqueeze(-1)  # [B, T, 1]
        roll_feat = roll.unsqueeze(-1)    # [B, T, 1]

        x_feat = torch.cat(
            [acc_body, gyro_body, acc_world, pitch_feat, roll_feat],
            dim=-1
        )  # [B, T, 11]

        extra = {
            "g_hat": g_hat,
            "acc_world": acc_world,
            "pitch": pitch,
            "roll": roll,
            "x_w": x_w,
            "y_w": y_w,
            "z_w": z_w,
        }

        return x_feat, extra
