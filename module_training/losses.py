import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Sequence, Tuple


try:
    from omegaconf import ListConfig
except Exception:
    ListConfig = tuple()

class CrossEntropyLossWrapper(nn.Module):
    def __init__(
        self,
        num_classes: int,
        reduction: str = "mean",
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        weight=None,
        **kwargs,   # used to receive alpha_*
    ):
        super().__init__()

        self.num_classes = num_classes
        self.ignore_index = ignore_index

        # --------------------------------------------------
        # NEW: construct class weights from alpha_0 ... alpha_{C-1}
        # --------------------------------------------------
        if weight is None:
            w = torch.ones(num_classes, dtype=torch.float32)

            for i in range(num_classes):
                key = f"alpha_{i}"
                if key in kwargs and kwargs[key] is not None:
                    w[i] = float(kwargs[key])

            self.register_buffer("weight", w)
        else:
            self.register_buffer(
                "weight",
                torch.as_tensor(weight, dtype=torch.float32),
            )

        self.ce = nn.CrossEntropyLoss(
            weight=self.weight,
            reduction=reduction,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )

    def forward(self, logits, targets):
        return self.ce(logits, targets)

class FocalLossWrapper_0(nn.Module):
    """
    Multi-class Focal Loss wrapper:
      - Interface is kept as close as possible to CrossEntropyLossWrapper
      - Supports:
          num_classes
          ignore_index
          reduction
          alpha_0 ... alpha_{C-1} as per-class weights
      - Can be directly used in Trainer as nn.Module(logits, targets)
    """

    def __init__(
        self,
        num_classes: int,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -100,
        label_smoothing: float = 0.0,  # kept for interface compatibility, not used
        weight=None,
        **kwargs,  # receives alpha_0, alpha_1, ...
    ):
        super().__init__()

        self.num_classes = num_classes
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

        # --------- build class weights from alpha_i ---------
        if weight is None:
            w = torch.ones(num_classes, dtype=torch.float32)

            # allow alpha_0 ... alpha_{num_classes-1} in cfg
            for i in range(num_classes):
                key = f"alpha_{i}"
                if key in kwargs and kwargs[key] is not None:
                    w[i] = float(kwargs[key])

            self.register_buffer("weight", w)
        else:
            self.register_buffer(
                "weight",
                torch.as_tensor(weight, dtype=torch.float32),
            )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  [B, C]
        targets: [B] (int64), values in 0..C-1 or ignore_index
        """
        # flatten batch dimension (also supports [B, T, C] input)
        if logits.dim() > 2:
            # [B, T, C] -> [B*T, C]
            logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)

        device = logits.device
        self.weight = self.weight.to(device)

        # mask out ignore_index
        valid_mask = (targets != self.ignore_index)
        if not valid_mask.any():
            # no valid samples, return 0 (keep computation graph)
            return logits.new_tensor(0.0)

        logits_v = logits[valid_mask]   # [N, C]
        targets_v = targets[valid_mask] # [N]

        # log_softmax
        log_probs = F.log_softmax(logits_v, dim=1)        # [N, C]
        # select log p_t for ground-truth classes
        log_pt = log_probs[torch.arange(targets_v.size(0)), targets_v]  # [N]
        pt = log_pt.exp()                                              # [N]

        # alpha_t (class weight)
        if self.weight is not None:
            alpha_t = self.weight[targets_v]  # [N]
        else:
            alpha_t = 1.0

        # focal loss: - alpha_t * (1 - pt)^gamma * log_pt
        focal_factor = (1.0 - pt) ** self.gamma
        loss = -alpha_t * focal_factor * log_pt  # [N]

        # reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            # "none"
            # note: only returns losses of valid samples
            return loss


class FocalLossWrapper(nn.Module):
    """
    Multi-class Focal Loss wrapper:
      - Interface is kept as close as possible to CrossEntropyLossWrapper
      - Supports:
          num_classes
          ignore_index
          reduction
          alpha_0 ... alpha_{C-1} as per-class weights
          alpha_block_1, alpha_block_2 for group-level weights (e.g. classes 6–8, 9–11)
      - Can be directly used in Trainer as nn.Module(logits, targets)
    """

    def __init__(
        self,
        num_classes: int,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -100,
        label_smoothing: float = 0.0,  # kept for interface compatibility, not used
        weight=None,
        **kwargs,  # receives alpha_0, alpha_1, ..., alpha_block_1, alpha_block_2
    ):
        super().__init__()

        self.num_classes = num_classes
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

        # --------- build class weights from alpha_i / alpha_block_j ---------
        if weight is None:
            w = torch.ones(num_classes, dtype=torch.float32)

            # 1) apply block-level alpha first (e.g. classes 6,7,8 and 9,10,11)
            #    per-class alpha_i will override these values if provided
            alpha_block_1 = kwargs.get("alpha_block_1", None)
            alpha_block_2 = kwargs.get("alpha_block_2", None)

            # block_1: classes 6,7,8
            if alpha_block_1 is not None:
                for i in [6, 7, 8]:
                    if i < num_classes and f"alpha_{i}" not in kwargs:
                        # only apply block value if no per-class alpha_i is specified
                        w[i] = float(alpha_block_1)

            # block_2: classes 9,10,11
            if alpha_block_2 is not None:
                for i in [9, 10, 11]:
                    if i < num_classes and f"alpha_{i}" not in kwargs:
                        w[i] = float(alpha_block_2)

            # 2) apply per-class alpha_i (higher priority than block)
            #    allow alpha_0 ... alpha_{num_classes-1} in cfg
            for i in range(num_classes):
                key = f"alpha_{i}"
                if key in kwargs and kwargs[key] is not None:
                    w[i] = float(kwargs[key])

            self.register_buffer("weight", w)
        else:
            self.register_buffer(
                "weight",
                torch.as_tensor(weight, dtype=torch.float32),
            )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  [B, C] or [B, T, C]
        targets: [B] or [B, T] (int64), values in 0..C-1 or ignore_index
        """
        # flatten batch / temporal dimensions
        if logits.dim() > 2:
            # [B, T, C] -> [B*T, C]
            logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)

        device = logits.device
        self.weight = self.weight.to(device)

        # mask out ignore_index
        valid_mask = (targets != self.ignore_index)
        if not valid_mask.any():
            # no valid samples, return 0 (keep computation graph)
            return logits.new_tensor(0.0)

        logits_v = logits[valid_mask]   # [N, C]
        targets_v = targets[valid_mask] # [N]

        # log_softmax
        log_probs = F.log_softmax(logits_v, dim=1)        # [N, C]
        # select log p_t for ground-truth classes
        log_pt = log_probs[torch.arange(targets_v.size(0), device=device), targets_v]  # [N]
        pt = log_pt.exp()                                              # [N]

        # alpha_t (class weight)
        if self.weight is not None:
            alpha_t = self.weight[targets_v]  # [N]
        else:
            alpha_t = 1.0

        # focal loss: - alpha_t * (1 - pt)^gamma * log_pt
        focal_factor = (1.0 - pt) ** self.gamma
        loss = -alpha_t * focal_factor * log_pt  # [N]

        # reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            # "none"
            # note: only returns losses of valid samples
            return loss






class AsymmetricStateLoss(nn.Module):
    """
    Wrapper that:
      1) computes per-sample CE / Focal loss
      2) applies extra penalty when GT is a transition class
         and the prediction is a static class
      3) can be enabled / disabled via a switch

    Supported logits shapes:
      - [B, T, K]
      - [N, K]
    targets:
      - [B, T] or [N]
    """
    def __init__(
        self,
        base_criterion: nn.Module,
        static_classes: Sequence[int] = (3, 4, 5),       # // replace with your static class indices
        transition_range: Tuple[int, int] = (6, 11),     # // replace with your transition class range [min, max]
        penalty_factor: float = 3.0,
        enabled: bool = True,
    ):
        super().__init__()
        self.base_criterion = base_criterion
        self.static_classes = tuple(static_classes)
        self.transition_range = transition_range
        self.penalty_factor = float(penalty_factor)
        self.enabled = bool(enabled)

        # base_criterion must return per-sample loss
        if getattr(self.base_criterion, "reduction", None) != "none":
            raise ValueError(
                "AsymmetricStateLoss: base_criterion must use reduction='none'"
            )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, T, K] or [N, K]
        targets: [B, T] or [N]
        """
        if logits.dim() == 3:
            B, T, K = logits.shape
            logits_flat = logits.reshape(-1, K)   # [N, K]
            targets_flat = targets.view(-1)       # [N]
        elif logits.dim() == 2:
            N, K = logits.shape
            logits_flat = logits
            targets_flat = targets.view(-1)
        else:
            raise ValueError(f"Unsupported logits shape: {logits.shape}")

        # 1) per-sample CE / Focal loss
        ce_per_sample = self.base_criterion(
            logits_flat,  # [N, K]
            targets_flat  # [N]
        )  # [N]

        # if asymmetric penalty is disabled, directly return mean
        if (not self.enabled) or (self.penalty_factor <= 1.0):
            return ce_per_sample.mean()

        # 2) construct asymmetric penalty
        with torch.no_grad():
            preds_flat = logits_flat.argmax(dim=-1)   # [N]
            gt_flat = targets_flat                   # [N]

            penalty = torch.ones_like(
                ce_per_sample, dtype=torch.float32
            )

            t_min, t_max = self.transition_range
            is_transition = (gt_flat >= t_min) & (gt_flat <= t_max)

            is_static_pred = torch.zeros_like(preds_flat, dtype=torch.bool)
            for c in self.static_classes:
                is_static_pred |= (preds_flat == c)

            bad_case = is_transition & is_static_pred
            penalty[bad_case] *= self.penalty_factor

        # 3) apply penalty and compute mean
        return (ce_per_sample * penalty).mean()
