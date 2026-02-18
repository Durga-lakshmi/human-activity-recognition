from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR, ReduceLROnPlateau

def build_scheduler(cfg, optimizer):
    """
    Returns:
        scheduler or None
    """
    sch_cfg = cfg.train_module.scheduler

    if sch_cfg.name.lower() in ["none", "null", "no"]:
        return None

    name = sch_cfg.name.lower()

    if name == "cosine":
        c = sch_cfg
        return CosineAnnealingLR(
            optimizer,
            T_max=int(c.T_max),
            eta_min=float(c.eta_min),
        )

    if name == "step":
        s = sch_cfg
        return StepLR(
            optimizer,
            step_size=int(s.step_size),
            gamma=float(s.gamma),
        )

    if name == "exponential":
        e = sch_cfg
        return ExponentialLR(
            optimizer,
            gamma=float(e.gamma),
        )


    if name in ["plateau", "reducelronplateau"]:
        p = sch_cfg
        return ReduceLROnPlateau(
            optimizer,
            mode=p.get("mode", "min"),          # "min" for loss
            factor=float(p.factor),             # e.g. 0.5
            patience=int(p.patience),           # e.g. 5
            threshold=float(p.threshold),       # e.g. 1e-3
            min_lr=float(p.min_lr),              # e.g. 1e-6
            verbose=True,
        )
        
    raise ValueError(f"Unknown scheduler: {sch_cfg.name}")
