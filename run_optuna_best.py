import os
import torch
import wandb
import numpy as np

from hydra import initialize, compose
from omegaconf import OmegaConf

from datasets import get_dataset, get_dataset_with_minor
from models.util import get_model
from trainer import Trainer_HAPT as Trainer
from eval import run_test_eval_HAPT

from module_training.augmentations import compute_sample_weights
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader


def load_base_cfg():
    """
    Same as in optuna_runner: use Hydra to compose the full config.
    Equivalent to:
      @hydra.main(config_path='config', config_name='default')
    """
    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name="default")
    return cfg


def apply_best_params(cfg, best):
    """
    Write the hyperparameters stored in best_optuna_params.yaml
    back into cfg according to the mapping rules used in the
    Optuna objective.

    Assumes that `best` contains the following keys:
      - lr
      - optim_lr
      - weight_decay
      - cnn_channels
      - tcn_channels
      - tcn_dropout
      - tcn_dilations_pattern
      - lambda_bayes / lambda_phys / lambda_trans / lambda_temp
    """

    # ---------- learning rate ----------
    if "lr" in best:
        cfg.lr = float(best.lr)

    # Optimizer lr / weight_decay
    if "optim_lr" in best:
        cfg.train_module.optimizer.lr = float(best.optim_lr)
    if "weight_decay" in best:
        cfg.train_module.optimizer.weight_decay = float(best.weight_decay)

    # ---------- model architecture ----------
    if "cnn_channels" in best:
        cfg.model.cnn_channels = int(best.cnn_channels)
    if "tcn_channels" in best:
        cfg.model.tcn_channels = int(best.tcn_channels)
    if "tcn_dropout" in best:
        cfg.model.tcn_dropout = float(best.tcn_dropout)

    # Dilation pattern -> concrete list
    if "tcn_dilations_pattern" in best:
        pattern = str(best.tcn_dilations_pattern)
        pattern_map = {
            "1_2_4":   [1, 2, 4],
            "1_2_4_8": [1, 2, 4, 8],
            "2_4":     [2, 4],
            "2_4_8":   [2, 4, 8],
        }
        if pattern not in pattern_map:
            raise ValueError(f"Unknown tcn_dilations_pattern: {pattern}")
        cfg.model.tcn_dilations = pattern_map[pattern]

    # ---------- loss coefficients ----------
    if "lambda_bayes" in best:
        cfg.train.lambda_bayes = float(best.lambda_bayes)
    if "lambda_phys" in best:
        cfg.train.lambda_phys = float(best.lambda_phys)
    if "lambda_trans" in best:
        cfg.train.lambda_trans = float(best.lambda_trans)
    if "lambda_temp" in best:
        cfg.train.lambda_temp = float(best.lambda_temp)

    return cfg


def main():
    # -------------------------------
    # Device
    # -------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------------
    # 1) Load base config
    # -------------------------------
    cfg = load_base_cfg()

    # -------------------------------
    # 2) Load Optuna best parameters
    # -------------------------------
    best_yaml_path = "optuna_log/best_optuna_params_16.yaml"
    if not os.path.exists(best_yaml_path):
        raise FileNotFoundError(
            f"{best_yaml_path} not found. "
            f"Please save study.best_params first in optuna_runner.py."
        )

    best_params = OmegaConf.load(best_yaml_path)
    print("\n[run_best] Loaded best params from best_optuna_params.yaml:")
    print(best_params)

    # Write best parameters into cfg
    cfg = apply_best_params(cfg, best_params)

    # -------------------------------
    # 3) Training epochs for the final run
    #    (you can change this here, e.g., 80 / 100)
    # -------------------------------
    # If you want to keep the original cfg.epochs, comment out this line
    cfg.epochs = 200  # getattr(cfg, "epochs", 100)
    print(f"[run_best] Using epochs = {cfg.epochs}")

    # -------------------------------
    # 4) Dataset & DataLoader
    # -------------------------------
    print(f"Dataset: {cfg.dataset.name}")

    if cfg.train.minor_dataloader:
        train_dataset, train_loader, minor_loader = get_dataset_with_minor(cfg)
    else:
        train_dataset, train_loader = get_dataset(cfg, split="train")

    val_dataset, val_loader = get_dataset(cfg, split="val")
    test_dataset, test_loader = get_dataset(cfg, split="test")

    labels = np.asarray(train_dataset.y, dtype=np.int64)

    # 1) Compute sample weights based on all labels (including -1);
    #    handling of -1 is already implemented inside compute_sample_weights
    sample_weights, class_counts, sample_class_weights = compute_sample_weights(
        labels, power=0.5
    )
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    train_aug_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
        sampler=sampler,
        shuffle=False,
    )

    # -------------------------------
    # 5) Model
    # -------------------------------
    model = get_model(cfg)
    model.to(device)

    # ====================================================
    # NEW: If fine_tune is enabled, load weights from
    #      finetune_ckpt
    #      cfg.train.fine_tune: bool
    #      cfg.train.fine_tune_ckpt: checkpoint path
    #      (recommended to set in yaml or via command line)
    # ====================================================
    if getattr(cfg.train, "fine_tune", False):
        ckpt_path = getattr(cfg.train, "fine_tune_ckpt", None)
        if ckpt_path is None:
            raise ValueError(
                "cfg.train.fine_tune=True but cfg.train.fine_tune_ckpt is not set."
            )
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"finetune_ckpt not found: {ckpt_path}")

        print(f"[run_best] Loading finetune checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)

        # Typically checkpoints store model weights like this.
        # If your key differs (e.g., 'state_dict'), modify here.
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print("[run_best] Checkpoint loaded for fine-tuning.")

    # -------------------------------
    # 6) WandB (recommended for the final training run)
    # -------------------------------
    wandb.init(
        #entity=cfg.wandb.init.entity,
        project=cfg.wandb.init.project,
        name=cfg.wandb.init.name,
        group=cfg.wandb.init.group,
        tags=list(cfg.wandb.init.tags) if "tags" in cfg.wandb.init else None,
        id=cfg.wandb.init.id,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    wandb.watch(
        model,
        log="gradients",
        log_freq=100,
    )

    # -------------------------------
    # 7) Trainer: training + validation
    # -------------------------------
    if cfg.train.minor_dataloader:
        trainer = Trainer(
            cfg=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            device=device,
            minor_loader=minor_loader,
        )
    else:
        trainer = Trainer(
            cfg=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            device=device,
        )

    trainer.train()

    # -------------------------------
    # 8) Test set evaluation
    # -------------------------------

    print("[run_best] Running final test evaluation...")
    run_test_eval(cfg, model, device)

    print("[run_best] Done.")


if __name__ == "__main__":
    main()