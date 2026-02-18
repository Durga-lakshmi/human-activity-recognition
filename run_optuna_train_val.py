import torch
import wandb
import numpy as np
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from datasets import get_dataset
from models.util import get_model
from trainer import Trainer_HAPT as Trainer
from module_training.augmentations import compute_sample_weights
from torch.utils.data import WeightedRandomSampler


def run_train_val(cfg, trial=None) -> float:
    """
    给 Optuna 用的统一入口：
      - 构建 dataset / dataloader
      - 构建 model + Trainer
      - 训练（包含 early stopping）
      - 返回验证集 best macro F1

    注意：
      - 这里不跑 test（为了调参速度），test 可以在你最终确定超参后再用原 main 跑。
      - wandb 这里默认用 mode='disabled'，避免超参搜索跑爆 wandb。
    """

    # -------------------------------
    # Device
    # -------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -------------------------------
    # Datasets and Dataloaders
    # -------------------------------
    # 和你原 main 一样：先 train 再 val/test
    train_dataset, train_loader = get_dataset(cfg, split="train")
    val_dataset, val_loader = get_dataset(cfg, split="val")
    test_dataset, test_loader = get_dataset(cfg, split="test")  # 这里只是保持行为一致，不在本函数中使用

    # ===== 注意：如果你想在调参时用 WeightedRandomSampler，把下面两行改成 trainer 用 train_aug_loader =====
    labels = np.asarray(train_dataset.y, dtype=np.int64)
    sample_weights, class_counts, sample_class_weights = compute_sample_weights(
        labels,
        power=0.5
    )
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    train_aug_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
        sampler=sampler,
        shuffle=False
    )

    # 现在你原 main 里 trainer 用的是 train_loader（不是 aug），
    # 你自己决定要不要在 HPO 里用增强：
    use_aug_for_hpo = False  # 想改就设为 True
    if use_aug_for_hpo:
        train_loader_used = train_aug_loader
    else:
        train_loader_used = train_loader

    # -------------------------------
    # Model
    # -------------------------------
    model = get_model(cfg)
    model.to(device)

    # -------------------------------
    # Weights & Biases（调参时禁用）
    # -------------------------------
    # 为了不影响 Trainer 里 wandb.log，这里一定要 init 一下，
    # 用 mode="disabled" 就不会真的往服务器传。
    wandb.init(
        mode="disabled",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # -------------------------------
    # Trainer
    # -------------------------------
    trainer = Trainer(
        cfg=cfg,
        train_loader=train_loader_used,
        val_loader=val_loader,
        model=model,
        device=device,
    )

    trainer.train()

    # Trainer 里已经维护了 best_val_f1
    best_val_f1 = float(trainer.best_val_f1)

    # 结束当前 wandb run
    wandb.finish()

    return best_val_f1
