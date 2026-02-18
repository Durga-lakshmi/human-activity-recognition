from torch.utils.data import DataLoader
from .hapt.dataset import HAPT12RawWindows as HAPT
from .har.dataset import HAR
# from .hapt.dataset_process import compute_canonical_gdir

import numpy as np
from torch.utils.data import Subset

def get_dataset(cfg, split: str = "train", mean=None, std=None, shuffle: bool = True):
    """
    Create the HAPT dataset and DataLoader.

    Notes:

    The mean and standard deviation are fully computed inside HAPT12RawWindows and written back to cfg.dataset.mean and cfg.dataset.std.

    Mean/std are no longer passed in or handled here.
    """
    # 
    #augment = (split == "train")


    # create Dataset
    # Note: Mean and standard deviation are no longer passed in
    #the dataset computes them during the first training run and caches the results internally.

    if cfg.name == "default_HAPT":
      dataset = HAPT(cfg, split=split)

      # create DataLoader
      dataloader = DataLoader(
          dataset,
          batch_size=cfg.dataset.batch_size,
          shuffle=(shuffle and split == "train"),
          num_workers=cfg.dataset.num_workers,
          pin_memory=True,
      )
      # return
      return dataset, dataloader

    else:
      DatasetClass = HAR
      compute_stats = None
      dataset = DatasetClass(cfg, split=split)
      dataloader = DataLoader(
        dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=shuffle if split=="train" else False,
        num_workers=cfg.dataset.num_workers,
        pin_memory=True)
      if split == "train":
        return dataset, dataloader, mean, std
      else:
          return dataset, dataloader




# ============================
# ===  double DataLoader =====
# ============================
def get_dataset_with_minor(cfg):
    """
    Returns:
      - train_dataset
      - main_loader   : all classes (1..12)
      - minor_loader  : minor classes (7..12)
      - val_loader
    """

    # -------- train dataset --------
    train_dataset = HAPT(cfg, split="train")

    main_loader = DataLoader(
      train_dataset,
      batch_size=cfg.dataset.batch_size,
      shuffle=True,
      num_workers=cfg.dataset.num_workers,
      pin_memory=True,
    )

    # -------- Create Minor Classes loader --------
    # y: shape [N] -> 1..12
    labels = train_dataset.y

    minor_mask = (labels >= 7) & (labels <= 12)
    minor_indices = np.where(minor_mask)[0]

    if len(minor_indices) == 0:
      raise RuntimeError("No minor-class samples found (7–12).")
    minor_dataset = Subset(train_dataset, minor_indices)

    minor_loader = DataLoader(
      minor_dataset,
      batch_size=cfg.dataset.batch_size,
      shuffle=True,
      num_workers=cfg.dataset.num_workers,
      pin_memory=True,
    )


    return train_dataset, main_loader, minor_loader