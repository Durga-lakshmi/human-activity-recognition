import os
from glob import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# -----------------------------
# Constants
# -----------------------------
ACTIVITIES = [
    "walking", "running", "sitting", "standing",
    "lying", "climbingup", "climbingdown", "jumping"
]
ACT2IDX = {a: i for i, a in enumerate(ACTIVITIES)}

POSITIONS = ["chest", "head", "shin", "thigh", "upperarm", "waist", "forearm"]

WINDOW_SIZE = 128
STRIDE = 128  # no overlap for val/test
FS = 50       # approx sampling frequency
TRIM_SECONDS = 5
TRIM_SAMPLES = TRIM_SECONDS * FS


# -----------------------------
# Helper functions
# -----------------------------
def parse_acc_filename(path):
    # Example: acc_climbingdown_chest.csv
    name = os.path.basename(path).replace(".csv", "")
    parts = name.split("_")
    if len(parts) != 3:
        return None
    _, activity, position = parts
    if activity not in ACT2IDX or position not in POSITIONS:
        return None
    return activity, position


def load_xyz(path):
    df = pd.read_csv(path)
    for c in ["attr_x", "attr_y", "attr_z", "attr_time"]:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in {path}")
    return df[["attr_time", "attr_x", "attr_y", "attr_z"]].to_numpy()


def align_acc_gyr(acc, gyr, tol_ms=5):
    """
    Align accelerometer and gyroscope by nearest timestamp
    Returns [N, 6] array: ax, ay, az, gx, gy, gz
    """
    acc_df = pd.DataFrame(acc, columns=["t", "ax", "ay", "az"])
    gyr_df = pd.DataFrame(gyr, columns=["t", "gx", "gy", "gz"])

    merged = pd.merge_asof(
        acc_df.sort_values("t"),
        gyr_df.sort_values("t"),
        on="t",
        direction="nearest",
        tolerance=tol_ms
    ).dropna()

    if len(merged) == 0:
        return np.empty((0, 6))

    return merged[["ax", "ay", "az", "gx", "gy", "gz"]].to_numpy()


# -----------------------------
# Dataset class
# -----------------------------
class HAR(Dataset):
    def __init__(self, cfg, split="train"):
        self.samples = []  # list of dicts: {position: [T,6]}
        self.labels = []

        # Strict subject split
        if split == "train":
            self.probands = [1, 2, 5, 8, 11, 12, 13, 15]
        elif split == "val":
            self.probands = [3]
        elif split == "test":
            self.probands = [9, 10]
        else:
            raise ValueError(f"Invalid split: {split}")

        for pid in self.probands:
            data_dir = os.path.join(cfg.dataset.path, f"proband{pid}", "data")
            for activity in ACTIVITIES:
                position_windows = {}
                for pos in POSITIONS:
                    acc_files = glob(os.path.join(data_dir, f"acc_{activity}_{pos}.csv"))
                    gyr_files = glob(os.path.join(data_dir, f"Gyroscope_{activity}_{pos}.csv"))

                    if not acc_files or not gyr_files:
                        continue  # skip missing files

                    acc = load_xyz(acc_files[0])
                    gyr = load_xyz(gyr_files[0])

                    # Align and trim
                    data = align_acc_gyr(acc, gyr)
                    if data.shape[0] < WINDOW_SIZE + 2*TRIM_SAMPLES:
                        continue
                    data = data[TRIM_SAMPLES:-TRIM_SAMPLES]

                    # Split into windows
                    windows = []
                    for i in range(0, len(data) - WINDOW_SIZE + 1, STRIDE):
                        win = data[i:i+WINDOW_SIZE]
                        # Per-window normalization
                        win = (win - win.mean(axis=0)) / (win.std(axis=0) + 1e-6)
                        windows.append(win)

                    if windows:
                        position_windows[pos] = windows

                # Keep only samples where all positions exist
                if len(position_windows) == len(POSITIONS):
                    min_windows = min(len(windows) for windows in position_windows.values())
                    for i in range(min_windows):
                        sample_dict = {pos: position_windows[pos][i] for pos in POSITIONS}
                        self.samples.append(sample_dict)
                        self.labels.append(ACT2IDX[activity])


        if not self.samples:
            raise RuntimeError("No valid multi-position samples found")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dict = {pos: torch.from_numpy(self.samples[idx][pos]).float() for pos in POSITIONS}
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample_dict, label