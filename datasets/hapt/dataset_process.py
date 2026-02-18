import os
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt






# ---------------------------------------
#            DATA AUGMENTATION
# ---------------------------------------

def compute_sample_weights(
    labels,
    num_classes=None,
    eps=1e-8,
    power=0.5
):
    """
    为 HAR 等不平衡分类任务计算 sample-level weights
    （用于 WeightedRandomSampler）

    Args:
        labels (array-like): shape (N,), 每个 window 的类别 id（int, 从 0 开始）
        num_classes (int, optional): 类别总数，若为 None 自动推断
        eps (float): 防止除零
        power (float): 频次的幂指数
                       0.5 = 1/sqrt(freq)（推荐）
                       1.0 = 1/freq（不推荐，太激进）

    Returns:
        sample_weights (torch.DoubleTensor): shape (N,)
        class_counts   (np.ndarray): shape (C,)
        class_weights  (np.ndarray): shape (C,)
    """
    labels = np.asarray(labels)

    if num_classes is None:
        num_classes = labels.max() + 1

    # 1️⃣ 每类频次
    class_counts = np.bincount(labels, minlength=num_classes)

    # 2️⃣ 每类权重（平滑）
    class_weights = 1.0 / (class_counts + eps) ** power

    # 3️⃣ 映射到 sample-level
    sample_weights = class_weights[labels]

    return (
        torch.DoubleTensor(sample_weights),
        class_counts,
        class_weights
    )

def har_augment(
    X,
    p_noise=0.5,
    p_scale=0.5,
    p_rotate=0.5,
    sigma=0.02,
    scale_range=(0.9, 1.1),
    max_angle_deg=15
):
    """
    Combined HAR augmentation.
    Call this in Dataset.__getitem__ (train only).
    """
    if np.random.rand() < p_noise:
        X = gaussian_noise(X, sigma)

    if np.random.rand() < p_scale:
        X = scaling(X, scale_range)

    if np.random.rand() < p_rotate:
        X = axis_rotation(X, max_angle_deg)

    return X


def gaussian_noise(X, sigma=0.02):
    """Add Gaussian noise (z-score normalized data)."""
    return X + np.random.normal(0.0, sigma, size=X.shape)


def scaling(X, scale_range=(0.9, 1.1)):
    """Scale all channels uniformly."""
    scale = np.random.uniform(*scale_range)
    return X * scale


def random_rotation_matrix(max_angle_deg=15):
    """Generate a random 3D rotation matrix."""
    angle = np.deg2rad(np.random.uniform(-max_angle_deg, max_angle_deg))
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)

    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c

    return np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C]
    ])


def axis_rotation(X, max_angle_deg=15):
    """
    Apply axis rotation.
    X shape: (T, C), where C = 3 (acc or gyro) or 6 (acc + gyro)
    """
    T, C = X.shape
    R = random_rotation_matrix(max_angle_deg)

    if C == 3:
        return X @ R.T
    elif C == 6:
        acc  = X[:, 0:3] @ R.T
        gyro = X[:, 3:6] @ R.T
        return np.concatenate([acc, gyro], axis=1)
    else:
        raise ValueError("axis_rotation supports only C=3 or C=6")