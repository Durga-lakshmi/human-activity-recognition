import numpy as np
import torch

# ----------------------------------------------------------------


def apply_stage2_crop_batch(x, crop_len=64):
    """
    x: [B, T, C]
    return:
      x_masked: [B, T, C]
      mask:     [B, T] bool
    """
    B, T, C = x.shape
    device = x.device
    x_out = x.clone()
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)

    for i in range(B):
        t0 = torch.randint(0, T - crop_len + 1, (1,), device=device).item()
        mask[i, t0:t0 + crop_len] = True
        x_out[i, ~mask[i]] = 0.0

    return x_out, mask

def compute_sample_weights(labels, power=0.5):
    """
    labels: 1D array-like of class indices (may contain -1 for 'ignore' / transition)
    power:  reweighting exponent, e.g. 0.5

    Returns:
    sample_weights       : [N], sampling weight for each sample (samples with label < 0 get weight = 0)
    class_counts         : [num_classes], number of samples per class (counting only labels >= 0)
    sample_class_weights : [num_classes], per-class weights (for use in loss_class_weights, etc.)
    """
    labels = np.asarray(labels, dtype=np.int64)

    mask_valid = labels >= 0
    labels_valid = labels[mask_valid]

    if labels_valid.size == 0:
        raise ValueError("No valid labels (>=0) found in compute_sample_weights.")

    num_classes = labels_valid.max() + 1
    class_counts = np.bincount(labels_valid, minlength=num_classes).astype(np.float32)

    nonzero = class_counts > 0
    freq = np.zeros_like(class_counts)
    freq[nonzero] = class_counts[nonzero] / class_counts[nonzero].sum()

    # class weight ~ freq^(-power)
    class_weights = np.zeros_like(freq)
    class_weights[nonzero] = freq[nonzero] ** (-power)

    # normalize class weights to sum to 1 (optional, but can help with stability)
    if class_weights.sum() > 0:
        class_weights = class_weights / class_weights.sum()

    # assign sample weights based on class weights, samples with label < 0 get weight = 0
    sample_weights = np.zeros_like(labels, dtype=np.float32)
    sample_weights[mask_valid] = class_weights[labels_valid]

    return sample_weights, class_counts, class_weights



# ------------------------------------------------------------------
def compute_sample_weights_0(labels, num_classes=None, eps=1e-8, power=0.5):
    """
    Robust sample-weight computation for WeightedRandomSampler.
    Supports:
      - labels shape (N,) integer class ids
      - labels shape (N,1)
      - labels shape (N,C) one-hot / soft labels -> argmax to ids
    """

    labels = np.asarray(labels)

    # 1) squeeze (N,1) -> (N,)
    labels = np.squeeze(labels)

    # 2) if still not 1D, assume one-hot/soft labels: (N,C) -> (N,)
    if labels.ndim > 1:
        labels = labels.argmax(axis=-1)

    # 3) ensure integer and 1D
    labels = labels.astype(np.int64).reshape(-1)

    if num_classes is None:
        num_classes = int(labels.max()) + 1

    # bincount requires non-negative ints
    if labels.min() < 0:
        raise ValueError(f"labels must be >=0 for bincount, got min={labels.min()}")

    class_counts = np.bincount(labels, minlength=num_classes)
    class_weights = 1.0 / (class_counts + eps) ** power
    sample_weights = class_weights[labels]

    return torch.DoubleTensor(sample_weights), class_counts, class_weights

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


