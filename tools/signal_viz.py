import numpy as np
import matplotlib.pyplot as plt

def extract_action_segment(X, y, target_label, min_len=200):
    """
    从连续序列中，截取一段 target_label 的连续区间
    """
    mask = (y == target_label)
    idx = np.where(mask)[0]

    if len(idx) == 0:
        raise ValueError(f"Label {target_label} not found")

    # 找最长连续段
    splits = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    seg = max(splits, key=len)

    if len(seg) < min_len:
        raise ValueError("Segment too short")

    return X[seg], y[seg]

def plot_raw_acc_gyro(X, fs=50.0):
    t = np.arange(len(X)) / fs

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Acc
    axes[0].plot(t, X[:, 0], label="acc_x")
    axes[0].plot(t, X[:, 1], label="acc_y")
    axes[0].plot(t, X[:, 2], label="acc_z")
    axes[0].set_ylabel("Acc (normed)")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Raw Accelerometer")

    # Gyro
    axes[1].plot(t, X[:, 3], label="gyro_x")
    axes[1].plot(t, X[:, 4], label="gyro_y")
    axes[1].plot(t, X[:, 5], label="gyro_z")
    axes[1].set_ylabel("Gyro (normed)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(loc="upper right")
    axes[1].set_title("Raw Gyroscope")

    plt.tight_layout()
    plt.show()

def plot_motion_intensity(X, fs=50.0):
    t = np.arange(len(X)) / fs

    plt.figure(figsize=(12, 3))
    plt.plot(t, X[:, 6], label="acc_norm")
    plt.plot(t, X[:, 7], label="gyro_norm")
    plt.plot(t, X[:, 8], label="jerk")

    plt.xlabel("Time (s)")
    plt.ylabel("Intensity")
    plt.title("Motion Intensity Features")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_posture(X, fs=50.0):
    t = np.arange(len(X)) / fs

    plt.figure(figsize=(12, 3))
    plt.plot(t, X[:, 9], label="pitch")
    plt.plot(t, X[:,10], label="roll")

    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.title("Posture Features")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_features_column_and_save(
    X,
    fs=50.0,
    save_path=None,
    show=False
):
    """
    三行一列（竖排）：
    1) Raw acc + gyro   （最高）
    2) Motion intensity （中）
    3) Posture          （低）
    """
    assert X.shape[1] >= 11, "Need at least 11 channels"

    t = np.arange(len(X)) / fs

    fig, axes = plt.subplots(
        3, 1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.8, 1.5]}
    )

    # =========================
    # (1) Raw acc + gyro
    # =========================
    ax = axes[0]
    ax.plot(t, X[:, 0], label="acc_x")
    ax.plot(t, X[:, 1], label="acc_y")
    ax.plot(t, X[:, 2], label="acc_z")
    ax.plot(t, X[:, 3], "--", label="gyro_x")
    ax.plot(t, X[:, 4], "--", label="gyro_y")
    ax.plot(t, X[:, 5], "--", label="gyro_z")

    ax.set_ylabel("Raw signals")
    ax.set_title("Raw Sensor Signals")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3)

    # =========================
    # (2) Motion intensity
    # =========================
    ax = axes[1]
    ax.plot(t, X[:, 6], label="acc_norm")
    ax.plot(t, X[:, 7], label="gyro_norm")
    ax.plot(t, X[:, 8], label="jerk")

    ax.set_ylabel("Intensity")
    ax.set_title("Motion Intensity")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # =========================
    # (3) Posture
    # =========================
    ax = axes[2]
    ax.plot(t, X[:, 9], label="pitch")
    ax.plot(t, X[:,10], label="roll")

    ax.set_ylabel("Angle (rad)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Posture")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


