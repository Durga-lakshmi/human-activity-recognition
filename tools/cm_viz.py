import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm,ListedColormap
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.cm as mpl_cm


def plot_confusion_matrix(
    y_true,
    y_pred,
    num_classes,
    save_path=None,
):
    """
    Poster-ready confusion matrix (row-normalized)

    y_true, y_pred : shape (N,), labels in [0, num_classes-1]
    """

    # 计算混淆矩阵
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(num_classes)
    )

    # 行归一化（每一行和为 1）
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # cm_norm 是 (12, 12) 的 row-normalized confusion matrix
    annot_mat = np.empty_like(cm_norm, dtype=object)

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            if cm_norm[i, j] < 1e-6:     # 认为是 0
                annot_mat[i, j] = ""     # 不显示
            else:
                annot_mat[i, j] = f"{cm_norm[i, j]:.2f}"


    plt.figure(figsize=(6, 5))

    custom_palette = ["#D6604D","#F2A481","#FBD8C3","#F8F4F2","#D4E6EF","#99C8E0","#4D9AC7",   "#276FAF","#1B3B70"]
    colors = ListedColormap(custom_palette)
    
    base_cmap = mpl_cm.get_cmap("RdYlBu_r")
    colors = base_cmap(np.linspace(0.05, 0.8, 256))
    deep_blue_cmap = plt.matplotlib.colors.ListedColormap(colors)

    #colors = [custom_palette[label] for label in y]

    sns.heatmap(
        cm_norm,
        cmap="Blues",          # ★ poster 首选
        norm=PowerNorm(gamma=0.3), # 强化高值对比
        vmin=0.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        square=True,
        cbar=True,
        linewidths=0,
        linecolor="white",
        annot_kws={
        "color": "white",   # ← 强制白字
        "fontsize": 8,
        },
    )

    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Row-normalized Prob. (after)", fontsize=14)

    # 标签显示成 1..K（如果你原来是 0..K-1）
    plt.xticks(
        ticks=np.arange(num_classes) + 0.5,
        labels=np.arange(1, num_classes + 1),
        rotation=0,
        fontsize=9
    )
    plt.yticks(
        ticks=np.arange(num_classes) + 0.5,
        labels=np.arange(1, num_classes + 1),
        rotation=0,
        fontsize=9
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
        print(f"[Confusion Matrix] Saved to {save_path}")
    else:
        plt.show()


    plt.close()

# =========================
# 2) 计数版本（整数）
# =========================
def plot_confusion_matrix_counts(
    y_true,
    y_pred,
    num_classes,
    save_path=None,
):
    """
    Confusion matrix with raw counts (no normalization).
    0 不显示，其它显示整数。
    """

    cm_counts = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(num_classes)
    )

    # ---------- annotation：整数 & 不显示 0 ----------
    annot_mat = np.empty_like(cm_counts, dtype=object)
    for i in range(cm_counts.shape[0]):
        for j in range(cm_counts.shape[1]):
            if cm_counts[i, j] == 0:
                annot_mat[i, j] = ""
            else:
                annot_mat[i, j] = str(cm_counts[i, j])

    # ---------- colormap 同一套：蓝 → 暗橙，不到红 ----------
    base_cmap = mpl_cm.get_cmap("RdYlBu_r")
    colors = base_cmap(np.linspace(0.1, 0.8, 256))
    deep_blue_cmap = ListedColormap(colors)

    vmax = cm_counts.max()

    custom_palette = ["#1B3B70","#276FAF","#4D9AC7","#99C8E0","#D4E6EF","#F8F4F2","#FBD8C3",   "#F2A481","#D6604D"]
    colors = ListedColormap(custom_palette)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm_counts,
        cmap="Blues",          # ★ poster 首选
        norm=PowerNorm(gamma=0.3),   # 轻微非线性：中等计数更突出
        vmin=0.0,
        vmax=vmax,
        annot=True,
        fmt="",
        square=True,
        cbar=True,
        linewidths=0.0,
        linecolor="white",
        annot_kws={
            "color": "white",
            "fontsize": 8,
        },
    )

    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Confusion Matrix - Counts (after)", fontsize=14)

    plt.xticks(
        ticks=np.arange(num_classes) + 0.5,
        labels=np.arange(1, num_classes + 1),
        rotation=0,
        fontsize=9,
    )
    plt.yticks(
        ticks=np.arange(num_classes) + 0.5,
        labels=np.arange(1, num_classes + 1),
        rotation=0,
        fontsize=9,
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
        print(f"[Confusion Matrix - Counts (before)] Saved to {save_path}")
    else:
        plt.show()