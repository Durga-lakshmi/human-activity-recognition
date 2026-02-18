"""
split_analysis.py

功能：
- 对 train / val / test 做数据分布诊断
- 自动画图并保存到 config 指定路径

使用方式（示例）：
    from split_analysis import run_split_analysis
    run_split_analysis(
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        save_dir=cfg.split_analysis.save_dir
    )
"""

import os
import numpy as np
from collections import Counter

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# ================================
# 工具函数
# ================================
def to_NTC(x: np.ndarray) -> np.ndarray:
    """
    统一成 (N, T, C)
    """
    if x.ndim != 3:
        raise ValueError(f"Expect 3D array, got {x.shape}")
    if x.shape[1] < x.shape[2]:
        return np.transpose(x, (0, 2, 1))  # (N, C, T) -> (N, T, C)
    return x


def save_fig(save_dir: str, name: str):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, name)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[Saved] {path}")


# ================================
# 1. 标签分布
# ================================
def plot_label_distribution(y_train, y_val, y_test, save_dir):
    splits = {
        "train": y_train,
        "val": y_val,
        "test": y_test,
    }

    all_classes = sorted(
        set(y_train.tolist()) |
        set(y_val.tolist()) |
        set(y_test.tolist())
    )

    x = np.arange(len(all_classes))
    width = 0.25

    plt.figure(figsize=(8, 4))
    for i, (name, y) in enumerate(splits.items()):
        cnt = Counter(y.tolist())
        freq = np.array([cnt.get(c, 0) for c in all_classes], dtype=float)
        freq /= freq.sum()
        plt.bar(x + i * width, freq, width, label=name)

    plt.xticks(x + width, all_classes)
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.title("Label distribution (train / val / test)")
    plt.legend()

    save_fig(save_dir, "label_distribution.png")


# ================================
# 2. 基础统计特征
# ================================
def plot_basic_feature_stats(X_train, X_val, X_test, save_dir):
    X_train = to_NTC(X_train)
    X_val   = to_NTC(X_val)
    X_test  = to_NTC(X_test)

    stats = {}
    for name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
        stats[name] = (X.mean(), X.std())

    names = list(stats.keys())
    means = [stats[n][0] for n in names]
    stds  = [stats[n][1] for n in names]

    plt.figure(figsize=(6, 4))
    plt.bar(names, means, yerr=stds, capsize=6)
    plt.ylabel("Value")
    plt.title("Global mean ± std")

    save_fig(save_dir, "global_mean_std.png")


# ================================
# 3. PCA 分布
# ================================
def plot_pca_projection(X_train, X_val, X_test, save_dir, max_samples=2000):
    X_train = to_NTC(X_train)
    X_val   = to_NTC(X_val)
    X_test  = to_NTC(X_test)

    def prep(x):
        n = min(len(x), max_samples)
        idx = np.random.choice(len(x), n, replace=False)
        return x[idx].reshape(n, -1)

    Xtr = prep(X_train)
    Xva = prep(X_val)
    Xte = prep(X_test)

    X_all = np.concatenate([Xtr, Xva, Xte], axis=0)
    split_ids = (
        ["train"] * len(Xtr) +
        ["val"]   * len(Xva) +
        ["test"]  * len(Xte)
    )

    X_all = StandardScaler().fit_transform(X_all)
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(X_all)

    plt.figure(figsize=(6, 6))
    for name, marker in [("train", "o"), ("val", "^"), ("test", "s")]:
        idx = [i for i, s in enumerate(split_ids) if s == name]
        plt.scatter(Z[idx, 0], Z[idx, 1], s=10, alpha=0.4, label=name, marker=marker)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA projection (explained var = {pca.explained_variance_ratio_.sum():.2f})")
    plt.legend()

    save_fig(save_dir, "pca_split_projection.png")


# ================================
# 4. Domain classifier
# ================================
def domain_classifier_score(X_train, X_test, save_dir, max_samples=3000):
    X_train = to_NTC(X_train)
    X_test  = to_NTC(X_test)

    def prep(x):
        n = min(len(x), max_samples)
        idx = np.random.choice(len(x), n, replace=False)
        return x[idx].reshape(n, -1)

    Xtr = prep(X_train)
    Xte = prep(X_test)

    X = np.concatenate([Xtr, Xte], axis=0)
    y = np.array([0] * len(Xtr) + [1] * len(Xte))  # 0=train, 1=test

    X = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=1000)
    acc = cross_val_score(clf, X, y, cv=5, scoring="accuracy")

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "domain_classifier.txt")
    with open(out_path, "w") as f:
        f.write(f"Train vs Test domain accuracy: {acc.mean():.4f} ± {acc.std():.4f}\n")

    print(f"[Saved] {out_path}")


# ================================
# 统一入口（你最推荐用这个）
# ================================
def run_split_analysis(
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    save_dir: str,
):
    plot_label_distribution(y_train, y_val, y_test, save_dir)
    plot_basic_feature_stats(X_train, X_val, X_test, save_dir)
    plot_pca_projection(X_train, X_val, X_test, save_dir)
    domain_classifier_score(X_train, X_test, save_dir)
