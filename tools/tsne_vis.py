import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

def run_tsne_visualization(
    model,
    loader,
    device,
    max_samples: int = 5000,
    perplexity: float = 30.0,
    save_path: str | None = None,
):
    model.eval()

    all_feats = []
    all_labels = []
    total = 0

    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            y = batch[1].to(device)

            # 1) 优先尝试：模型支持 return_feat
            try:
                out = model(x, return_feat=True, return_phys_loss=False)
            except TypeError:
                # 2) 不支持这些参数，就退回普通 forward
                out = model(x)

            # ========= 统一解析 out =========
            feat = None

            # 情况 A：字典（包含 feat）
            if isinstance(out, dict):
                if "feat" in out:
                    feat = out["feat"]
                elif "logits" in out:
                    feat = out["logits"]
                else:
                    raise RuntimeError("dict 输出中没有 'feat' 也没有 'logits'，无法做 t-SNE")

            # 情况 B：tuple / list
            elif isinstance(out, (tuple, list)):
                # 最后一项当成特征（比如 (logits, feat) 或 (logits, phys_loss, feat)）
                feat = out[-1]

            # 情况 C：直接是 Tensor
            elif isinstance(out, torch.Tensor):
                # 说明模型没实现 return_feat，那就直接用输出当特征
                feat = out

            else:
                raise RuntimeError(
                    f"model(...) 返回类型 {type(out)}，无法解析为特征，用于 t-SNE"
                )

            all_feats.append(feat.detach().cpu())
            all_labels.append(y.detach().cpu())

            total += feat.size(0)
            if total >= max_samples:
                break

    if len(all_feats) == 0:
        print("[t-SNE] No features collected. Check your loader / labels.")
        return

    X = torch.cat(all_feats, dim=0).numpy()   # [N, D]
    y = torch.cat(all_labels, dim=0).numpy()  # [N]

    print(f"================== t-SNE Analyse ==================")
    print(f"[t-SNE] Using {X.shape[0]} samples, feature dim = {X.shape[1]}")
    print("labels in tsne:", np.unique(y))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=200,
        init="pca",
        random_state=0,
        verbose=1,
    )
    X_2d = tsne.fit_transform(X)


    custom_palette = [
    "#0E8088",  
    "#10739E",  
    "#67AB9F",  
    "#97D077",  
    "#7EA6E0",  
    "#23445D",  
    "#EA6B66",  
    "#A680B8",  
    "#BAC8D3",
    "#A20025",
    "#FFB570",
    "#FF9DA7",
    ]

    # 如果你的 y 是 1..12，就减 1 变成 0..11
    labels = y - 1

    colors = ListedColormap(custom_palette)

    CLASS_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
    "#aec7e8",  # light blue
    "#ffbb78",  # light orange
    ]
    cmap = ListedColormap(CLASS_COLORS)

    cmap = plt.get_cmap("tab20c")

    # 明确指定 tab20c 的索引
    idx_locomotion = [0, 1, 2]                 # 前 3 个
    idx_static     = [5, 6, 7]                 # 中间 3 个
    #idx_transition = [14, 15, 16, 17, 18, 19]   # 后 6 个

    # 外部黄色系：ColorBrewer
    ylorbr = plt.get_cmap("Greens")
    colors_transition = ylorbr(np.linspace(0.45, 0.9, 6))  # 6 个同色系黄

    # 直接取 RGBA
    color_list = (
        [cmap(i) for i in idx_locomotion] +
        [cmap(i) for i in idx_static] +
        [colors_transition[i] for i in range(6)]
    )

    #color_list = [cmap(i) for i in range(12)]
    # 3) 为每个样本生成颜色数组（关键一步）
    #y_np = np.asarray(y, dtype=int)           # y: shape [N]，取值 1..12
    #point_colors = np.array([color_list[k-1] for k in y_np])

    # ===== 1) 定义 3 个 base colormap =====
    blue_cmap   = plt.get_cmap("Blues")
    orange_cmap = plt.get_cmap("Oranges")
    green_cmap  = plt.get_cmap("Greens")

    # ===== 2) 从同一 colormap 里取不同深浅 =====
    # 注意：不要取 0.0 或 1.0，视觉不好
    colors_locomotion = blue_cmap(np.linspace(0.55, 0.85, 3))     # 3 个蓝
    colors_static     = orange_cmap(np.linspace(0.55, 0.85, 3))   # 3 个橙
    colors_transition = green_cmap(np.linspace(0.45, 0.85, 6))    # 6 个绿

    # ===== 3) 合并成最终 color_list（顺序 = label 1..12）=====
    #color_list = (
    #    list(colors_locomotion) +
    #    list(colors_static) +
    #    list(colors_transition)
    #)
    y_np = np.asarray(y, dtype=int)   # y ∈ [1..12]
    point_colors = np.array([color_list[k-1] for k in y_np])






    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        #c=y,
        c=point_colors,        # >>> 改动 1：指定离散 colormap（12 类用 tab20）
        s=10,                 # >>> 改动 2：点稍微变大，poster 上更清楚
        alpha=0.85,          # >>> 改动 3：加透明度，重叠区域更有层次
        edgecolors="none",    # >>> 改动 4：去掉点边框，避免视觉噪声
        #linewidths=0.25
    )



    #plt.colorbar(scatter) # >>> 改动 5：建议关闭（多类别 colorbar 没信息量）

      # --------------------------
    # 2) 用 ACTIVITY_MAP_1_12 生成 legend
    # --------------------------
    handles = []
    for k, name in ACTIVITY_MAP_1_12.items():
        # k: 1..12，对应 color_list[k-1]
        handles.append(
            Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                markersize=6,
                markerfacecolor=color_list[k - 1],
                markeredgecolor="none",
                label=name,
            )
        )

    plt.legend(
        handles=handles,
        title="Activities",
        loc="upper left",
        bbox_to_anchor=(0.005, 0.98),  # 左上角，图内
        borderaxespad=0.0,
        frameon=False,
    )

    plt.title("t-SNE of Model Features ", fontsize=14)  # >>> 改动 6：标题字号明确
    plt.xticks([])            # >>> 改动 7：去掉 x 轴刻度
    plt.yticks([])            # >>> 改动 8：去掉 y 轴刻度
    #plt.axis("off")           # >>> 改动 9：t-SNE 无绝对坐标，poster 更干净
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=400,          # >>> 改动 10：提高分辨率，适合 A0 poster
            bbox_inches="tight"
        )
    else:
        plt.show()



    print(f"[t-SNE] Visualization done.")


    
ACTIVITY_MAP_1_12 = {
    1:  "walking",
    2:  "walking_up",
    3:  "walking_down",
    4:  "sitting",
    5:  "standing",
    6:  "laying",
    7:  "stand_to_sit",
    8:  "sit_to_stand",
    9:  "sit_to_lie",
    10: "lie_to_sit",
    11: "stand_to_lie",
    12: "lie_to_stand",
}