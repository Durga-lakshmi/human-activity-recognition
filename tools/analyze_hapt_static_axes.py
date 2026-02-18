import torch

@torch.no_grad()
def analyze_hapt_static_axes(
    data_loader,
    static_classes=(3, 4, 5),
    device=None,
    use_abs=True,
):
    """
    在 HAPT 的 DataLoader 上统计静态类 3/4/5 的 acc_x/acc_y/acc_z 分布。

    假设：
      - batch 是 (x, y) 或 (x, y_state, y_trans)
      - x: [B, T, C]，前 3 维是 acc_x, acc_y, acc_z
      - y: [B]，标签 0..5（3/4/5 是静态类）

    参数:
      data_loader: 你的 train_loader 或 test_loader
      static_classes: 要分析的类，默认 (3,4,5)
      device: 如果为 None，就用数据自己的 device；否则 x/y 会被搬到这个 device
      use_abs: 是否用 |g| 做比较（一般建议 True）

    功能：
      - 对每个静态类 c：
          * 统计窗口数 N_c
          * 统计 acc 均值 g_vec = mean_t(acc[t]) 的平均值
          * 统计 “哪一轴最大” 的频率分布 (argmax(|g|))
      - 在控制台打印结果
    """
    static_classes = list(static_classes)

    axis_hist = {}
    g_sum = {}
    count = {c: 0 for c in static_classes}

    inited = False  # 延迟在首个 batch 上初始化统计张量

    for batch in data_loader:
        # 兼容 (x, y) 和 (x, y_state, y_trans)
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch

        if device is not None:
            x = x.to(device)
            y = y.to(device)
        else:
            # 不指定的话就用数据自己的 device
            x = x.to(x.device)
            y = y.to(x.device)

        if not inited:
            cur_device = x.device
            axis_hist = {
                c: torch.zeros(3, dtype=torch.long, device=cur_device)
                for c in static_classes
            }
            g_sum = {
                c: torch.zeros(3, dtype=torch.float64, device=cur_device)
                for c in static_classes
            }
            inited = True

        y = y.long()              # [B]
        acc = x[..., :3]          # acc_x, acc_y, acc_z -> [B, T, 3]
        g_vec = acc.mean(dim=1)   # [B, 3]

        g_comp = g_vec.abs() if use_abs else g_vec
        axis_max = g_comp.argmax(dim=1)  # [B], 取值 0/1/2

        for c in static_classes:
            mask_c = (y == c)
            if mask_c.any():
                g_c = g_vec[mask_c]          # [Nc, 3]
                axis_c = axis_max[mask_c]    # [Nc]

                g_sum[c] += g_c.double().sum(dim=0)
                count[c] += int(mask_c.sum().item())

                for k in range(3):
                    axis_hist[c][k] += (axis_c == k).sum()

    # 打印结果
    print("\n===== HAPT 静态类 acc 轴分布分析 =====")
    axis_names = ["x", "y", "z"]

    for c in static_classes:
        if count[c] == 0:
            print(f"\nClass {c}: 没有样本")
            continue

        mean_g = (g_sum[c] / count[c]).cpu().numpy()
        hist = axis_hist[c].cpu().numpy().astype(float)
        hist_ratio = hist / hist.sum()

        print(f"\n--- Class {c} ---")
        print(f"窗口数: {count[c]}")
        print(
            "g_vec 均值 (acc_x, acc_y, acc_z): "
            f"[{mean_g[0]:.4f}, {mean_g[1]:.4f}, {mean_g[2]:.4f}]"
        )
        print("轴最大次数 (argmax(|g|))：")
        for i in range(3):
            print(
                f"  axis {axis_names[i]}: "
                f"{int(hist[i])} 次 ({hist_ratio[i]*100:.2f}%)"
            )


@torch.no_grad()
def analyze_train_and_test(
    train_loader,
    test_loader,
    static_classes=(3, 4, 5),
    device=None,
    use_abs=True,
):
    """
    同时查看训练集和测试集的静态类 acc 轴分布。
    """
    print("\n================ TRAIN SET ================")
    analyze_hapt_static_axes(
        data_loader=train_loader,
        static_classes=static_classes,
        device=device,
        use_abs=use_abs,
    )

    print("\n================ TEST SET ================")
    analyze_hapt_static_axes(
        data_loader=test_loader,
        static_classes=static_classes,
        device=device,
        use_abs=use_abs,
    )
