import torch
from collections import defaultdict

# 你需要：loader 输出 (x, y_state, y_trans) ；x: [B,T,C]
# y_state: [B] in {0..5} ; y_trans: [B] in {0,1}

@torch.no_grad()
def compute_canonical_gdir(loader, device="cuda"):
    stats = {3: [], 4: [], 5: []}  # 3=SITTING, 4=STANDING, 5=LAYING (按你当前 state label 定义)

    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            x, y_state, y_trans = batch
        else:
            raise ValueError("loader must yield (x, y_state, y_trans)")

        x = x.to(device)                 # [B,T,C]
        y_state = y_state.to(device)     # [B]
        y_trans = y_trans.to(device)     # [B]

        acc = x[..., :3]                 # [B,T,3]
        g_mean = acc.mean(dim=1)         # [B,3]
        g_dir = g_mean / (g_mean.norm(dim=-1, keepdim=True) + 1e-8)  # [B,3]

        mask = (y_trans == 0) & (y_state >= 3) & (y_state <= 5)
        if not mask.any():
            continue

        for k in [3, 4, 5]:
            sel = g_dir[(y_state == k) & mask]
            if sel.numel() > 0:
                stats[k].append(sel.detach().cpu())

    canon = {}
    for k in [3, 4, 5]:
        if len(stats[k]) == 0:
            raise RuntimeError(f"No samples collected for state {k}. Check labels/mask.")
        g_all = torch.cat(stats[k], dim=0)  # [N,3]
        canon[k] = g_all.mean(dim=0)        # [3]
        canon[k] = canon[k] / (canon[k].norm() + 1e-8)  # normalize again

    print("=== Canonical g_dir ===")
    for k in [3, 4, 5]:
        print(k, canon[k].numpy())

    # 可选：保存到文件，后续训练直接加载
    # torch.save(canon, "artifacts/canonical_gdir.pt")
    return canon