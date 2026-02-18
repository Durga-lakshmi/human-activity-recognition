import os
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
# -----------------------------
# Normalization
# -----------------------------
def zscore_normalize(X, mean, std, eps=1e-8):
    return (X - mean) / (std + eps)

# -----------------------------
# Window labeling
# -----------------------------
def label_window_majority(y_window, num_classes):
    labels = y_window[y_window > 0]  # positive labels only
    if len(labels) == 0:
        return None
    majority = np.bincount(labels).argmax()
    one_hot = np.zeros(num_classes, dtype=np.float32)
    one_hot[majority - 1] = 1
    return one_hot


# -----------------------------
# Compute train statistics
# -----------------------------
def compute_train_stats(cfg):
    all_data = []

    for user in cfg.dataset.train_users:
        X, _ = HAPT(cfg, split="train")._load_user(user)
        all_data.append(X)

    all_data = np.vstack(all_data)
    mean = all_data.mean(axis=0)
    std = all_data.std(axis=0)
    return mean, std

def compute_train_stats_0(cfg):
    ds = HAPT(cfg, split="train")  # only read once
    all_data = []

    for user in cfg.dataset.train_users:
        X, _ = ds._load_user(user)   # X: (n_win, win_len, C) / (N, C)
        all_data.append(X)

    all_data = np.concatenate(all_data, axis=0)

    # pass to compute mean/std
    if all_data.ndim == 2:
        mean = all_data.mean(axis=0)
        std  = all_data.std(axis=0)
    elif all_data.ndim == 3:
        mean = all_data.mean(axis=(0,1))  # (C,)
        std  = all_data.std(axis=(0,1))
    else:
        raise ValueError(f"Unexpected shape: {all_data.shape}")

    std = np.maximum(std, 1e-8)  # aviod /0
    return mean, std


# -----------------------------
# HAPT Dataset
# -----------------------------
class HAPT(Dataset):
    def __init__(self, cfg, split,  build_windows=True,mean=None, std=None, augment=False,crop=True):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.mean = mean
        self.std = std
        self.augment = augment
        self.crop = crop

        if split == "train":
            self.users = cfg.dataset.train_users
        elif split == "val":
            self.users = cfg.dataset.val_users
        elif split == "test":
            self.users = cfg.dataset.test_users
        else:
            raise ValueError(f"Invalid split: {split}")

        # Load labels once
        self.labels_all = np.loadtxt(os.path.join(cfg.dataset.path,  "labels.txt"))
        # Columns: exp_id, user_id, activity_id, start_idx, end_idx

        print(f"Building {split} dataset with users: {self.users}")

        if build_windows:
            self.X, self.y = self._build_windows(mean, std)

            # 转 numpy 方便检查
            #if torch.is_tensor(self.y):
            #    y_np = self.y.detach().cpu().numpy()
            #else:
            #    y_np = np.asarray(self.y)

            #print("Label raw shape:", y_np.shape)

            # squeeze 后再看
            #y_sq = np.squeeze(y_np)
            #print("Label squeezed shape:", y_sq.shape)

            # 如果是 one-hot / 多维
            #if y_sq.ndim > 1:
            #    y_id = y_sq.argmax(axis=-1)
            #else:
            #    y_id = y_sq

            #print("Label min / max:", y_id.min(), y_id.max())
            #print("Unique labels:", np.unique(y_id))

        



    # -----------------------------
    # Load a user's raw sensor data and map labels
    # -----------------------------
    def _load_user(self, user_id):
        raw_dir = os.path.join(self.cfg.dataset.path)

        # Find all accelerometer and gyro files for this user
        acc_files = sorted([f for f in os.listdir(raw_dir) if f.startswith("acc") and f"user{user_id:02d}" in f])
        gyro_files = sorted([f for f in os.listdir(raw_dir) if f.startswith("gyro") and f"user{user_id:02d}" in f])

        X_list, y_list = [], []

        # Filter label rows for this user
        labels_user = self.labels_all[self.labels_all[:,1] == user_id]

        for acc_file, gyro_file in zip(acc_files, gyro_files):
            acc_data = np.loadtxt(os.path.join(raw_dir, acc_file))
            gyro_data = np.loadtxt(os.path.join(raw_dir, gyro_file))
            X_exp = np.hstack([acc_data[:, :3], gyro_data[:, :3]])

            # Initialize label array
            y_exp = np.zeros(X_exp.shape[0], dtype=int)

            # Assign labels using start/end indices
            exp_id = int(acc_file.split("_")[1][3:])  # extract expXX
            labels_exp = labels_user[labels_user[:,0] == exp_id]

            for _, _, activity_id, start_idx, end_idx in labels_exp:
                start_idx = int(start_idx)
                end_idx = int(end_idx)
                y_exp[start_idx:end_idx] = int(activity_id)

                #print(f"User {user_id:02d} | {acc_file} + {gyro_file} | Activity {activity_id} | Indices {start_idx}-{end_idx} | Length {end_idx - start_idx}")

            X_list.append(X_exp)
            y_list.append(y_exp)

        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        return X, y

    # -----------------------------
    # Build sliding windows
    # -----------------------------
    def _build_windows(self, mean, std):
        X_windows, y_windows = [], []
        debug_records = []

        ws = self.cfg.sliding_window.window_size
        shift = self.cfg.sliding_window.window_shift
        purity_th = self.cfg.sliding_window.label_purity_th
        num_classes = self.cfg.num_classes
        debug_save_path = self.cfg.debug.save_path
        
        # 这两个可以先设 None（只启用 purity/amp）
        # 后面你在 train 统计好阈值表后，把 dict 塞进 self 或 cfg，再从这里读出来
        energy_range = getattr(self, "energy_range", None)      # dict: {label(1-based): (lo,hi)} or None
        var_threshold = getattr(self, "var_threshold", None)    # dict: {label(1-based): vlow} or None

        # 可选：把 amp_limit 也放到 cfg，没放就用 20.0
        amp_limit = getattr(self.cfg.sliding_window, "amp_limit", 20.0)

        for user in self.users:
            X, y = self._load_user(user)

            if mean is not None and std is not None:
                X = zscore_normalize(X, mean, std, self.cfg.eps)

            T = len(X)
            for i in range(0, T - ws + 1, shift):
                Xw = X[i:i + ws]
                yw = y[i:i + ws]

                # 1:
                #label = label_window_majority(yw, num_classes)
                #label, purity = label_window_majority_with_purity(yw, num_classes, purity_th)
                #if label is None:
                #    continue

                # 2:
                E = window_energy(Xw)
                V = window_variance(Xw)
                # label for one-hot
                ok, label, purity, y_label = filter_window_onehot(
                    x_window=Xw,
                    y_window=yw,
                    num_classes=num_classes,
                    energy_range=energy_range,      
                    var_threshold=var_threshold,    
                    purity_th=purity_th,
                    amp_limit=amp_limit
                )

                debug_records.append({
                    "label": int(y_label) if label is not None else -1,
                    "purity": float(purity) if purity is not None else np.nan,
                    "E": float(E),
                    "V": float(V),
                    "ok": bool(ok),
                })

                if not ok:
                    continue


                X_windows.append(Xw)
                y_windows.append(label)


        # 防止全被过滤导致 np.stack 报错
        if len(X_windows) == 0:
        # 这里的 C 需要从数据推断；如果拿不到就先返回空 tensor
            return torch.empty((0, ws, 0), dtype=torch.float32), \
                   torch.empty((0, num_classes), dtype=torch.float32)

        # 保存 debug（路径你自己决定）

        #if getattr(self.cfg, "debug", None) is not None and getattr(self.cfg.debug, "save_path", None):
        #    # 转成结构化数组更好存
        #    labels = np.array([r["label"] for r in debug_records], dtype=np.int32)
        #    purity = np.array([r["purity"] for r in debug_records], dtype=np.float32)
        #    E = np.array([r["E"] for r in debug_records], dtype=np.float32)
        #    V = np.array([r["V"] for r in debug_records], dtype=np.float32)
        #    ok = np.array([r["ok"] for r in debug_records], dtype=np.bool_)
        #    np.savez(debug_save_path, label=labels, purity=purity, E=E, V=V, ok=ok)

        #y_arr = np.array(y_windows, dtype=np.int64)
        #assert y_arr.min() >= 0 and y_arr.max() < num_classes, (y_arr.min(), y_arr.max(), num_classes)

        #return torch.tensor(np.stack(X_windows), dtype=torch.float32), \
        #    torch.tensor(y_arr, dtype=torch.long)

        return torch.tensor(np.stack(X_windows), dtype=torch.float32), \
               torch.tensor(np.stack(y_windows), dtype=torch.float32)

        #return torch.tensor(np.stack(X_windows), dtype=torch.float32), \
        #        torch.tensor(np.array(y_windows, dtype=np.int64), dtype=torch.long)
    # -----------------------------
    # Dataset API
    # -----------------------------
    def __getitem__(self, index):

        X = self.X[index]
        y_onehot = self.y[index]   # (12,)

        # 统一转 numpy（做增强/argmax）
        if torch.is_tensor(X):
            X = X.detach().cpu().numpy()
        else:
            X = np.asarray(X)

        if torch.is_tensor(y_onehot):
            y_onehot = y_onehot.detach().cpu().numpy()
        else:
            y_onehot = np.asarray(y_onehot)

        # one-hot -> class id (0..11)
        label_id = int(np.argmax(y_onehot))

        # train only augmentation
        if self.augment:
            # 1️⃣ 物理增强
            X = har_augment(X, p_noise=0.5, p_scale=0.5, p_rotate=0.3)
            if label_id in TRANSITION:
                X = har_augment(X, p_noise=0.7, p_scale=0.7, p_rotate=0.5)
                #X = window_cropping_np(X, min_crop=48)

            #else:

            #    if np.random.rand() < 0.3:
            #        X = window_cropping_np(X, min_crop=int(0.8 * X.shape[1]))

            # 2️⃣ Window Cropping（核心）
            #X = window_cropping_np(X, min_crop=64)


        # 返回 torch（collate 稳定）
        X = torch.from_numpy(X).float()
        y = torch.tensor(label_id, dtype=torch.long)



        return X, y

    def __len__(self):
        return len(self.y)



ACTIVITY_MAP_1_12 = {
    1:  "WALKING",
    2:  "WALKING_UP",
    3:  "WALKING_DOWN",
    4:  "SITTING",
    5:  "STANDING",
    6:  "LAYING",
    7:  "STAND_TO_SIT",
    8:  "SIT_TO_STAND",
    9:  "SIT_TO_LIE",
    10: "LIE_TO_SIT",
    11: "STAND_TO_LIE",
    12: "LIE_TO_STAND",
}


DYNAMIC = [0,1,2]
STATIC = [3,4,5]
TRANSITION = [6, 7, 8, 9, 10, 11]



# ---------------------------------------
#          DATA PROCESSING   
# ---------------------------------------

def window_energy(x_window):
    # x_window: (T, C)
    return np.mean(np.sum(x_window ** 2, axis=1))

def window_variance(x_window):
    return float(np.mean(np.var(x_window, axis=0)))


def label_window_majority_with_purity(y_window,num_classes,purity_th):
    labels = y_window[y_window > 0]
    if len(labels) == 0:
        return None, None, None

    counts = np.bincount(labels, minlength=num_classes + 1)
    majority_label = counts.argmax()   # still 1-based
    purity = counts[majority_label] / len(labels)
    
    if purity < purity_th:
        return None, purity, majority_label

    
    one_hot = np.zeros(num_classes, dtype=np.float32)
    one_hot[majority_label - 1] = 1.0
    return one_hot, purity, majority_label

def filter_window_onehot(
    x_window, y_window,
    num_classes,
    energy_range, var_threshold,
    purity_th=0.8,
    amp_limit=20.0,
    static_labels=None,          # e.g. {4,5,6}
    transition_labels=None       # 可选：转换类只做 purity
):
    # 默认：按你现在阈值结果推断的静态类
    if static_labels is None:
        static_labels = {4, 5, 6}
    if transition_labels is None:
        transition_labels = {7,8,9,10,11,12}

    # 1) NaN/Inf
    if not np.isfinite(x_window).all():
        return False, None, None, None

    # 2) physical amplitude
    if np.abs(x_window).max() > amp_limit:
        return False, None, None, None

    # 3) purity + majority -> onehot
    y_onehot, purity, label = label_window_majority_with_purity(y_window, num_classes, purity_th)
    if y_onehot is None:
        return False, None, purity, label

    E = window_energy(x_window)
    V = window_variance(x_window)

    # 4) 转换类：只做 purity（不做能量/方差硬筛）
    if label in transition_labels:
        return True, y_onehot, purity, label


    # 5) 静态类：只做能量上限（防爆点），不做 var_threshold
    if label in static_labels:
        if energy_range is not None and label in energy_range:
            _, hi = energy_range[label]
            if E > hi:
                return False, None, purity, label
        return True, y_onehot, purity, label

     # 6) 动态类：能量范围 + 方差下限
    if energy_range is not None:
        if label not in energy_range:
            return False, None, purity, label
        lo, hi = energy_range[label]
        if not (lo <= E <= hi):
            return False, None, purity, label


    # 4) energy range by label(1-based)
    if energy_range is not None:
        if label not in energy_range:
            return False, None, purity, label
        E = window_energy(x_window)
        lo, hi = energy_range[label]
        if not (lo <= E <= hi):
            return False, None, purity, label

    # 5) variance threshold for dynamic labels
    if var_threshold is not None and label in var_threshold:
        if window_variance(x_window) < var_threshold[label]:
            return False, None, purity, label

    return True, y_onehot, purity, label



def compute_train_thresholds(cfg, mean, std, save_path: str):
    ds = HAPT(cfg, split="train", build_windows=False)

    ws        = cfg.sliding_window.window_size
    shift     = cfg.sliding_window.window_shift
    purity_th = cfg.sliding_window.label_purity_th
    num_cls   = cfg.num_classes
    amp_limit = getattr(cfg.sliding_window, "amp_limit", 20.0)

    save_path = cfg.debug.save_threshold_path 
    debug_save_path = cfg.debug.save_path

    debug_records = []

    # 每类收集 E / V（1-based class id）
    E_by_c = {c: [] for c in range(1, num_cls + 1)}
    V_by_c = {c: [] for c in range(1, num_cls + 1)}

    for user in cfg.dataset.train_users:
        X, y = ds._load_user(user)   # X:(T,C), y:(T,)
        X = zscore_normalize(X, mean, std, cfg.eps)  # 与训练时一致：在归一化后统计阈值

        T = len(X)
        for i in range(0, T - ws + 1, shift):
            Xw = X[i:i+ws]
            yw = y[i:i+ws]

            # 先用 purity 判定窗口是否“干净”
            y_onehot, purity, label_id = label_window_majority_with_purity(yw, num_cls, purity_th)
            if y_onehot is None:
                # 收集该类的 E / V
                E = window_energy(Xw)
                V = window_variance(Xw)
                debug_records.append({
                    "label": int(label_id) if label_id is not None else -1,
                    "purity": float(purity) if purity is not None else np.nan,
                    "E": float(E),
                    "V": float(V),
                    "ok": False,
                })
                continue

            # 可选：跟训练过滤一致，先用 amp 去掉爆点窗口，避免污染阈值
            amp = np.percentile(np.abs(Xw), 99)
            if amp > amp_limit:
                continue

            # 收集该类的 E / V
            E = window_energy(Xw)
            V = window_variance(Xw)
            E_by_c[int(label_id)].append(float(E))
            V_by_c[int(label_id)].append(float(V))


            # debug：这里的 ok=True 表示“进入阈值统计集合”
            debug_records.append({
                "label": int(label_id),
                "purity": float(purity),
                "E": float(E),
                "V": float(V),
                "ok": True,
            })

    # 用分位数得到阈值（你可以按需调整）
    # energy_range: (1% , 99%)；var_threshold: 5% 作为 vlow
    energy_range = {}
    var_threshold = {}

  # 推荐：能量 5-95；动态类方差 10（如果你暂时不分动态/静态，就先全类给 vlow，但静态类后面会误伤）
    q_lo, q_hi = 0.05, 0.95
    q_vlow = 0.10

    for c in range(1, num_cls + 1):
        E_list = np.asarray(E_by_c[c], dtype=np.float32)
        V_list = np.asarray(V_by_c[c], dtype=np.float32)

        if len(E_list) < 50:
            continue

        lo, hi = np.quantile(E_list, [q_lo, q_hi])
        energy_range[c] = (float(lo), float(hi))

        # 注意：如果你后续要区分静态类，这里先都算出来也行，应用时对静态类不启用 var_threshold
        vlow = np.quantile(V_list, q_vlow)
        var_threshold[c] = float(vlow)

    # 6) 保存 thresholds
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(
        save_path,
        energy_range=np.array([[k, lo, hi] for k, (lo, hi) in energy_range.items()], dtype=np.float32),
        var_threshold=np.array([[k, v] for k, v in var_threshold.items()], dtype=np.float32),
    )

    # 7) 保存 debug windows（可选）
    if debug_save_path:
        os.makedirs(os.path.dirname(debug_save_path), exist_ok=True)
        labels = np.array([r["label"] for r in debug_records], dtype=np.int32)
        purity = np.array([r["purity"] for r in debug_records], dtype=np.float32)
        E_arr = np.array([r["E"] for r in debug_records], dtype=np.float32)
        V_arr = np.array([r["V"] for r in debug_records], dtype=np.float32)
        ok_arr = np.array([r["ok"] for r in debug_records], dtype=np.bool_)
        np.savez(debug_save_path, label=labels, purity=purity, E=E_arr, V=V_arr, ok=ok_arr)

    print("\n===== Energy Range (per class) =====")
    for c in sorted(energy_range.keys()):
        lo, hi = energy_range[c]
        print(f"class {c:02d}: E_lo={lo:.6f}, E_hi={hi:.6f}")

    print("\n===== Variance Threshold (per class) =====")
    for c in sorted(var_threshold.keys()):
        v = var_threshold[c]
        print(f"class {c:02d}: V_low={v:.6f}")

    return energy_range, var_threshold

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


def window_cropping_with_mask(x, min_crop=64):
    C, T = x.shape
    crop_len = random.randint(min_crop, T)
    start = random.randint(0, T - crop_len)

    cropped = torch.zeros_like(x)
    mask = torch.zeros(T, dtype=torch.bool)

    cropped[:, :crop_len] = x[:, start:start + crop_len]
    mask[:crop_len] = True

    return cropped, mask

def window_cropping_np(X, min_crop=64):
    """
    X: np.ndarray [C, T]
    """
    C, T = X.shape

    # 🔒 关键保护
    if T <= min_crop:
        return X.copy()   # 不 crop，直接返回

    crop_len = np.random.randint(min_crop, T + 1)
    start = np.random.randint(0, T - crop_len + 1)

    cropped = X[:, start:start + crop_len]

    if crop_len < T:
        pad = np.zeros((C, T - crop_len), dtype=X.dtype)
        cropped = np.concatenate([cropped, pad], axis=1)

    return cropped

# ---------------------------------------
#                  EDA
# ---------------------------------------
def plot_one_user_raw(cfg, user_id=1, exp_pick=0, max_len=None, save_path=None):
    """
    画一个 user 的某个实验(exp)的原始 6 通道数据 + label 时间轴
    - exp_pick: 选第几个实验文件（按排序后的 acc 文件顺序）
    - max_len: 只画前 N 个点（太长就截断）
    """
    raw_dir = os.path.join(cfg.dataset.path)

    # 找该 user 的文件
    acc_files = sorted([f for f in os.listdir(raw_dir) if f.startswith("acc") and f"user{user_id:02d}" in f])
    gyro_files = sorted([f for f in os.listdir(raw_dir) if f.startswith("gyro") and f"user{user_id:02d}" in f])

    assert len(acc_files) > 0, f"No acc files for user {user_id}"
    assert len(acc_files) == len(gyro_files), f"acc/gyro file count mismatch for user {user_id}"

    acc_file = acc_files[exp_pick]
    gyro_file = gyro_files[exp_pick]

    acc = np.loadtxt(os.path.join(raw_dir, acc_file))
    gyro = np.loadtxt(os.path.join(raw_dir, gyro_file))
    X = np.hstack([acc[:, :3], gyro[:, :3]])  # (T,6)

    # 读 labels，并铺到逐点 y
    labels_all = np.loadtxt(os.path.join(raw_dir, "labels.txt"))
    labels_user = labels_all[labels_all[:, 1] == user_id]

    exp_id = int(acc_file.split("_")[1][3:])  # 解析 expXX
    labels_exp = labels_user[labels_user[:, 0] == exp_id]

    y = np.zeros(X.shape[0], dtype=int)
    for _, _, act, s, e in labels_exp:
        s = int(s); e = int(e)
        y[s:e] = int(act)   # 注意：如果 end_idx 是 inclusive，这里要改成 y[s:e+1]

    # 截断显示
    if max_len is not None:
        X = X[:max_len]
        y = y[:max_len]

    t = np.arange(len(X))

    # 画图：上面 6 通道，下面 label
    fig = plt.figure(figsize=(14, 7))
    ax1 = plt.subplot2grid((7,1), (0,0), rowspan=6)
    ax2 = plt.subplot2grid((7,1), (6,0), rowspan=1, sharex=ax1)

    names = ["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"]
    for k in range(6):
        ax1.plot(t, X[:, k], label=names[k], linewidth=0.8)

    ax1.set_title(f"User {user_id:02d} | {acc_file} + {gyro_file}")
    ax1.set_ylabel("sensor value")
    ax1.legend(ncol=3, fontsize=9)

    ax2.step(t, y, where="post")
    ax2.set_ylabel("label")
    ax2.set_xlabel("time index")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()

    # 顺便打印一下标签分布，帮你判断是否对齐
    uniq, cnt = np.unique(y, return_counts=True)
    print("Label counts:", dict(zip(uniq.tolist(), cnt.tolist())))




def plot_one_user_raw_separate_axes(cfg, user_id=1, exp_pick=0, max_len=None, save_path=None):
    raw_dir = os.path.join(cfg.dataset.path)

    acc_files = sorted([f for f in os.listdir(raw_dir) if f.startswith("acc") and f"user{user_id:02d}" in f])
    gyro_files = sorted([f for f in os.listdir(raw_dir) if f.startswith("gyro") and f"user{user_id:02d}" in f])

    if len(acc_files) == 0:
        raise FileNotFoundError(f"No acc files for user {user_id} in {raw_dir}")
    if len(acc_files) != len(gyro_files):
        raise RuntimeError(f"acc/gyro count mismatch: {len(acc_files)} vs {len(gyro_files)}")

    acc_file = acc_files[exp_pick]
    gyro_file = gyro_files[exp_pick]

    acc = np.loadtxt(os.path.join(raw_dir, acc_file))
    gyro = np.loadtxt(os.path.join(raw_dir, gyro_file))
    X = np.hstack([acc[:, :3], gyro[:, :3]])  # (T,6)

    # labels
    labels_all = np.loadtxt(os.path.join(raw_dir, "labels.txt"))
    labels_user = labels_all[labels_all[:, 1] == user_id]

    exp_id = int(acc_file.split("_")[1][3:])  # expXX
    labels_exp = labels_user[labels_user[:, 0] == exp_id]

    y = np.zeros(X.shape[0], dtype=int)
    for _, _, act, s, e in labels_exp:
        s = int(s); e = int(e)
        y[s:e] = int(act)  # 若 end_idx inclusive -> 改成 y[s:e+1] = int(act)

    # truncate for display
    if max_len is not None:
        X = X[:max_len]
        y = y[:max_len]

    t = np.arange(len(X))
    names = ["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"]

    # 6 channels + 1 label row
    fig, axes = plt.subplots(
        nrows=7, ncols=1, sharex=True,
        figsize=(14, 10),
        gridspec_kw={"height_ratios": [1,1,1,1,1,1,0.6]}
    )

    fig.suptitle(f"User {user_id:02d} | {acc_file} + {gyro_file}", y=0.995)

    for k in range(6):
        axes[k].plot(t, X[:, k], linewidth=0.8)
        axes[k].set_ylabel(names[k])
        axes[k].grid(True, linewidth=0.3, alpha=0.6)

    axes[6].step(t, y, where="post", linewidth=1.0)
    axes[6].set_ylabel("label")
    axes[6].set_xlabel("time index")
    axes[6].grid(True, linewidth=0.3, alpha=0.6)

    plt.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()

    uniq, cnt = np.unique(y, return_counts=True)
    print("Label counts:", dict(zip(uniq.tolist(), cnt.tolist())))







def _load_one_exp_raw(cfg, user_id: int, exp_pick: int = 0):
    raw_dir = os.path.join(cfg.dataset.path)
    acc_files = sorted([f for f in os.listdir(raw_dir) if f.startswith("acc") and f"user{user_id:02d}" in f])
    gyro_files = sorted([f for f in os.listdir(raw_dir) if f.startswith("gyro") and f"user{user_id:02d}" in f])

    if len(acc_files) == 0:
        raise FileNotFoundError(f"No acc files for user {user_id} in {raw_dir}")
    if len(acc_files) != len(gyro_files):
        raise RuntimeError(f"acc/gyro count mismatch: {len(acc_files)} vs {len(gyro_files)}")

    acc_file = acc_files[exp_pick]
    gyro_file = gyro_files[exp_pick]

    acc = np.loadtxt(os.path.join(raw_dir, acc_file))
    gyro = np.loadtxt(os.path.join(raw_dir, gyro_file))
    X = np.hstack([acc[:, :3], gyro[:, :3]])  # (T,6)

    exp_id = int(acc_file.split("_")[1][3:])  # expXX
    return X, exp_id, acc_file, gyro_file


def plot_examples_by_activity_12(
    cfg,
    user_id=1,
    exp_pick=0,
    seconds=4,
    fs=50,
    max_examples_per_activity=1,
    save_dir=None,
    activity_map=ACTIVITY_MAP_1_12,
    pad_seconds=0.5,
):
    """
    从 labels.txt 区间中，为每个 activity(1..12) 画样例片段（最可信）
    - seconds: 每个样例展示时长（秒）
    - fs: 采样率（HAPT 常见 50Hz）
    - max_examples_per_activity: 每个动作画几个区间样例
    - pad_seconds: 在区间中间取片段，并在两侧留一点余量（防止切到边界）
    """
    X, exp_id, acc_file, gyro_file = _load_one_exp_raw(cfg, user_id, exp_pick)
    raw_dir = os.path.join(cfg.dataset.path)

    labels_all = np.loadtxt(os.path.join(raw_dir, "labels.txt"))
    labels_user = labels_all[labels_all[:, 1] == user_id]
    labels_exp = labels_user[labels_user[:, 0] == exp_id]

    # 按 activity_id 分组区间
    segments = {}
    for _, _, act, s, e in labels_exp:
        act = int(act)
        s = int(s); e = int(e)
        segments.setdefault(act, []).append((s, e))

    names = ["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"]
    win_len = int(seconds * fs)
    pad = int(pad_seconds * fs)

    for act in sorted(activity_map.keys()):
        if act not in segments or len(segments[act]) == 0:
            print(f"[WARN] activity {act} ({activity_map[act]}) not found in user{user_id:02d} exp{exp_id:02d}")
            continue

        for ex_i, (s, e) in enumerate(segments[act][:max_examples_per_activity]):
            seg_len = e - s
            if seg_len < 5:
                continue

            # 从区间中间取固定长度窗口
            mid = (s + e) // 2
            half = win_len // 2

            a = max(0, mid - half)
            b = min(len(X), mid + half)

            # 给一点边界余量（如果你希望看到过渡前后）
            a = max(0, a - pad)
            b = min(len(X), b + pad)

            Xseg = X[a:b]
            t = np.arange(len(Xseg)) / fs

            fig, axes = plt.subplots(nrows=6, ncols=1, sharex=True, figsize=(14, 9))
            fig.suptitle(
                f"User {user_id:02d} | exp{exp_id:02d} | {activity_map[act]} (id={act}) | seg[{s}:{e}] | plot[{a}:{b}]",
                y=0.995
            )

            for k in range(6):
                axes[k].plot(t, Xseg[:, k], linewidth=0.8)
                axes[k].set_ylabel(names[k])
                axes[k].grid(True, linewidth=0.3, alpha=0.6)

            axes[-1].set_xlabel("time (s)")
            plt.tight_layout(rect=[0, 0, 1, 0.97])

            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                out = os.path.join(
                    save_dir,
                    f"user{user_id:02d}_exp{exp_id:02d}_act{act:02d}_{activity_map[act]}_{ex_i}.png"
                )
                plt.savefig(out, dpi=200, bbox_inches="tight")

            plt.show()


def plot_examples_by_activity_12_new(
    cfg,
    user_id=1,
    exp_pick=0,
    seconds=4,
    fs=50,
    max_examples_per_activity=1,
    save_dir=None,
    activity_map=ACTIVITY_MAP_1_12,
    pad_seconds=0.5,
):
    """
    从 labels.txt 区间中，为每个 activity(1..12) 画样例片段（最可信）
    - seconds: 每个样例展示时长（秒）
    - fs: 采样率（HAPT 常见 50Hz）
    - max_examples_per_activity: 每个动作画几个区间样例
    - pad_seconds: 在区间中间取片段，并在两侧留一点余量（防止切到边界）
    """
    X, exp_id, acc_file, gyro_file = _load_one_exp_raw(cfg, user_id, exp_pick)
    raw_dir = os.path.join(cfg.dataset.path)

    labels_all = np.loadtxt(os.path.join(raw_dir, "labels.txt"))
    labels_user = labels_all[labels_all[:, 1] == user_id]
    labels_exp = labels_user[labels_user[:, 0] == exp_id]

    # 按 activity_id 分组区间
    segments = {}
    for _, _, act, s, e in labels_exp:
        act = int(act)
        s = int(s); e = int(e)
        segments.setdefault(act, []).append((s, e))

    win_len = int(seconds * fs)
    pad = int(pad_seconds * fs)

    for act in sorted(activity_map.keys()):
        if act not in segments or len(segments[act]) == 0:
            print(f"[WARN] activity {act} ({activity_map[act]}) not found in user{user_id:02d} exp{exp_id:02d}")
            continue

        for ex_i, (s, e) in enumerate(segments[act][:max_examples_per_activity]):
            seg_len = e - s
            if seg_len < 5:
                continue

            # 从区间中间取固定长度窗口
            mid = (s + e) // 2
            half = win_len // 2

            a = max(0, mid - half)
            b = min(len(X), mid + half)

            # 给一点边界余量（如果你希望看到过渡前后）
            a = max(0, a - pad)
            b = min(len(X), b + pad)

            Xseg = X[a:b]            # (L, 6)
            t = np.arange(len(Xseg)) / fs

            # 计算 Energy 和 Variance（对 6 通道整体）
            energy = np.sum(Xseg**2, axis=1)   # (L,)
            variance = np.var(Xseg, axis=1)    # (L,)

            # 创建 4 个坐标轴：
            # 1: acc_x/acc_y/acc_z
            # 2: gyro_x/gyro_y/gyro_z
            # 3: Energy
            # 4: Variance
            fig, axes = plt.subplots(
                nrows=4,
                ncols=1,
                sharex=True,
                figsize=(14, 9),
                gridspec_kw={"height_ratios": [1.2, 1.2, 0.9, 0.9]}
            )

            fig.suptitle(
                f"User {user_id:02d} | exp{exp_id:02d} | {activity_map[act]} (id={act}) | seg[{s}:{e}] | plot[{a}:{b}]",
                y=0.995
            )

            # 颜色约定
            color_x = "g"   # green
            color_y = "r"   # red
            color_z = "b"   # blue

            # 第 1 行：加速度 3 通道
            ax_acc = axes[0]
            ax_acc.plot(t, Xseg[:, 0], color=color_x, linewidth=0.8, label="acc_x")
            ax_acc.plot(t, Xseg[:, 1], color=color_y, linewidth=0.8, label="acc_y")
            ax_acc.plot(t, Xseg[:, 2], color=color_z, linewidth=0.8, label="acc_z")
            ax_acc.set_ylabel("acc")
            ax_acc.grid(True, linewidth=0.3, alpha=0.6)
            ax_acc.legend(loc="upper right", fontsize=8)

            # 第 2 行：陀螺仪 3 通道
            ax_gyro = axes[1]
            ax_gyro.plot(t, Xseg[:, 3], color=color_x, linewidth=0.8, label="gyro_x")
            ax_gyro.plot(t, Xseg[:, 4], color=color_y, linewidth=0.8, label="gyro_y")
            ax_gyro.plot(t, Xseg[:, 5], color=color_z, linewidth=0.8, label="gyro_z")
            ax_gyro.set_ylabel("gyro")
            ax_gyro.grid(True, linewidth=0.3, alpha=0.6)
            ax_gyro.legend(loc="upper right", fontsize=8)

            # 第 3 行：Energy
            ax_e = axes[2]
            ax_e.plot(t, energy, linewidth=0.8)
            ax_e.set_ylabel("Energy")
            ax_e.grid(True, linewidth=0.3, alpha=0.6)

            # 第 4 行：Variance
            ax_v = axes[3]
            ax_v.plot(t, variance, linewidth=0.8)
            ax_v.set_ylabel("Var")
            ax_v.set_xlabel("time (s)")
            ax_v.grid(True, linewidth=0.3, alpha=0.6)

            plt.tight_layout(rect=[0, 0, 1, 0.97])

            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                out = os.path.join(
                    save_dir,
                    f"user{user_id:02d}_exp{exp_id:02d}_act{act:02d}_{activity_map[act]}_{ex_i}.png"
                )
                plt.savefig(out, dpi=200, bbox_inches="tight")

            plt.show()




def compute_activity_duration_table(
    cfg,
    split="train",
    fs=50,
    save_path=None,
    activity_map=ACTIVITY_MAP_1_12,
):
    """
    统计指定 split (train/val/test) 中，各类 activity 的时长分布，
    只依赖 labels.txt（帧级标注），不需要跑 Dataset/窗口。

    时长单位：秒
    dur = (end_idx - start_idx) / fs

    输出：
      - 控制台打印一张表
      - 如果 save_path 不为 None，则保存为 CSV 文件
    """
    # 1. 根据 split 选 user 列表
    if split == "train":
        users = cfg.dataset.train_users
    elif split == "val":
        users = cfg.dataset.val_users
    elif split == "test":
        users = cfg.dataset.test_users
    else:
        raise ValueError(f"Unknown split: {split}")

    users = set(int(u) for u in users)

    # 2. 读取 labels.txt
    raw_dir = os.path.join(cfg.dataset.path)
    labels_path = os.path.join(raw_dir, "labels.txt")
    labels_all = np.loadtxt(labels_path)
    # 列: exp_id, user_id, activity_id, start_idx, end_idx

    # 3. 过滤到当前 split 的 user
    #    注意：labels_all[:,1] 是 user_id
    mask = np.isin(labels_all[:, 1].astype(int), list(users))
    labels_split = labels_all[mask]

    if labels_split.size == 0:
        print(f"[WARN] No labels found for split={split}")
        return

    # 4. 按 activity_id 收集所有区间的时长
    durations_by_act = {}  # act_id -> [durations...]
    for row in labels_split:
        _, user_id, act_id, start_idx, end_idx = row
        act_id = int(act_id)
        start_idx = int(start_idx)
        end_idx = int(end_idx)

        # 这里默认 end_idx 是 "exclusive"
        dur_sec = (end_idx - start_idx) / fs

        if dur_sec <= 0:
            # 异常区间直接跳过
            continue

        durations_by_act.setdefault(act_id, []).append(dur_sec)

    if len(durations_by_act) == 0:
        print(f"[WARN] After filtering, no valid segments for split={split}")
        return

    # 5. 计算统计量
    records = []
    for act_id in sorted(durations_by_act.keys()):
        durs = np.asarray(durations_by_act[act_id], dtype=np.float32)
        if durs.size == 0:
            continue

        record = {
            "act_id": act_id,
            "activity": activity_map.get(act_id, f"ACT_{act_id}"),
            "num_segments": int(durs.size),
            "total_sec": float(durs.sum()),
            "mean_sec": float(durs.mean()),
            "median_sec": float(np.median(durs)),
            "p10_sec": float(np.quantile(durs, 0.10)),
            "p90_sec": float(np.quantile(durs, 0.90)),
        }
        records.append(record)

    # 6. 控制台打印结果
    print(f"\n===== Activity duration stats (split={split}) =====")
    print("act_id, activity, num_segments, total_sec, mean_sec, median_sec, p10_sec, p90_sec")
    for r in records:
        print(
            f'{r["act_id"]:2d}, '
            f'{r["activity"]:>15s}, '
            f'{r["num_segments"]:4d}, '
            f'{r["total_sec"]:8.3f}, '
            f'{r["mean_sec"]:8.3f}, '
            f'{r["median_sec"]:8.3f}, '
            f'{r["p10_sec"]:8.3f}, '
            f'{r["p90_sec"]:8.3f}'
        )

    # 7. 如需保存 CSV 记录表
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            f.write("act_id,activity,num_segments,total_sec,mean_sec,median_sec,p10_sec,p90_sec\n")
            for r in records:
                f.write(
                    f'{r["act_id"]},'
                    f'{r["activity"]},'
                    f'{r["num_segments"]},'
                    f'{r["total_sec"]:.6f},'
                    f'{r["mean_sec"]:.6f},'
                    f'{r["median_sec"]:.6f},'
                    f'{r["p10_sec"]:.6f},'
                    f'{r["p90_sec"]:.6f}\n'
                )
        print(f"[INFO] Duration table saved to {save_path}")

    return records




def plot_activity_duration_figures(
    csv_path,
    save_dir,
):
    """
    从 activity_duration.csv 生成 4 张分析图，并直接保存为 PNG

    图包括：
    1) 各 activity 的 median_sec 柱状图
    2) 各 activity 的 median + p10~p90 误差棒
    3) 按 group(DYNAMIC/STATIC/TRANSITION) 聚合的 median_sec 柱状图
    4) num_segments vs median_sec 散点图（按 group）

    参数
    ----
    csv_path : str
        activity_duration.csv 路径
    save_dir : str
        图片保存目录
    """

    os.makedirs(save_dir, exist_ok=True)

    # -------------------------------------------------
    # 1. 读数据 + 分组
    # -------------------------------------------------
    df = pd.read_csv(csv_path)

    dynamic_set = {"WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"}
    static_set  = {"SITTING", "STANDING", "LAYING"}

    def map_group(act):
        if act in dynamic_set:
            return "DYNAMIC"
        if act in static_set:
            return "STATIC"
        return "TRANSITION"

    df["group"] = df["activity"].apply(map_group)

    # -------------------------------------------------
    # 图 1：median_sec per activity（柱状图）
    # -------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.bar(df["activity"], df["median_sec"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Median duration (sec)")
    plt.title("Median duration per activity")
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "fig1_median_per_activity.png"),
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()

    # -------------------------------------------------
    # 图 2：median + p10~p90 误差棒
    # -------------------------------------------------
    y = df["median_sec"].values
    yerr_lower = df["median_sec"] - df["p10_sec"]
    yerr_upper = df["p90_sec"] - df["median_sec"]

    plt.figure(figsize=(10, 5))
    plt.errorbar(
        df["activity"],
        y,
        yerr=[yerr_lower, yerr_upper],
        fmt="o",
        capsize=5
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Duration (sec)")
    plt.title("Median duration with p10–p90 range")
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "fig2_median_p10_p90.png"),
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()

    # -------------------------------------------------
    # 图 3：按 group 聚合的 median_sec
    # -------------------------------------------------
    group_stats = (
        df.groupby("group")["median_sec"]
          .mean()
          .reset_index()
    )

    plt.figure(figsize=(6, 5))
    plt.bar(group_stats["group"], group_stats["median_sec"])
    plt.ylabel("Average median duration (sec)")
    plt.title("Average median duration by group")
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "fig3_group_median.png"),
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()

    # -------------------------------------------------
    # 图 4：num_segments vs median_sec（散点）
    # -------------------------------------------------
    plt.figure(figsize=(7, 5))
    for g in df["group"].unique():
        sub = df[df["group"] == g]
        plt.scatter(
            sub["num_segments"],
            sub["median_sec"],
            label=g,
            s=60
        )

    plt.xlabel("Number of segments")
    plt.ylabel("Median duration (sec)")
    plt.title("Num segments vs median duration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "fig4_segments_vs_duration.png"),
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()

    print(f"[OK] Figures saved to: {save_dir}")





def plot_one_user_signal_0(
    cfg,
    user_id=1,
    exp_pick=0,
    max_len=None,
    fs=50.0,
    save_path=None
):
    """
    仿照 plot_one_user_raw_separate_axes 的风格，画：
      1) Raw acc + gyro
      2) Motion intensity: acc_norm, gyro_norm, jerk
      3) Posture: pitch, roll
      4) Label 序列

    每个都是单独一行坐标轴，sharex。
    """

    # -----------------------------
    # 1) 读 RawData: acc/gyro + labels
    # -----------------------------
    raw_dir = os.path.join(cfg.dataset.path)

    acc_files = sorted([
        f for f in os.listdir(raw_dir)
        if f.startswith("acc") and f"user{user_id:02d}" in f
    ])
    gyro_files = sorted([
        f for f in os.listdir(raw_dir)
        if f.startswith("gyro") and f"user{user_id:02d}" in f
    ])

    if len(acc_files) == 0:
        raise FileNotFoundError(f"No acc files for user {user_id} in {raw_dir}")
    if len(acc_files) != len(gyro_files):
        raise RuntimeError(f"acc/gyro count mismatch: {len(acc_files)} vs {len(gyro_files)}")

    acc_file = acc_files[exp_pick]
    gyro_file = gyro_files[exp_pick]

    acc = np.loadtxt(os.path.join(raw_dir, acc_file))   # (T,3)
    gyro = np.loadtxt(os.path.join(raw_dir, gyro_file)) # (T,3)

    if acc.shape[0] != gyro.shape[0]:
        raise RuntimeError(f"T mismatch acc({acc.shape[0]}) vs gyro({gyro.shape[0]})")

    X_raw = np.hstack([acc[:, :3], gyro[:, :3]])  # (T,6)

    # 读 labels.txt
    labels_all = np.loadtxt(os.path.join(raw_dir, "labels.txt"))
    labels_user = labels_all[labels_all[:, 1] == user_id]

    # 从 acc_file 里解析 exp_id，例如 acc_exp01_user01.txt
    exp_id = int(acc_file.split("_")[1][3:])  # "expXX" → XX
    labels_exp = labels_user[labels_user[:, 0] == exp_id]

    y = np.zeros(X_raw.shape[0], dtype=int)
    for _, _, act, s, e in labels_exp:
        s = int(s)
        e = int(e)
        y[s:e] = int(act)   # 如果 end 是 inclusive，改成 s:e+1

    # 截断展示长度
    if max_len is not None:
        X_raw = X_raw[:max_len]
        y = y[:max_len]

    T = X_raw.shape[0]
    t = np.arange(T) / fs   # 换成秒，更直观

    # -----------------------------
    # 2) 计算物理特征：acc_norm / gyro_norm / jerk / pitch / roll
    # -----------------------------
    acc_xyz = X_raw[:, 0:3]
    gyro_xyz = X_raw[:, 3:6]

    # 幅值
    acc_norm = np.linalg.norm(acc_xyz, axis=1)          # (T,)
    gyro_norm = np.linalg.norm(gyro_xyz, axis=1)        # (T,)

    # jerk（对 acc_norm 做差分）
    jerk = np.zeros_like(acc_norm)
    jerk[1:] = np.abs(acc_norm[1:] - acc_norm[:-1]) * fs

    # pitch / roll（和你 Dataset 里的一致）
    ax = acc_xyz[:, 0]
    ay = acc_xyz[:, 1]
    az = acc_xyz[:, 2]
    eps = 1e-8

    pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2) + eps)
    roll  = np.arctan2( ay, az + eps)

    # -----------------------------
    # 3) 画图：4 行 1 列 + label
    # -----------------------------
    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        sharex=True,
        figsize=(14, 9),
        gridspec_kw={"height_ratios": [2.5, 1.8, 1.5, 0.8]}
    )

    fig.suptitle(
        f"User {user_id:02d} | {acc_file} + {gyro_file}",
        y=0.995
    )

    # (1) Raw acc + gyro
    ax0 = axes[0]
    ax0.plot(t, X_raw[:, 0], label="acc_x")
    ax0.plot(t, X_raw[:, 1], label="acc_y")
    ax0.plot(t, X_raw[:, 2], label="acc_z")
    ax0.plot(t, X_raw[:, 3], "--", label="gyro_x")
    ax0.plot(t, X_raw[:, 4], "--", label="gyro_y")
    ax0.plot(t, X_raw[:, 5], "--", label="gyro_z")
    ax0.set_ylabel("raw")
    ax0.set_title("Raw Accelerometer & Gyroscope")
    ax0.legend(fontsize=8, ncol=3)
    ax0.grid(True, linewidth=0.3, alpha=0.6)

    # (2) Motion intensity
    ax1 = axes[1]
    ax1.plot(t, acc_norm, label="acc_norm")
    ax1.plot(t, gyro_norm, label="gyro_norm")
    ax1.plot(t, jerk, label="jerk")
    ax1.set_ylabel("intensity")
    ax1.set_title("Motion Intensity")
    ax1.legend(fontsize=8)
    ax1.grid(True, linewidth=0.3, alpha=0.6)

    # (3) Posture
    ax2 = axes[2]
    ax2.plot(t, pitch, label="pitch")
    ax2.plot(t, roll, label="roll")
    ax2.set_ylabel("angle (rad)")
    ax2.set_title("Posture (pitch / roll)")
    ax2.legend(fontsize=8)
    ax2.grid(True, linewidth=0.3, alpha=0.6)

    # (4) Label 序列
    ax3 = axes[3]
    ax3.step(t, y, where="post", linewidth=1.0)
    ax3.set_ylabel("label")
    ax3.set_xlabel("time (s)")
    ax3.grid(True, linewidth=0.3, alpha=0.6)

    plt.tight_layout()

    # 保存
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    # 打印标签统计
    uniq, cnt = np.unique(y, return_counts=True)
    print("Label counts:", dict(zip(uniq.tolist(), cnt.tolist())))

def label_to_segments(t, y):
    """
    把逐帧 label 转成 [(start_t, end_t, label), ...]
    """
    segments = []
    if len(y) == 0:
        return segments

    cur_label = y[0]
    start = 0

    for i in range(1, len(y)):
        if y[i] != cur_label:
            segments.append((t[start], t[i], cur_label))
            cur_label = y[i]
            start = i

    segments.append((t[start], t[-1], cur_label))
    return segments

def get_activity_colors_0():
    """
    给每个 activity 一个固定背景色
    """
    cmap = plt.get_cmap("tab20")
    colors = {}
    for k in range(13):   # 0~12
        colors[k] = cmap(k % 20)
    return colors

def get_activity_colors_00():
    """
    给每个 activity 一个固定背景色
    1–3   : 蓝色系（Walking）
    4–6   : 绿色系（Posture）
    7–12  : 红色系（Transition）
    """
    colors = {}

    cmap_blue  = plt.get_cmap("Blues")
    cmap_green = plt.get_cmap("Greens")
    cmap_red   = plt.get_cmap("Reds")

    # 1–3: WALKING 类（蓝色系）
    blue_ids = [1, 2, 3]
    for i, k in enumerate(blue_ids):
        colors[k] = cmap_blue(0.3 + 0.4 * i / (len(blue_ids) - 1))

    # 4–6: POSTURE 类（绿色系）
    green_ids = [4, 5, 6]
    for i, k in enumerate(green_ids):
        colors[k] = cmap_green(0.3 + 0.4 * i / (len(green_ids) - 1))

    # 7–12: TRANSITION 类（红色系）
    red_ids = [7, 8, 9, 10, 11, 12]
    for i, k in enumerate(red_ids):
        colors[k] = cmap_red(0.3 + 0.5 * i / (len(red_ids) - 1))

    # 可选：0 类（无效 / padding），给一个透明或浅灰
    colors[0] = (0.9, 0.9, 0.9, 1.0)

    return colors

def get_activity_colors():
    """
    给每个 activity 一个固定背景色（高对比度版本）
    1–3   : 蓝色系（Walking）
    4–6   : 绿色/青色系（Posture）
    7–12  : 红/橙系（Transition）
    0     : 浅灰，用于“无标签”
    """
    colors = {}

    # 0: 无活动 / padding
    colors[0] = "#f0f0f0"  # 很浅的灰色

    # 1–3: WALKING 系（蓝色系，但彼此明显不同）
    colors[1] = "#fb9a99"  # 
    colors[2] = "#ff7f0e"# 
    colors[3] = "#d62728"  # 

    # 4–6: POSTURE 系（绿 / 青色系）
    colors[4] = "#2ca02c"  # 亮绿
    colors[5] = "#66c2a5"# 偏青的浅绿
    colors[6] = "#006d2c"  # 深绿

    # 7–12: TRANSITION 系（红 / 橙色系，暖色一族）
    colors[7]  = "#f9e79f"
    colors[8]  = "#f1c40f" 
    colors[9]  =  "#d6b3e8"#  
    colors[10] =  "#7a1fa2"# 
    colors[11] = "#f7b6d2"
    colors[12] = "#c2185b" 

    return colors

def build_activity_legend_elements(colors, activity_map, alpha=0.15):
    """
    根据 activity 的颜色和名字，构造 legend 用的 Patch
    """
    legend_elements = []

    for k in sorted(activity_map.keys()):
        legend_elements.append(
            Patch(
                facecolor=colors[k],
                edgecolor="none",
                alpha=alpha,
                label=activity_map[k]
            )
        )
    return legend_elements


def plot_one_user_signal(
    cfg,
    user_id=1,
    exp_pick=0,
    max_len=None,
    fs=50.0,
    save_path=None
):
    """
    画 6 行：
      1) acc_x / acc_y / acc_z
      2) gyro_x / gyro_y / gyro_z
      3) acc_norm + gyro_norm
      4) jerk
      5) pitch + roll
      6) label

    大标题分块：
      - 行 1: "Raw Sensor Signals"
      - 行 3: "Motion Intensity"
      - 行 5: "Posture"
    """

    # -----------------------------
    # 1) 读 RawData: acc/gyro + labels
    # -----------------------------
    raw_dir = os.path.join(cfg.dataset.path)

    acc_files = sorted([
        f for f in os.listdir(raw_dir)
        if f.startswith("acc") and f"user{user_id:02d}" in f
    ])
    gyro_files = sorted([
        f for f in os.listdir(raw_dir)
        if f.startswith("gyro") and f"user{user_id:02d}" in f
    ])

    if len(acc_files) == 0:
        raise FileNotFoundError(f"No acc files for user {user_id} in {raw_dir}")
    if len(acc_files) != len(gyro_files):
        raise RuntimeError(f"acc/gyro count mismatch: {len(acc_files)} vs {len(gyro_files)}")

    acc_file = acc_files[exp_pick]
    gyro_file = gyro_files[exp_pick]

    acc = np.loadtxt(os.path.join(raw_dir, acc_file))   # (T,3)
    gyro = np.loadtxt(os.path.join(raw_dir, gyro_file)) # (T,3)

    if acc.shape[0] != gyro.shape[0]:
        raise RuntimeError(f"T mismatch acc({acc.shape[0]}) vs gyro({gyro.shape[0]})")

    X_raw = np.hstack([acc[:, :3], gyro[:, :3]])  # (T,6)

    # 读 labels.txt
    labels_all = np.loadtxt(os.path.join(raw_dir, "labels.txt"))
    labels_user = labels_all[labels_all[:, 1] == user_id]

    # 从 acc_file 里解析 exp_id，例如 acc_exp01_user01.txt
    exp_id = int(acc_file.split("_")[1][3:])  # "expXX" → XX
    labels_exp = labels_user[labels_user[:, 0] == exp_id]

    y = np.zeros(X_raw.shape[0], dtype=int)
    for _, _, act, s, e in labels_exp:
        s = int(s)
        e = int(e)
        y[s:e] = int(act)   # 如果 end 是 inclusive，改成 s:e+1

    # 截断展示长度
    if max_len is not None:
        X_raw = X_raw[:max_len]
        y = y[:max_len]

    T = X_raw.shape[0]
    t = np.arange(T) / fs   # 换成秒

    # -----------------------------
    # 2) 计算物理特征：acc_norm / gyro_norm / jerk / pitch / roll
    # -----------------------------
    acc_xyz = X_raw[:, 0:3]
    gyro_xyz = X_raw[:, 3:6]

    # 幅值
    acc_norm = np.linalg.norm(acc_xyz, axis=1)          # (T,)
    gyro_norm = np.linalg.norm(gyro_xyz, axis=1)        # (T,)

    # jerk（对 acc_norm 做差分）
    jerk = np.zeros_like(acc_norm)
    jerk[1:] = np.abs(acc_norm[1:] - acc_norm[:-1]) * fs

    # pitch / roll（和 Dataset 里一致）
    ax = acc_xyz[:, 0]
    ay = acc_xyz[:, 1]
    az = acc_xyz[:, 2]
    eps = 1e-8

    pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2) + eps)
    roll  = np.arctan2( ay, az + eps)

    # -----------------------------
    # 3) 画图：6 行 1 列
    # -----------------------------
    fig, axes = plt.subplots(
        nrows=6,
        ncols=1,
        sharex=True,
        figsize=(14, 12),
        gridspec_kw={"height_ratios": [1.4, 1.4, 1.1, 0.9, 1.1, 0.7]}
    )

    fig.suptitle(
        f"User {user_id:02d} | {acc_file} + {gyro_file}",
        y=0.995
    )

    # 固定颜色：x=绿, y=红, z=蓝（在 acc/gyro 中保持一致）
    color_x = "g"
    color_y = "r"
    color_z = "b"

    # (1) Acc: x/y/z
    ax0 = axes[0]
    ax0.plot(t, X_raw[:, 0], color=color_x, linewidth=0.8, label="acc_x")
    ax0.plot(t, X_raw[:, 1], color=color_y, linewidth=0.8, label="acc_y")
    ax0.plot(t, X_raw[:, 2], color=color_z, linewidth=0.8, label="acc_z")
    ax0.set_ylabel("acc")
    ax0.set_title("Raw Sensor Signals")
    ax0.legend(fontsize=8, ncol=3)
    ax0.grid(True, linewidth=0.3, alpha=0.6)

    # (2) Gyro: x/y/z（沿用同一套颜色）
    ax1 = axes[1]
    ax1.plot(t, X_raw[:, 3], color=color_x, linewidth=0.8, label="gyro_x")
    ax1.plot(t, X_raw[:, 4], color=color_y, linewidth=0.8, label="gyro_y")
    ax1.plot(t, X_raw[:, 5], color=color_z, linewidth=0.8, label="gyro_z")
    ax1.set_ylabel("gyro")
    ax1.legend(fontsize=8, ncol=3)
    ax1.grid(True, linewidth=0.3, alpha=0.6)

    # (3) Motion intensity: acc_norm + gyro_norm
    ax2 = axes[2]
    ax2.plot(t, acc_norm, linewidth=0.8, label="acc_norm")
    ax2.plot(t, gyro_norm, linewidth=0.8, label="gyro_norm")
    ax2.set_ylabel("intensity")
    ax2.set_title("Motion Intensity")
    ax2.legend(fontsize=8)
    ax2.grid(True, linewidth=0.3, alpha=0.6)

    # (4) Jerk 单独一行
    ax3 = axes[3]
    ax3.plot(t, jerk,color="g",linewidth=0.8, label="jerk")
    ax3.set_ylabel("jerk")
    ax3.legend(fontsize=8)
    ax3.grid(True, linewidth=0.3, alpha=0.6)

    color_pitch = "#1BA1E2"
    color_roll = "#EA6B66"


    # (5) Posture: pitch + roll
    ax4 = axes[4]
    ax4.plot(t, pitch, color=color_pitch, linewidth=0.8, label="pitch")
    ax4.plot(t, roll, color=color_roll, linewidth=0.8, label="roll")
    ax4.set_ylabel("angle (rad)")
    ax4.set_title("Posture")
    ax4.legend(fontsize=8)
    ax4.grid(True, linewidth=0.3, alpha=0.6)

    # (6) Label 序列
    ax5 = axes[5]
    ax5.step(t, y, where="post", linewidth=1.0)
    ax5.set_ylabel("label")
    ax5.set_xlabel("time (s)")
    ax5.grid(True, linewidth=0.3, alpha=0.6)

    plt.tight_layout()

    # 保存
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    # 打印标签统计
    uniq, cnt = np.unique(y, return_counts=True)
    print("Label counts:", dict(zip(uniq.tolist(), cnt.tolist())))

def plot_one_user_signal_with_label_background(
    cfg,
    user_id=1,
    exp_pick=0,
    max_len=None,
    fs=50.0,
    save_path=None
):
    """
    6 行 → 5 行（去掉 label 子图）
    用 label 给所有子图涂背景色
    """

    # -----------------------------
    # 1) 读 RawData
    # -----------------------------
    raw_dir = os.path.join(cfg.dataset.path)

    acc_files = sorted([f for f in os.listdir(raw_dir)
                        if f.startswith("acc") and f"user{user_id:02d}" in f])
    gyro_files = sorted([f for f in os.listdir(raw_dir)
                         if f.startswith("gyro") and f"user{user_id:02d}" in f])

    acc_file = acc_files[exp_pick]
    gyro_file = gyro_files[exp_pick]

    acc = np.loadtxt(os.path.join(raw_dir, acc_file))
    gyro = np.loadtxt(os.path.join(raw_dir, gyro_file))
    X_raw = np.hstack([acc[:, :3], gyro[:, :3]])

    # labels
    labels_all = np.loadtxt(os.path.join(raw_dir, "labels.txt"))
    labels_user = labels_all[labels_all[:, 1] == user_id]
    exp_id = int(acc_file.split("_")[1][3:])
    labels_exp = labels_user[labels_user[:, 0] == exp_id]

    y = np.zeros(X_raw.shape[0], dtype=int)
    for _, _, act, s, e in labels_exp:
        y[int(s):int(e)] = int(act)

    if max_len is not None:
        X_raw = X_raw[:max_len]
        y = y[:max_len]

    T = X_raw.shape[0]
    t = np.arange(T) / fs

    # -----------------------------
    # 2) 物理特征
    # -----------------------------
    acc_xyz = X_raw[:, 0:3]
    gyro_xyz = X_raw[:, 3:6]

    acc_norm = np.linalg.norm(acc_xyz, axis=1)
    gyro_norm = np.linalg.norm(gyro_xyz, axis=1)

    jerk = np.zeros_like(acc_norm)
    jerk[1:] = np.abs(acc_norm[1:] - acc_norm[:-1]) * fs

    ax, ay, az = acc_xyz.T
    eps = 1e-8
    pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2) + eps)
    roll  = np.arctan2( ay, az + eps)

    # -----------------------------
    # 3) label → 时间区间
    # -----------------------------
    segments = label_to_segments(t, y)
    label_colors = get_activity_colors()

    # --- NEW: 色谱条绘制函数（只解释颜色-标签映射）
    from matplotlib.patches import Rectangle

    def draw_activity_color_strip(ax_strip, colors, activity_map):
        keys = sorted(activity_map.keys())
        n = len(keys)
        for i, k in enumerate(keys):
            # 色块
            ax_strip.add_patch(
                Rectangle(
                    (i, 0), 1, 1,
                    facecolor=colors[k],
                    edgecolor="none"
                )
            )
            # 文字（斜着放）
            ax_strip.text(
                i + 0.5, -0.15,
                activity_map[k],
                ha="right",
                va="top",
                rotation=55,
                fontsize=8
            )

        ax_strip.set_xlim(0, n)
        ax_strip.set_ylim(-0.8, 1)
        ax_strip.axis("off")

    def draw_activity_color_strip(ax_strip, colors, activity_map, alpha=0.15):  # --- CHANGED: 增加 alpha 参数，默认 0.15
        keys = sorted(activity_map.keys())
        n = len(keys)
        for i, k in enumerate(keys):
            # 色块
            ax_strip.add_patch(
                Rectangle(
                    (i, 0), 1, 1,
                    facecolor=colors[k],
                    edgecolor="none",
                    alpha=alpha           # --- CHANGED: 使用 0.15 不透明度
                )
            )
            # 文字（斜着放）
            ax_strip.text(
                i + 0.5, -0.18,         # --- CHANGED: 稍微往下放一点，适配更扁的高度
                activity_map[k],
                ha="center",
                va="top",
                #rotation=55,
                fontsize=8
            )

        ax_strip.set_xlim(0, n)
        ax_strip.set_ylim(-0.9, 1)
        ax_strip.axis("off")

    # -----------------------------
    # 4) 画图：5 行
    # -----------------------------
    fig, axes = plt.subplots(
        nrows=5, ncols=1, sharex=True,
        figsize=(14, 11),
        gridspec_kw={"height_ratios": [1.4, 1.4, 1.1, 0.9, 1.1]}
    )

    #fig.suptitle(f"User {user_id:02d} | {acc_file} ", y=0.995)

    # 统一涂背景函数
    def paint_background(ax_plot):
        for t0, t1, lab in segments:
            if lab == 0:
                continue
            ax_plot.axvspan(
                t0, t1,
                color=label_colors[lab],
                alpha=0.12,
                linewidth=0
            )

    # 固定颜色
    cx, cy, cz = "g", "r", "b"

    # (1) acc
    ax0 = axes[0]
    paint_background(ax0)
    ax0.plot(t, acc_xyz[:, 0], cx, lw=0.8, label="acc_x")
    ax0.plot(t, acc_xyz[:, 1], cy, lw=0.8, label="acc_y")
    ax0.plot(t, acc_xyz[:, 2], cz, lw=0.8, label="acc_z")
    ax0.set_title("Raw Sensor Signals")
    ax0.legend(
        fontsize=8,
        ncol=3,
        loc="upper left"      # 统一放左上
    )
    ax0.grid(alpha=0.3)

    # (2) gyro
    ax1 = axes[1]
    paint_background(ax1)
    ax1.plot(t, gyro_xyz[:, 0], cx, lw=0.8, label="gyro_x")
    ax1.plot(t, gyro_xyz[:, 1], cy, lw=0.8, label="gyro_y")
    ax1.plot(t, gyro_xyz[:, 2], cz, lw=0.8, label="gyro_z")
    ax1.legend(
        fontsize=8,
        ncol=3,
        loc="upper left"
    )
    ax1.grid(alpha=0.3)

    # (3) intensity
    ax2 = axes[2]
    paint_background(ax2)
    ax2.plot(t, acc_norm, lw=0.8, label="acc_norm")
    ax2.plot(t, gyro_norm, lw=0.8, label="gyro_norm")
    ax2.set_title("Motion Intensity")
    ax2.legend(
        fontsize=8,
        loc="upper left"
    )
    ax2.grid(alpha=0.3)

    # (4) jerk
    ax3 = axes[3]
    paint_background(ax3)
    ax3.plot(t, jerk, '#60A917', lw=0.8, label="jerk")
    ax3.legend(
        fontsize=8,
        loc="upper left"
    )
    ax3.grid(alpha=0.3)

    color_pitch = "#1BA1E2"
    color_roll = "#EA6B66"

    # (5) posture
    ax4 = axes[4]
    paint_background(ax4)
    ax4.plot(t, pitch, color_pitch, lw=0.8, label="pitch")
    ax4.plot(t, roll,  color_roll,  lw=0.8, label="roll")
    ax4.set_title("Posture")
    ax4.set_xlabel("time (s)")
    ax4.legend(
        fontsize=8,
        loc="upper left"
    )
    ax4.grid(alpha=0.3)

     # --- CHANGED: 给底部色谱条留出空间（略缩小一点，腾出 0–0.12）
    plt.tight_layout(rect=[0, 0.12, 1, 0.98])

    # --- CHANGED: 使用 subplotpars 对齐主图左右，并把色谱条高度压扁
    left = fig.subplotpars.left
    right = fig.subplotpars.right
    ax_strip = fig.add_axes([
        left,          # left：与上方子图对齐
        0.09,          # bottom：靠近底部，可微调
        right - left,  # width：与上方子图等宽
        0.035           # height：更扁的色谱条
    ])
    draw_activity_color_strip(ax_strip, label_colors, ACTIVITY_MAP_1_12, alpha=0.15)

    for ax in axes:
        ax.grid(False)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_examples_by_activity_12_signal(
    cfg,
    user_id=1,
    exp_pick=0,
    seconds=4,
    fs=50,
    max_examples_per_activity=1,
    save_dir=None,
    activity_map=ACTIVITY_MAP_1_12,
    pad_seconds=0.5,
):
    """
    从 labels.txt 区间中，为每个 activity(1..12) 画样例片段（最可信）

    改造后版本：
      1 行：acc_x / acc_y / acc_z
      2 行：gyro_x / gyro_y / gyro_z
      3 行：acc_norm / gyro_norm / jerk
      4 行：pitch / roll

    - seconds: 每个样例展示时长（秒）
    - fs: 采样率（HAPT 常见 50Hz）
    - max_examples_per_activity: 每个动作画几个区间样例
    - pad_seconds: 在区间中间取片段，两侧留一点余量
    """
    # ---- 1) 读取一个 experiment 的原始 6 通道 ----
    X, exp_id, acc_file, gyro_file = _load_one_exp_raw(cfg, user_id, exp_pick)
    # X: [T,6] = acc(3) + gyro(3)

    raw_dir = os.path.join(cfg.dataset.path)

    # ---- 2) 读取标签，并筛出该 user + exp ----
    labels_all = np.loadtxt(os.path.join(raw_dir, "labels.txt"))
    labels_user = labels_all[labels_all[:, 1] == user_id]
    labels_exp = labels_user[labels_user[:, 0] == exp_id]

    # 按 activity_id 分组区间
    segments = {}
    for _, _, act, s, e in labels_exp:
        act = int(act)
        s = int(s)
        e = int(e)
        segments.setdefault(act, []).append((s, e))

    win_len = int(seconds * fs)
    pad = int(pad_seconds * fs)

    # ---- 3) 遍历每个动作，画样例 ----
    for act in sorted(activity_map.keys()):
        if act not in segments or len(segments[act]) == 0:
            print(f"[WARN] activity {act} ({activity_map[act]}) not found in user{user_id:02d} exp{exp_id:02d}")
            continue

        for ex_i, (s, e) in enumerate(segments[act][:max_examples_per_activity]):
            seg_len = e - s
            if seg_len < 5:
                continue

            # 从区间中间取固定长度窗口
            mid = (s + e) // 2
            half = win_len // 2

            a = max(0, mid - half)
            b = min(len(X), mid + half)

            # 给一点边界余量
            a = max(0, a - pad)
            b = min(len(X), b + pad)

            Xseg = X[a:b]               # [L, 6]
            L = Xseg.shape[0]
            t = np.arange(L) / fs       # 用秒作为横轴

            # -----------------------------
            # 3a) 计算新的物理特征
            # -----------------------------
            acc_xyz = Xseg[:, 0:3]
            gyro_xyz = Xseg[:, 3:6]

            # 幅值
            acc_norm = np.linalg.norm(acc_xyz, axis=1)   # [L,]
            gyro_norm = np.linalg.norm(gyro_xyz, axis=1) # [L,]

            # jerk（acc_norm 的差分）
            jerk = np.zeros_like(acc_norm)
            jerk[1:] = np.abs(acc_norm[1:] - acc_norm[:-1]) * fs

            # pitch / roll（和 Dataset 保持一致）
            ax = acc_xyz[:, 0]
            ay = acc_xyz[:, 1]
            az = acc_xyz[:, 2]
            eps = 1e-8

            pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2) + eps)
            roll  = np.arctan2( ay, az + eps)

            # -----------------------------
            # 3b) 画 4 行子图：acc / gyro / intensity / posture
            # -----------------------------
            fig, axes = plt.subplots(
                nrows=4,
                ncols=1,
                sharex=True,
                figsize=(14, 9),
                gridspec_kw={"height_ratios": [1.4, 1.4, 1.0, 1.0]}
            )

            fig.suptitle(
                f"User {user_id:02d} | exp{exp_id:02d} | "
                f"{activity_map[act]} (id={act}) | seg[{s}:{e}] | plot[{a}:{b}]",
                y=0.995
            )

            # 颜色约定
            color_x = "g"
            color_y = "r"
            color_z = "b"

            # 行 1：加速度 3 通道
            ax_acc = axes[0]
            ax_acc.plot(t, Xseg[:, 0], color=color_x, linewidth=0.8, label="acc_x")
            ax_acc.plot(t, Xseg[:, 1], color=color_y, linewidth=0.8, label="acc_y")
            ax_acc.plot(t, Xseg[:, 2], color=color_z, linewidth=0.8, label="acc_z")
            ax_acc.set_ylabel("acc")
            ax_acc.grid(True, linewidth=0.3, alpha=0.6)
            ax_acc.legend(loc="upper right", fontsize=8)

            # 行 2：陀螺仪 3 通道
            ax_gyro = axes[1]
            ax_gyro.plot(t, Xseg[:, 3], color=color_x, linewidth=0.8, label="gyro_x")
            ax_gyro.plot(t, Xseg[:, 4], color=color_y, linewidth=0.8, label="gyro_y")
            ax_gyro.plot(t, Xseg[:, 5], color=color_z, linewidth=0.8, label="gyro_z")
            ax_gyro.set_ylabel("gyro")
            ax_gyro.grid(True, linewidth=0.3, alpha=0.6)
            ax_gyro.legend(loc="upper right", fontsize=8)

            # 行 3：Motion intensity
            ax_int = axes[2]
            ax_int.plot(t, acc_norm, linewidth=0.8, label="acc_norm")
            ax_int.plot(t, gyro_norm, linewidth=0.8, label="gyro_norm")
            ax_int.plot(t, jerk, linewidth=0.8, label="jerk")
            ax_int.set_ylabel("intensity")
            ax_int.grid(True, linewidth=0.3, alpha=0.6)
            ax_int.legend(loc="upper right", fontsize=8)

            color_pitch = "#1BA1E2"
            color_roll = "#EA6B66"

            # 行 4：Posture
            ax_pos = axes[3]
            ax_pos.plot(t, pitch,color=color_pitch, linewidth=0.8, label="pitch")
            ax_pos.plot(t, roll, color=color_roll, linewidth=0.8, label="roll")
            ax_pos.set_ylabel("angle (rad)")
            ax_pos.set_xlabel("time (s)")
            ax_pos.grid(True, linewidth=0.3, alpha=0.6)
            ax_pos.legend(loc="upper right", fontsize=8)

            plt.tight_layout(rect=[0, 0, 1, 0.97])

            # 保存
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                out = os.path.join(
                    save_dir,
                    f"user{user_id:02d}_exp{exp_id:02d}_act{act:02d}_{activity_map[act]}_{ex_i}.png"
                )
                plt.savefig(out, dpi=200, bbox_inches="tight")

            plt.show()