import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch


class HAPT12RawWindows(Dataset):
    def __init__(self, cfg, split: str, augment=False):
        super().__init__()
        self.cfg = cfg.dataset
        self.split = split

        self.augment = augment

        # -----------------------------------------------
        # 0) Optionally save externally provided mean/std
        # -----------------------------------------------
        self.mean = self.cfg.mean
        self.std  = self.cfg.std
        self.feature_mean = None
        self.feature_std  = None

        # === MOD 1: Cache full continuous sequences (for transition matrix statistics, train split only) ===
        # Each element is a dict: {"user_id": int, "X": np.ndarray[T, C], "y": np.ndarray[T]}
        self.full_sequences = []

        if hasattr(cfg.model, "in_channels"):
            self.in_channels = int(cfg.model.in_channels)

        elif hasattr(self.cfg.model, "in_ch"):
            self.in_channels = int(cfg.model.in_ch)

        elif hasattr(self.cfg.model, "input_dim"):
            self.in_channels = int(cfg.model.input_dim)

        else:
            raise RuntimeError(
                "in_channels not found. "
                "Please set dataset.in_channels or model.in_channels in config."
            )

        
        if self.in_channels not in (6, 9, 11):
            raise ValueError(
                f"Unsupported in_channels={self.in_channels}. "
                "Expected one of (6, 9, 11)."
            )

        self.raw_dir = self.cfg.path

        # users split
        if split == "train":
            self.users = self.cfg.train_users
        elif split == "val":
            self.users = self.cfg.val_users
        elif split == "test":
            self.users = self.cfg.test_users
        else:
            raise ValueError(f"Invalid split: {split}")

        # --------------------------------------------------------------------------
        # 2) Ensure mean/std are ready (computed only during the first training run)
        # --------------------------------------------------------------------------
        self.setup_normalization(self.cfg)

        # --------------------------------------------------------------------------
        # 3) Build continuous data for the current split and apply normalization
        # --------------------------------------------------------------------------
        # === MOD 2: Also store full_sequences for the train split (used for transition matrix statistics) ===
        store_full = (self.split == "train")
        X_cont, y_cont = self._build_split_users_continuous(self.users, store_full=store_full)
        self.X_cont = X_cont
        self.y_cont = y_cont

        # sliding window params
        sw = self.cfg.sliding_window
        self.ws = int(sw.window_size)
        self.shift = int(sw.window_shift)
        self.label_mode = getattr(sw, "label_mode", "center")  # "center" 或 "majority"

        self.X, self.y = self._build_windows_from_continuous(self.X_cont, self.y_cont)

    def _load_user_continuous(self, user_id: int, stats: "RunningStats | None" = None):
        """
        Load all experiments of a specific user and concatenate them into continuous signals X_user and y_user.

        Returns:
            X_user: np.ndarray, shape [T_total, C]
            y_user: np.ndarray, shape [T_total], 0 (unlabeled) or 1..12
        """
        labels_df = self.load_hapt_labels()

        # Find all experiments for this user (preserve original order)
        user_segs = labels_df[labels_df["user_id"] == user_id]
        exp_ids = user_segs["exp_id"].unique().tolist()

        X_all = []
        y_all = []

        for exp_id in exp_ids:
            # 1. Read RawData → [T,6]
            X_exp = self.load_exp_raw_6ch(exp_id, user_id)  # [T,6]

            # 1.5 kombine physical features → [T, 6+K]
            X_exp = self.add_physical_features(X_exp, fs=50.0)

            T = X_exp.shape[0]

            # 2. Initialize frame-level labels (0 = unlabeled)
            y_exp = np.zeros(T, dtype=np.int64)

            # 3. Fill the labeled segments of the experiment
            segs = user_segs[user_segs["exp_id"] == exp_id]

            for _, row in segs.iterrows():
                act_id = int(row["act_id"])   # 1..12
                start  = int(row["start"])    # 1-based inclusive
                end    = int(row["end"])      # 1-based inclusive

                s = start - 1
                e = end                       # python slice [s:e)
                y_exp[s:e] = act_id

            
            if stats is not None:
                stats.update(X_exp)  # X_exp: [T, C_total]

            X_all.append(X_exp)
            y_all.append(y_exp)

        # 4. Concatenate experiments into a continuous signal
        X_user = np.concatenate(X_all, axis=0)   # [T_total, C]
        y_user = np.concatenate(y_all, axis=0)   # [T_total]

        return X_user, y_user

    def _build_train_users_continuous(self, cfg):
        stats = RunningStats()
        X_users = []
        y_users = []

        for user_id in self.cfg.train_users:
            user_id = int(user_id)
            
            X_user, y_user = self._load_user_continuous(user_id, stats=stats)
            X_users.append(X_user)
            y_users.append(y_user)

        # Obtain continuous data for all train users (optional)
        X_train_cont = np.concatenate(X_users, axis=0)
        y_train_cont = np.concatenate(y_users, axis=0)

        # Obtain mean/std in one pass, already including physical features
        mean, std = stats.finalize()

        # Save for later use in dataset normalization
        self.feature_mean = mean
        self.feature_std  = std
        print(f"mean: {self.feature_mean}")
        print(f"std: {self.feature_std}")
        print("-----------------------------------------\n")

        return X_train_cont, y_train_cont

    def _normalize(self, X):
        # X: [T, C]
        return (X - self.feature_mean) / (self.feature_std + 1e-8)

    # === MOD 3: Add store_full flag to _build_split_users_continuous and write to self.full_sequences ===
    def _build_split_users_continuous(self, user_ids, store_full: bool = False):
        """
        Generic: build continuous data for a given set of user_ids and apply normalization.
        This function no longer receives stats, nor does it modify mean/std.

        If store_full=True, the continuous sequence of each user
        is saved to self.full_sequences for later use when computing the transition matrix.
        """
        X_users = []
        y_users = []

        for user_id in user_ids:
            user_id = int(user_id)
            X_user, y_user = self._load_user_continuous(user_id, stats=None)


            X_user = self._normalize(X_user)   # [T, C]

            X_users.append(X_user)
            y_users.append(y_user)

            if store_full:
                self.full_sequences.append(
                    {
                        "user_id": user_id,
                        "X": X_user,   # [T, C]
                        "y": y_user,   # [T]
                    }
                )

        X_all = np.concatenate(X_users, axis=0)
        y_all = np.concatenate(y_users, axis=0)
        return X_all, y_all

    def setup_normalization(self, cfg):
        """
        Normalization parameter preparation logic:
        - If mean/std already exist, they are never recomputed.
        - They are computed exactly once on the train split only when completely absent.
        """

        # 2) Already present in cfg (indicating they were computed in
        if getattr(self.cfg, "mean", None) is not None and getattr(self.cfg, "std", None) is not None:
            self.feature_mean = np.array(self.cfg.mean, dtype=np.float32)
            self.feature_std  = np.array(self.cfg.std,  dtype=np.float32)
            return

        # 3) At this point, mean/std do not exist anywhere and must be computed once.
        if self.split != "train":
            raise RuntimeError(
                "mean/std not available yet. "
                "You must build the train dataset first."
            )

        print("Computing train mean/std from continuous signals...")
        self._build_train_users_continuous(cfg)

        # Inside _build_train_users_continuous, self.feature_mean/std are already set.
        self.cfg.mean = self.feature_mean.tolist()
        self.cfg.std  = self.feature_std.tolist()

    def _build_windows_from_continuous(self, X_cont, y_cont):
        """
        Build sliding windows from continuous X_cont and y_cont.

        X_cont: [T_total, C]  already normalized
        y_cont: [T_total]     0 = unlabeled, 1..12 = valid classes

        The labeling strategy is controlled by self.label_mode:
            - "center"   : use only the center frame; discard the window if the center label is 0
            - "majority" : majority vote over non-zero labels within the window
        """
        T, C = X_cont.shape
        W = self.ws
        S = self.shift

        X_windows = []
        y_windows = []

        for start in range(0, T - W + 1, S):
            end = start + W
            Xw = X_cont[start:end, :]   # [W, C]
            yw = y_cont[start:end]      # [W]

            # 1) If the entire window is unlabeled (all zeros), skip it directly
            nonzero = yw[yw > 0]
            if nonzero.size == 0:
                continue

            # 2) Select the label according to label_mode
            if self.label_mode == "center":
                # Center-frame strategy: discard the window if the center label is 0
                c = W // 2
                center_label = int(yw[c])
                if center_label == 0:
                    continue
                main_label = center_label

            elif self.label_mode == "majority":
                # Majority-vote strategy: use the mode of non-zero labels within the window
                values, counts = np.unique(nonzero, return_counts=True)
                main_label = int(values[np.argmax(counts)])

            else:
                raise ValueError(f"Unknown label_mode: {self.label_mode}")

            X_windows.append(Xw)
            y_windows.append(main_label)

        if len(X_windows) == 0:
            raise RuntimeError("No windows built. Check window params / labels / label_mode.")

        X_windows = np.stack(X_windows, axis=0).astype(np.float32)  # [N, W, C]
        y_windows = np.array(y_windows, dtype=np.int64)             # [N]

        return X_windows, y_windows

    def add_physical_features(self, X: np.ndarray, fs: float = 50.0):
        """
        X: [T, 6] = [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]

        The output channel dimension is determined by self.in_channels:
            - 6  : no physical features added; return the original 6 channels
            - 9  : add [acc_norm, gyro_norm, jerk] (3 additional features)
            - 11 : based on 9 channels, further add [pitch, roll] (2 additional features)

        Returns:
            X_out: [T, self.in_channels]
        """
        assert X.shape[1] == 6, f"Expect 6 channels (acc+gyro), got {X.shape[1]}"

        # original 6 channels
        acc = X[:, 0:3]   # [T,3]
        gyro = X[:, 3:6]  # [T,3]


        if self.in_channels == 6:
            return X  # [T, 6]


        acc_norm = np.linalg.norm(acc, axis=1, keepdims=True)   # [T,1]
        gyro_norm = np.linalg.norm(gyro, axis=1, keepdims=True) # [T,1]

        ax, ay, az = acc[:, 0], acc[:, 1], acc[:, 2]
        eps = 1e-8

        # jerk of acc_norm
        jerk = np.zeros_like(acc_norm)
        jerk[1:, 0] = np.abs(acc_norm[1:, 0] - acc_norm[:-1, 0]) * fs  # 差分 * 采样率

        # pitch / roll 
        pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2) + eps)   # [T]
        roll  = np.arctan2( ay, az + eps)                       # [T]

        pitch = pitch[:, None]  # [T,1]
        roll  = roll[:, None]   # [T,1]

        # --------------------
        # 9 channels
        # --------------------
        if self.in_channels == 9:
            phys_feats = np.concatenate(
                [acc_norm, gyro_norm, jerk],  # [T, 3]
                axis=1
            )
            X_out = np.concatenate([X, phys_feats], axis=1)  # [T, 9]
            return X_out

        # --------------------
        # 11 channels
        # --------------------
        if self.in_channels == 11:
            phys_feats = np.concatenate(
                [acc_norm, gyro_norm, jerk, pitch, roll],  # [T, 5]
                axis=1
            )
            X_out = np.concatenate([X, phys_feats], axis=1)  # [T, 11]
            return X_out

        
        raise ValueError(
            f"Unsupported in_channels={self.in_channels}, "
            f"expected one of [6, 9, 11]."
        )


    def load_exp_raw_6ch(self, exp_id: int, user_id: int):
        """
        raw_dir: Root directory, e.g. "RawData"
        exp_id:  Experiment ID (int), e.g. 1, 2, ...
        user_id: User ID (int), e.g. 1, 2, ...

        Returns:
            X: np.ndarray, shape [T, 6]
            Column order: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
        """
        acc_path = os.path.join(
            self.raw_dir,
            f"acc_exp{exp_id:02d}_user{user_id:02d}.txt"
        )
        gyro_path = os.path.join(
            self.raw_dir,
            f"gyro_exp{exp_id:02d}_user{user_id:02d}.txt"
        )

        
        acc = np.loadtxt(acc_path)   # [T_acc, 3]
        gyro = np.loadtxt(gyro_path) # [T_gyro, 3]

        # In most cases, T_acc == T_gyro; it is still recommended to assert this explicitly.
        if acc.shape[0] != gyro.shape[0]:
            raise ValueError(
                f"exp {exp_id}, user {user_id}: acc len={acc.shape[0]}, gyro len={gyro.shape[0]}"
            )

        X = np.concatenate([acc, gyro], axis=1)  # [T, 6]
        return X

    def load_hapt_labels(self):
        """
        Read the HAPT RawData/labels.txt file and return a DataFrame with column names.
        """
        label_path = os.path.join(self.raw_dir, "labels.txt")

        if not os.path.isfile(label_path):
            raise FileNotFoundError(f"labels.txt not found at: {label_path}")

        df = pd.read_csv(label_path, sep=r"\s+", header=None)
        df.columns = ["exp_id", "user_id", "act_id", "start", "end"]

        return df

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # numpy → torch
        Xw = torch.from_numpy(self.X[idx]).float()      # [W, C]
        yw = torch.tensor(self.y[idx], dtype=torch.long)  # 标量

        
        if self.augment and self.split == "train":
            # Xw = self._augment_window(Xw)  
            pass

        return Xw, yw

    # === MOD 4: Provide an iterator interface over full_sequences for transition matrix statistics ===
    def iter_full_sequences(self):
        """
        Iterate over the full continuous sequences stored for the current split
        (available only for the train split).

        Yields:
            user_id : int
            X_user  : np.ndarray, shape [T, C]  (already normalized)
            y_user  : np.ndarray, shape [T]     (0 = unlabeled, 1..12 = activity)
        """
        for seq in self.full_sequences:
            yield seq["user_id"], seq["X"], seq["y"]

    def get_full_label_sequences(self):
        """
        Return a list where each element is the continuous y label sequence of a user.
        """
        return [seq["y"] for seq in self.full_sequences]


class RunningStats:
    """
    Compute channel-wise mean and standard deviation online over continuous X_exp.
    """
    def __init__(self, eps: float = 1e-8):
        self.count = 0
        self.mean = None  # (C,)
        self.M2   = None  # (C,)
        self.eps  = eps

    def update(self, X: np.ndarray):
        """
        X: [T, C], which will be flattened to [N, C] for statistics.
        """
        if X.ndim != 2:
            raise ValueError(f"Expect X.ndim == 2, got {X.ndim}")

        x = X.reshape(-1, X.shape[-1]).astype(np.float64)  # [N, C]
        n = x.shape[0]
        if n == 0:
            return

        if self.mean is None:
            self.mean = x.mean(axis=0)
            self.M2   = ((x - self.mean) ** 2).sum(axis=0)
            self.count = n
        else:
            new_count = self.count + n
            delta = x.mean(axis=0) - self.mean
            self.mean = self.mean + delta * (n / new_count)
            self.M2 = self.M2 + ((x - self.mean) ** 2).sum(axis=0)
            self.count = new_count

    def finalize(self):
        """
        Return:
            mean: (C,)
            std : (C,)
        """
        if self.count <= 1 or self.mean is None:
            raise RuntimeError("Not enough data to compute statistics.")
        var = self.M2 / max(self.count - 1, 1)
        std = np.sqrt(np.maximum(var, self.eps))
        return self.mean.astype(np.float32), std.astype(np.float32)
