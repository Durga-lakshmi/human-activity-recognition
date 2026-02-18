# transition_mask
import torch
import os
# learnable_prior
import torch.nn as nn
import torch.nn.functional as F

# bayesian_reweight
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import os

def build_transition_mask(num_classes=12):
    mask = torch.zeros(num_classes, num_classes)

    # =========================================================
    # 0–2: WALKING Category → No forced constraints (fully permitted)
    # =========================================================
    #for prev in [0, 1, 2]:
    #    mask[prev, :] = 1

    # =========================================================
    # 0–2：WALKING 类
    # =========================================================
    # 原来：不强行约束（全允许）
    # for prev in [0, 1, 2]:
    #     mask[prev, :] = 1

    # >>> CHANGED: 只允许
    #     - 自环（继续走路）

    for prev in [0, 1, 2]:
        mask[prev, prev] = 1   # WALK_i -> WALK_i
        

    # =========================================================
    # 3–5: Static State FSM
    # =========================================================
    # Self-holding
    mask[3, 3] = 1  # SITTING
    mask[4, 4] = 1  # STANDING
    mask[5, 5] = 1  # LAYING

    # -----------------
    # Legitimate static redirections
    # -----------------
    mask[3, 4] = 1  # SIT -> STAND
    mask[4, 3] = 1  # STAND -> SIT
    mask[3, 5] = 1  # SIT -> LIE
    mask[5, 3] = 1  # LIE -> SIT
    mask[4, 5] = 1  # STAND -> LIE
    mask[5, 4] = 1  # LIE -> STAND

    # =========================================================
    # 6–11：Change Class (determined by prev)
    # =========================================================
    # STAND_TO_SIT
    mask[4, 6] = 1

    # SIT_TO_STAND
    mask[3, 7] = 1

    # SIT_TO_LIE
    mask[3, 8] = 1

    # LIE_TO_SIT
    mask[5, 9] = 1

    # STAND_TO_LIE
    mask[4, 10] = 1

    # LIE_TO_STAND
    mask[5, 11] = 1

    # =========================================================
    # Transitional Class: “Self-Loop + Completion”
    # =========================================================
    mask[6, 6] = 1   # STAND_TO_SIT -> self
    mask[6, 3] = 1   # STAND_TO_SIT -> SITTING

    mask[7, 7] = 1   # SIT_TO_STAND -> self
    mask[7, 4] = 1   # SIT_TO_STAND -> STANDING

    mask[8, 8] = 1   # SIT_TO_LIE -> self
    mask[8, 5] = 1   # SIT_TO_LIE -> LAYING

    mask[9, 9] = 1   # LIE_TO_SIT -> self
    mask[9, 3] = 1   # LIE_TO_SIT -> SITTING

    mask[10, 10] = 1 # STAND_TO_LIE -> self
    mask[10, 5] = 1  # STAND_TO_LIE -> LAYING

    mask[11, 11] = 1 # LIE_TO_STAND -> self
    mask[11, 4] = 1  # LIE_TO_STAND -> STANDING



    return mask



def build_fixed_transition_logits_full(
    mask: torch.Tensor,
    w_self_walk: float = 9.0,
    w_walk_to_stand: float = 1.0,
    w_self_posture: float = 8.0,
    w_posture_to_trans: float = 1.0,
    w_self_trans: float = 3.0,
    w_trans_to_posture: float = 7.0,
) -> torch.Tensor:
    """
    Construct a complete 12×12 transition matrix logits based on “Weights” for FixedTransitionPrior.

        Category conventions (HAPT):
        0: WALKING
        1: WALKING_UP
        2: WALKING_DOWN
        3: SITTING
        4: STANDING
        5: LAYING
        6: STAND_TO_SIT
        7: SIT_TO_STAND
        8: SIT_TO_LIE
        9: LIE_TO_SIT
        10: STAND_TO_LIE
        11: LIE_TO_STAND

        Approach:
        - For each row, first write the “relative weight,” then normalize it to a probability within that row, and finally take the log.
        - Only write weights at positions where mask[i, j] == 1; keep other positions at -50 (virtually 0 probability).
    """
    num_states = mask.size(0)
    device = mask.device

    # Initialization: All values set to -50; subsequent overwrites occur only at weighted positions.
    init_logits = torch.full(
        (num_states, num_states), fill_value=-50.0, dtype=torch.float32, device=device
    )

    # Store “weights” for easy row-wise normalization.
    weights = torch.zeros_like(init_logits)

    def add_weight(i: int, j: int, w: float):
        """Apply weighting only at valid positions"""
        if i < num_states and j < num_states and mask[i, j] == 1 and w > 0:
            weights[i, j] = w

    # ==============================
    # 0,1,2: WALKING family
    # ==============================
    # High entropy + strong self-loop:
    for i in [0, 1, 2]:
        add_weight(i, i, w_self_walk)       # self
        #add_weight(i, 4, w_walk_to_stand)   # -> STANDING

    # ==============================
    # 3,4,5: Posture (SIT / STAND / LIE)
    # ==============================
    # 3: SITTING -> SITTING / SIT_TO_STAND(7) / SIT_TO_LIE(8)
    add_weight(3, 3, w_self_posture)
    add_weight(3, 7, w_posture_to_trans)    # SIT_TO_STAND
    add_weight(3, 8, w_posture_to_trans)    # SIT_TO_LIE

    # 4: STANDING -> STANDING / STAND_TO_SIT(6) / STAND_TO_LIE(10)
    add_weight(4, 4, w_self_posture)
    add_weight(4, 6, w_posture_to_trans)    # STAND_TO_SIT
    add_weight(4, 10, w_posture_to_trans)   # STAND_TO_LIE

    # 5: LAYING -> LAYING / LIE_TO_SIT(9) / LIE_TO_STAND(11)
    add_weight(5, 5, w_self_posture)
    add_weight(5, 9, w_posture_to_trans)    # LIE_TO_SIT
    add_weight(5, 11, w_posture_to_trans)   # LIE_TO_STAND

    # ==============================
    # 6..11: Transition states
    # Each transition: mainly self-loop + flow to the unique target posture; all others remain 0 (controlled by floor).
    # ==============================
    # 6: STAND_TO_SIT -> self / SITTING(3)
    add_weight(6, 6, w_self_trans)
    add_weight(6, 3, w_trans_to_posture)

    # 7: SIT_TO_STAND -> self / STANDING(4)
    add_weight(7, 7, w_self_trans)
    add_weight(7, 4, w_trans_to_posture)

    # 8: SIT_TO_LIE -> self / LAYING(5)
    add_weight(8, 8, w_self_trans)
    add_weight(8, 5, w_trans_to_posture)

    # 9: LIE_TO_SIT -> self / SITTING(3)
    add_weight(9, 9, w_self_trans)
    add_weight(9, 3, w_trans_to_posture)

    # 10: STAND_TO_LIE -> self / LAYING(5)
    add_weight(10, 10, w_self_trans)
    add_weight(10, 5, w_trans_to_posture)

    # 11: LIE_TO_STAND -> self / STANDING(4)
    add_weight(11, 11, w_self_trans)
    add_weight(11, 4, w_trans_to_posture)

    # ==============================
    # weights[i, :] -> probs[i, :] -> log probs
    # ==============================
    for i in range(num_states):
        row_w = weights[i]
        
        pos = row_w > 0
        if pos.any():
            row_sum = row_w[pos].sum()
            probs = row_w.clone()
            probs[pos] = row_w[pos] / row_sum
            # 取 log 写入 logits
            init_logits[i, pos] = torch.log(probs[pos])

    return init_logits

### used in transition prior !!! 
#according to real data statistics of train set
def build_fixed_transition_logits_from_data(
    mask: torch.Tensor,
    w_self_walk: float = 5.0,
    w_other_walk: float = 1.0,
    w_self_posture: float = 8.0,
    w_posture_to_trans: float = 1.0,
    w_self_trans: float = 4.0,
    w_trans_to_posture: float = 2.0,
    floor: float = -50.0,
) -> torch.Tensor:
    """
    Construct a complete 12×12 transition matrix logits based on “Weights” for FixedTransitionPrior.

    Category conventions (HAPT):
      0: WALKING
      1: WALKING_UP
      2: WALKING_DOWN
      3: SITTING
      4: STANDING
      5: LAYING
      6: STAND_TO_SIT
      7: SIT_TO_STAND
      8: SIT_TO_LIE
      9: LIE_TO_SIT
      10: STAND_TO_LIE
      11: LIE_TO_STAND

    Design Philosophy (fully aligned with our prior analysis):
      - walk(0,1,2): High entropy + strong self-loops, expressing only “temporal continuity” without hard-coding semantic transitions.
      - posture(3,4,5): The only structured block, permitting only:
            Self-loops + flows to corresponding transition classes (no direct posture→posture).
      - transition(6..11): Primarily self-loops + flow toward target postures, all others set to 0.
      - Write weights only at positions where mask[i, j] == 1; maintain floor (nearly 0 probability) elsewhere.
      - Per row: First write “relative weights” → Normalize row to probability → Take log.
    """
    num_states = mask.size(0)
    device = mask.device

    # Initialization
    init_logits = torch.full(
        (num_states, num_states),
        fill_value=floor,
        dtype=torch.float32,
        device=device,
    )

    # Store “weights” for easy row-wise normalization.
    weights = torch.zeros_like(init_logits)

    def add_weight(i: int, j: int, w: float):
        """Apply weighting only at valid positions"""
        if (
            0 <= i < num_states
            and 0 <= j < num_states
            and w > 0.0
            and mask[i, j] == 1
        ):
            weights[i, j] = w


    # ==============================
    # 0,1,2: WALKING family
    # High entropy + strong self-loop:
    #   - Assign a base weight w_other_walk for all permissible j (mask[i,j]==1)
    #   - Add an additional weight w_self_walk for the walk itself
    # ==============================
    walk_ids = [0, 1, 2]
    for i in walk_ids:
        # First, assign a small weight to all permissible targets to create high entropy.
        for j in range(num_states):
            if mask[i, j] == 1:
                add_weight(i, j, w_other_walk)

        # Then, add a strong self-loop weight to the walk itself.
        add_weight(i, i, weights[i, i].item() + w_self_walk)

    # ==============================
    # 3,4,5: Posture (SIT / STAND / LIE)
    # Only allow:
    #   Self-loop + Transitions corresponding to flow direction
    # Do not directly transition from posture to posture
    # ==============================
    # 3: SITTING -> SITTING / SIT_TO_STAND(7) / SIT_TO_LIE(8)
    add_weight(3, 3, w_self_posture)
    add_weight(3, 7, w_posture_to_trans)    # SIT_TO_STAND
    add_weight(3, 8, w_posture_to_trans)    # SIT_TO_LIE

    # 4: STANDING -> STANDING / STAND_TO_SIT(6) / STAND_TO_LIE(10)
    add_weight(4, 4, w_self_posture)
    add_weight(4, 6, w_posture_to_trans)    # STAND_TO_SIT
    add_weight(4, 10, w_posture_to_trans)   # STAND_TO_LIE

    # 5: LAYING -> LAYING / LIE_TO_SIT(9) / LIE_TO_STAND(11)
    add_weight(5, 5, w_self_posture)
    add_weight(5, 9, w_posture_to_trans)    # LIE_TO_SIT
    add_weight(5, 11, w_posture_to_trans)   # LIE_TO_STAND

    # ==============================
    # 6..11: Transition states
    # Each transition: self-loop + toward unique target pose
    # All others remain 0 (controlled by floor to be extremely low probability)
    # ==============================
    # 6: STAND_TO_SIT -> self / SITTING(3)
    add_weight(6, 6, w_self_trans)
    add_weight(6, 3, w_trans_to_posture)

    # 7: SIT_TO_STAND -> self / STANDING(4)
    add_weight(7, 7, w_self_trans)
    add_weight(7, 4, w_trans_to_posture)

    # 8: SIT_TO_LIE -> self / LAYING(5)
    add_weight(8, 8, w_self_trans)
    add_weight(8, 5, w_trans_to_posture)

    # 9: LIE_TO_SIT -> self / SITTING(3)
    add_weight(9, 9, w_self_trans)
    add_weight(9, 3, w_trans_to_posture)

    # 10: STAND_TO_LIE -> self / LAYING(5)
    add_weight(10, 10, w_self_trans)
    add_weight(10, 5, w_trans_to_posture)

    # 11: LIE_TO_STAND -> self / STANDING(4)
    add_weight(11, 11, w_self_trans)
    add_weight(11, 4, w_trans_to_posture)

    # ==============================
    # Normalization：weights[i, :] -> probs[i, :] -> log probs
    # ==============================
    for i in range(num_states):
        row_w = weights[i]
        pos = row_w > 0
        if pos.any():
            row_sum = row_w[pos].sum()
            probs = row_w.clone()
            probs[pos] = row_w[pos] / row_sum
            init_logits[i, pos] = torch.log(probs[pos])

    return init_logits


class FixedTransitionPrior(nn.Module):
    def __init__(self, mask, init_logits):
        super().__init__()
        # as buffers, following the .to(device) path but not participating in gradient calculations.
        self.register_buffer("mask", mask)
        self.register_buffer("logits", init_logits)

    def forward(self, prev_labels):
        # fixed: log p(y_t | y_{t-1})
        logits = self.logits.masked_fill(self.mask == 0, -50)
        log_probs = F.log_softmax(logits, dim=1)
        return log_probs[prev_labels]   # [B, C]
        



def bayesian_reweight_learned(logits, prev_labels, transition_prior):
    """
    logits: [B, C]
    prev_labels: [B]

    Returns: Normalized posterior log-probability, shape [B, C]
    """
    # The emission portion is normalized first.
    log_p_nn = F.log_softmax(logits, dim=-1)      # [B, C]
    log_prior = transition_prior(prev_labels)     # [B, C], log p(y_t | y_{t-1})

    # First perform the unnormalized log-sum, then apply log_softmax once to obtain the true log posterior.
    log_unnorm = log_p_nn + log_prior            # [B, C]
    log_post = F.log_softmax(log_unnorm, dim=-1) # [B, C]

    return log_post

def plot_transition_prior(prior_module, class_names):
    """
    prior_module: A learnable prior module with .logits and .mask
    class_names: A list of class names with length C
    """

    with torch.no_grad():
        logits = prior_module.logits
        mask = prior_module.mask

        # Consistent with forward: mask out disallowed transitions
        logits = logits.masked_fill(mask == 0, -50)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    class_names = [
        "WALKING", "WALKING_UP", "WALKING_DOWN",
        "SITTING", "STANDING", "LAYING",
        "STAND→SIT", "SIT→STAND", "SIT→LIE",
        "LIE→SIT", "STAND→LIE", "LIE→STAND"
    ]

    C = len(class_names)

    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        probs,
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="viridis",
        annot=False,
        cbar_kws={"label": "P(y_t | y_{t-1})"}
    )

    
    ax.set_xlabel("Current state  y_t", fontsize=12)
    ax.set_ylabel("Previous state  y_{t-1}", fontsize=12)
    ax.set_title("Transition Prior  P(y_t | y_{t-1})", fontsize=14)

    ax.tick_params(axis='x', labelrotation=45, labelsize=10)
    ax.tick_params(axis='y', labelrotation=0, labelsize=10)

    # —— Semantic Grouping: 3 Dynamic | 3 Static | 6 Transitional ——
    split1 = 3
    split2 = 6

    ax.axhline(split1, color="white", linewidth=2)
    ax.axvline(split1, color="white", linewidth=2)

    ax.text(split1 / 2, -0.8, "Dynamic classes", ha="center", va="center", fontsize=11)
    ax.text((split1 + split2) / 2, -0.8, "Static classes", ha="center", va="center", fontsize=11)
    ax.text(split2 + (C - split2) / 2, -0.8, "Transition classes", ha="center", va="center", fontsize=11)

    # ===============================
    # Only report non-zero probabilities (rounded to two decimal places)
    # ===============================
    thr = 1e-6
    for i in range(C):
        for j in range(C):
            val = probs[i, j]
            if val > thr:
                ax.text(
                    j + 0.5, i + 0.5,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=13,
                    fontweight="bold",
                    color="white" #if val > 0.6 else "black"
                )

    plt.tight_layout()

    save_dir = "notebooks/EDA/HAPT_post/transition_matrix"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(
        os.path.join(save_dir, "transition_prior.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    #plt.show()


def bayesian_greedy_decode(
    logits_batch: torch.Tensor,
    transition_prior,
    lambda_bayes: float = 0.1,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Perform prior-based greedy decoding on the logits [B, T, C].

    Parameters:
      - logits_batch: [B, T, C], raw logits output by the model
      - transition_prior: LearnableTransitionPrior or FixedTransitionPrior
      - lambda_bayes: Prior weight (recommended initial values: 0.0 / 0.1 / 0.2)
      - lengths: [B], effective length of each sequence (optional)

    Returns:
        - preds: [B, T], predicted labels for each time step (positions where t >= length may be omitted)
    """
    assert logits_batch.dim() == 3, "logits_batch 必须是 [B, T, C]"
    B, T, C = logits_batch.shape
    device = logits_batch.device

    # First convert the NN part into log probabilities to avoid repeating softmax for each frame.
    log_p_nn = F.log_softmax(logits_batch, dim=-1)  # [B, T, C]

    # predictions to return
    preds = torch.zeros(B, T, dtype=torch.long, device=device)

    if lengths is None:
        lengths = torch.full((B,), T, dtype=torch.long, device=device)

    # t = 0: No prior frame information; directly uses NN results
    preds[:, 0] = log_p_nn[:, 0].argmax(dim=-1)

    # t >= 1: Use previous frame prediction + transition prior for reweighting
    for t in range(1, T):
        # Valid Sequence Mask
        valid_mask = (lengths > t)
        if not valid_mask.any():
            break

        prev_labels = preds[:, t - 1]           # [B]
        log_prior = transition_prior(prev_labels)  # [B, C]

        log_unnorm = log_p_nn[:, t, :] + lambda_bayes * log_prior  # [B, C]
        log_post = F.log_softmax(log_unnorm, dim=-1)               # [B, C]

        # Update predictions only for valid positions
        cur_preds = log_post.argmax(dim=-1)  # [B]
        preds[valid_mask, t] = cur_preds[valid_mask]

    return preds

class_names = [
    "WALKING", "WALKING_UP", "WALKING_DOWN",
    "SITTING", "STANDING", "LAYING",
    "STAND→SIT", "SIT→STAND", "SIT→LIE",
    "LIE→SIT", "STAND→LIE", "LIE→STAND"
]


def count_transitions_simple(labels_0based: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Simplified Version: Given labels already in the range 0..K-1, count the number of transitions between adjacent frames.

    labels_0based: [T], int in [0, num_classes-1]
    """
    labels_0based = np.asarray(labels_0based, dtype=np.int64)
    if labels_0based.shape[0] < 2:
        return np.zeros((num_classes, num_classes), dtype=np.int64)

    count = np.zeros((num_classes, num_classes), dtype=np.int64)
    prev = labels_0based[:-1]
    curr = labels_0based[1:]

    for i in range(num_classes):
        idx = (prev == i)
        if not np.any(idx):
            continue
        hist = np.bincount(curr[idx], minlength=num_classes)
        count[i] += hist

    return count


def counts_to_probs(count: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    count -> Probability Matrix (Row Normalization + Laplace Smoothing)

    P[i, j] = (count[i, j] + alpha) / (∑_j count[i, j] + alpha * K)
    """
    count = np.asarray(count, dtype=np.float64)
    K = count.shape[0]

    count_smooth = count + alpha
    row_sum = count_smooth.sum(axis=1, keepdims=True)
    P = count_smooth / row_sum
    return P


def compute_transition_from_dataset(dataset, num_classes: int = 12):
    """
    Calculate transition matrices for two versions from Dataset.full_sequences (recommended only on train_dataset):

    - count_raw   : Raw timeline (no gap crossing), counts only transitions where y(t-1)>0 and y(t)>0
    - count_nogap: Transitions counted after removing all frames where y==0 (activities spanning gaps are merged)

    y encoding convention:
        0          : Unlabeled / gap
        1..K (K=12): Valid categories

    Returns:
        count_raw, count_nogap, P_raw, P_nogap
    """
    count_raw = np.zeros((num_classes, num_classes), dtype=np.int64)
    count_nogap = np.zeros((num_classes, num_classes), dtype=np.int64)

    for user_id, X_user, y_user in dataset.iter_full_sequences():
        y_user = np.asarray(y_user, dtype=np.int64)

        # 1) Original Timeline Statistics: Only counts non-zero to non-zero transitions.
        for t in range(1, len(y_user)):
            a = y_user[t - 1]
            b = y_user[t]
            if a == 0 or b == 0:
                continue  # If there's a gap, skip this pair
            i = a - 1  # 1..K -> 0..K-1
            j = b - 1
            if 0 <= i < num_classes and 0 <= j < num_classes:
                count_raw[i, j] += 1

        # 2) Gap removal: First delete all y==0, then count the compressed sequence.
        y_nogap = y_user[y_user > 0]  # Keep only 1..K
        if y_nogap.shape[0] >= 2:
            labels_0based = y_nogap - 1  # -> [0..K-1]
            count_nogap += count_transitions_simple(labels_0based, num_classes)

    P_raw = counts_to_probs(count_raw, alpha=1.0)
    P_nogap = counts_to_probs(count_nogap, alpha=1.0)
    return count_raw, count_nogap, P_raw, P_nogap



def print_transition_summary(P, class_names=None, topk=3, title="Transition Matrix"):
    """
    Print the top-k most probable transitions for each state to facilitate sanity checks.
    """
    K = P.shape[0]
    print(f"\n===== {title} =====")
    for i in range(K):
        row = P[i]
        idx = np.argsort(row)[::-1][:topk]
        name_i = class_names[i] if class_names else f"class_{i}"
        tops = ", ".join(
            [f"{class_names[j] if class_names else j}:{row[j]:.3f}" for j in idx]
        )
        print(f"{name_i:>15s} -> {tops}")
    print("=" * 40)


def save_transition_matrix(P,
                           out_dir: str,
                           name: str,
                           class_names=None,
                           save_heatmap: bool = True):
    """
    Save the transfer matrix to:
      - .npy (numerical)
      - .csv (human-readable)
      - .png (heatmap)
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) save npy
    npy_path = os.path.join(out_dir, f"{name}.npy")
    np.save(npy_path, P)

    # 2) save csv
    if class_names is None:
        class_names = [f"class_{i}" for i in range(P.shape[0])]

    df = pd.DataFrame(P, index=class_names, columns=class_names)
    csv_path = os.path.join(out_dir, f"{name}.csv")
    df.to_csv(csv_path)

    # 3) save heatmap
    if save_heatmap:
        plt.figure(figsize=(8, 6))
        im = plt.imshow(P, cmap="viridis")
        plt.colorbar(im, fraction=0.046)

        plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
        plt.yticks(range(len(class_names)), class_names)

        plt.title(name)
        plt.tight_layout()
        png_path = os.path.join(out_dir, f"{name}.png")
        plt.savefig(png_path, dpi=200)
        plt.close()

    print(f"[Saved] {name}:")
    print(f"  - {npy_path}")
    print(f"  - {csv_path}")
    if save_heatmap:
        print(f"  - {png_path}")