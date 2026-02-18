import hydra
from hydra.utils import instantiate
import torch
import wandb
import os
import numpy as np
from datasets import get_dataset
from models import get_model
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report
)

from module_training.transition import (
    build_transition_mask,
    FixedTransitionPrior,
    build_fixed_transition_logits_full,
    bayesian_reweight_learned,
    bayesian_greedy_decode
)

from evaluator import Evaluator_HAPT 
from module_training.transition import (
    build_transition_mask,
    build_fixed_transition_logits_full,
    build_fixed_transition_logits_from_data,
    FixedTransitionPrior,
    bayesian_reweight_learned,
    plot_transition_prior
)

from tools.cm_viz import plot_confusion_matrix,plot_confusion_matrix_counts
from tools.tsne_vis import run_tsne_visualization

from hydra import  compose
from omegaconf import OmegaConf,DictConfig


def apply_best_params(cfg, best):
    """
    Write the hyperparameters from best_optuna_params.yaml back into cfg
    according to the mapping rules defined in the Optuna objective.

    Note:
    This function assumes that the best parameter set contains the following keys:
    - lr
    - optim_lr
    - weight_decay
    - cnn_channels
    - tcn_channels
    - tcn_dropout
    - tcn_dilations_pattern
    - lambda_bayes / lambda_phys / lambda_trans / lambda_temp
    """

    # ---------- lr ----------
    if "lr" in best:
        cfg.lr = float(best.lr)

    # Optimizer lr / weight_decay
    if "optim_lr" in best:
        cfg.train_module.optimizer.lr = float(best.optim_lr)
    if "weight_decay" in best:
        cfg.train_module.optimizer.weight_decay = float(best.weight_decay)

    # ---------- Model Structure ----------
    if "cnn_channels" in best:
        cfg.model.cnn_channels = int(best.cnn_channels)
    if "tcn_channels" in best:
        cfg.model.tcn_channels = int(best.tcn_channels)
    if "tcn_dropout" in best:
        cfg.model.tcn_dropout = float(best.tcn_dropout)

    # dilation pattern -> list
    if "tcn_dilations_pattern" in best:
        pattern = str(best.tcn_dilations_pattern)
        pattern_map = {
            "1_2_4":   [1, 2, 4],
            "1_2_4_8": [1, 2, 4, 8],
            "2_4":     [2, 4],
            "2_4_8":   [2, 4, 8],
        }
        if pattern not in pattern_map:
            raise ValueError(f"Unknown tcn_dilations_pattern: {pattern}")
        cfg.model.tcn_dilations = pattern_map[pattern]

    # ---------- loss lamda ----------
    if "lambda_bayes" in best:
        cfg.train.lambda_bayes = float(best.lambda_bayes)
    if "lambda_phys" in best:
        cfg.train.lambda_phys = float(best.lambda_phys)
    if "lambda_trans" in best:
        cfg.train.lambda_trans = float(best.lambda_trans)
    if "lambda_temp" in best:
        cfg.train.lambda_temp = float(best.lambda_temp)

    return cfg



def render_eval_report_from_results(
    *,
    dataset_name: str,
    num_classes: int,
    results: dict,
    save_path: str | None = None,
):
    all_targets = results["all_targets"]
    all_preds = results["all_preds"]
    cm = results["confusion_matrix"]

    lines = []

    lines.append("=" * 60)
    lines.append(f"DATASET : {dataset_name}")
    lines.append(f"CLASSES : {num_classes}")
    lines.append("=" * 60)

    lines.append("\n[Evaluation Results]")
    lines.append(f"Loss            : {results['loss']:.6f}")
    lines.append(f"Accuracy        : {results['accuracy'] * 100:.2f}%")
    lines.append(f"Precision (mac) : {results['precision']:.4f}")
    lines.append(f"Recall    (mac) : {results['recall']:.4f}")
    lines.append(f"F1-score  (mac) : {results['f1']:.4f}")

    lines.append("\n[Classification Report | labels=0..K-1]")
    lines.append(
        classification_report(
            all_targets,
            all_preds,
            digits=4,
            zero_division=0,
        )
    )

    lines.append("\n[Confusion Matrix]")
    lines.append(np.array2string(cm))
    lines.append("=" * 60)

    report_text = "\n".join(lines)


    print(report_text)


    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report_text)

    return report_text

@hydra.main(version_base=None, config_path="config", config_name="default.yaml")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

    if cfg.check_optuna:
        # -------------------------------
        # 1) Load the base cfg
        # -------------------------------

        # -------------------------------
        # 2) Load the Optuna best parameters
        # -------------------------------
        best_yaml_path = "optuna/best_optuna_params_16.yaml"
        if not os.path.exists(best_yaml_path):
            raise FileNotFoundError(
                f"{best_yaml_path} not found. Please save study.best_params first in runner_optuna.py."
            )

        best_params = OmegaConf.load(best_yaml_path)
        print("[run_best] Loaded best params from best_optuna_params.yaml:")
        print(dict(best_params))

        # 3) Write best_params back into cfg before model instantiation 
        apply_best_params(cfg, best_params)

        # 4) Build a model with the same architecture as the checkpoint using the updated cfg
        
    model = get_model(cfg)
    if cfg.name == "default_HAPT":
        run_test_eval_HAPT(cfg, model, device)
    else:
        run_test_eval_RW(cfg, model, device)


# ======================================================
# Final test evaluation entry point
# ======================================================
def run_test_eval_HAPT(cfg, model, device):
    # --------------------------------------------------
    # 1) Load checkpoint
    # --------------------------------------------------
    if cfg.check_modus:
        ckpt_path = cfg.check_path
    else:
        ckpt_path = cfg.save.path

    ckpt_name = os.path.basename(ckpt_path)
    stem = os.path.splitext(ckpt_name)[0]
    report_path = f"./optuna_log/eval_report/{stem}_eval_report.txt"

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()



    # --------------------------------------------------
    # 2) Load test dataset
    # --------------------------------------------------
    _, test_loader = get_dataset(cfg, split="test")

    # Collect all logits (in temporal order) and raw labels (1..K)
    #logits_test, y_raw = eval_collect_logits(model, test_loader, device)

    use_bayes_prior = cfg.train.use_bayes_prior
    lambda_bayes = cfg.train.lambda_bayes

    if use_bayes_prior:

        num_states = getattr(cfg.train, "num_states", 12)
        mask = build_transition_mask(num_states).to(device)       
        init_logits = build_fixed_transition_logits_full(mask)
        #init_logits = build_fixed_transition_logits_from_data(mask)
        transition_prior = FixedTransitionPrior(mask, init_logits)

    else:
        transition_prior = None


    use_bayes_seq = cfg.eval.use_bayes_seq
    lambda_bayes_seq = cfg.eval.lambda_bayes_seq

    
    evaluator = Evaluator_HAPT(
        cfg=cfg,
        eval_loader=test_loader,
        model=model,
        device=device,
        transition_prior=transition_prior,
        use_bayes_seq=use_bayes_seq,          # <<< only in test
        lambda_bayes_seq=lambda_bayes_seq,
    )
    

    test_results = evaluator.eval()

    render_eval_report_from_results(dataset_name="HAPT",num_classes=cfg.num_classes,results=test_results,save_path= report_path) #"eval_report.txt")

    #test_metrics["confusion_matrix"]

    #if getattr(cfg.eval, "plot_cm", False):
        # Confusion matrix (normalized)
        #plot_confusion_matrix(y_true=test_results["all_targets"],y_pred=test_results["all_preds"],num_classes=cfg.num_classes,save_path=cfg.eval.cm_save_path)

        # Confusion matrix (counts)
        #plot_confusion_matrix_counts(y_true=test_results["all_targets"],y_pred=test_results["all_preds"],num_classes=cfg.num_classes,save_path=cfg.eval.cm_save_path)

    #print("\n===== HAPT Test Set Results (Evaluator) =====")
    #print(f"Accuracy        : {test_metrics['accuracy']*100:.2f}%")
    #print(f"Precision (mac) : {test_metrics['precision']:.4f}")
    #print(f"Recall (mac)    : {test_metrics['recall']:.4f}")
    #print(f"F1-score (mac)  : {test_metrics['f1']:.4f}")

    #print("\nClassification Report ( labels=0..K-1):")
    #print(test_metrics["classification_report"])

    #print("\nConfusion Matrix (labels=0..K-1):")
    #print(test_metrics["confusion_matrix"])


    if getattr(cfg.eval, "do_tsne", False):

        run_tsne_visualization(
            model=model,
            loader=test_loader,
            device=device,
            max_samples=getattr(cfg.eval, "tsne_max_samples", 5000),
            perplexity=getattr(cfg.eval, "tsne_perplexity", 30.0),
            save_path=getattr(cfg.eval, "tsne_save_path", "tsne_test.png"),
        )

def run_test_eval_RW(cfg, model, device):
    # --------------------------------------------------
    # 1. Load best trained HAPT model
    # --------------------------------------------------
    
    #best_model_path = f"artifacts/models/har_HARLSTM_bestmodel93.pth"
    best_model_path = f"artifacts/models/{cfg.dataset.name.lower()}_{cfg.model.name}_bestmodel.pth"
    #if not os.path.exists(os.path.dirname(best_path)):
    #     os.makedirs(os.path.dirname(best_path))
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    # --------------------------------------------------
    # 2. Load test dataset (users 22–27)
    # --------------------------------------------------
    _, test_loader = get_dataset(cfg, split="test")

    all_preds = []
    all_labels = []

    # --------------------------------------------------
    # 3. Forward pass
    # --------------------------------------------------
    with torch.no_grad():
        for x, y in test_loader:
            if isinstance(x, dict):  # multi-position
                x = {pos: data.to(device).float() for pos, data in x.items()}
            else:
                x = x.to(device).float()  
            if cfg.dataset.name.lower() == "hapt":                   
                y = y.argmax(dim=1).to(device)  # [B]
            else:
                y = y.to(device)# [B]
            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())

    # --------------------------------------------------
    # 4. Metrics (MACRO — critical for HAPT)
    # --------------------------------------------------
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    precision_macro = precision_score(all_labels, all_preds, average="macro")
    recall_macro = recall_score(all_labels, all_preds, average="macro")

    print("\n===== HAR Test Set Results =====")
    print(f"Accuracy        : {acc * 100:.2f}%")
    print(f"Precision (mac) : {precision_macro:.4f}")
    print(f"Recall (mac)    : {recall_macro:.4f}")
    print(f"F1-score (mac)  : {f1_macro:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            all_labels,
            all_preds,
            digits=4
        )
    )

    # --------------------------------------------------
    # 5. Confusion Matrix
    # --------------------------------------------------
    cm = confusion_matrix(all_labels, all_preds)
    ACTIVITIES = [
        "walking", "running", "sitting", "standing",
        "lying", "climbingup", "climbingdown", "jumping"
        ]
    def save_confusion_matrix(cm, class_names, save_path="cm.png"):
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.title("Confusion Matrix HAR Test Set")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Confusion matrix saved to {save_path}")

    # Example usage in your script
    # class_names = [f"Class {i}" for i in range(len(cm))]  # replace with actual activity names
    # save_confusion_matrix(cm, ACTIVITIES, save_path="artifacts/plots/HAR/test_cm_har.png")
    print(cm)


if __name__ == "__main__":
    main()


def build_eval_transition_prior(cfg, device):
    """
    Reconstruct a FixedTransitionPrior during evaluation,
    consistent with the training setup (not learned, fixed prior only).
    """

    num_states = getattr(getattr(cfg, "model", None), "num_classes", None)
    if num_states is None:
        num_states = getattr(cfg.train, "num_states", 12)

    # structure constraint mask: [num_states, num_states], bool
    mask = build_transition_mask(num_states).to(device)

    # according to mask and fixes rules to build logits 
    #init_logits = build_fixed_transition_logits_full(mask)
    init_logits = build_fixed_transition_logits_from_data(mask)


    prior = FixedTransitionPrior(mask, init_logits).to(device)
    return prior

def bayesian_decode_sequence(logits: torch.Tensor,
                             transition_prior,
                             posture_only: bool = True) -> torch.Tensor:
    """
    Perform Bayesian decoding on a 1D sequence:
        logits: [N, C], ordered temporally (same as in eval_collect_logits)
        transition_prior: FixedTransitionPrior module
        posture_only: when True, apply the prior only if prev ∈ {3, 4, 5}

    Returns:
        preds: [N], Bayesian-decoded predictions (0..C-1)
    """
    N, C = logits.shape
    device = logits.device

    preds = torch.zeros(N, dtype=torch.long, device=device)

    preds[0] = logits[0].argmax(dim=-1)

    for t in range(1, N):
        prev = preds[t - 1:t]  # shape [1]

        # only apply Bayes reweighting when the previous state is a posture (3,4,5)
        if posture_only and not (3 <= prev.item() <= 5):
            preds[t] = logits[t].argmax(dim=-1)
            continue

        log_post = bayesian_reweight_learned(
            logits[t:t + 1],          # [1, C]
            prev_labels=prev,         # [1]
            transition_prior=transition_prior,
        )                             # [1, C]

        preds[t] = log_post.argmax(dim=-1)
        if t < 5:
            print("[DEBUG] t =", t,
                "prev =", prev.item(),
                "argmax_nn =", int(logits[t].argmax()),
                "argmax_bayes =", int(log_post.argmax()))

    

    return preds


# ======================================================
# Logit-level temporal smoothing
# ======================================================
def smooth_logits_over_time(logits, kernel=None):
    """
    logits: (N, C) numpy array, ordered temporally
    kernel: 1D weight list, e.g., [1, 2, 1] or [1, 1, 1]
    Returns: (N, C) smoothed logits
    """
    import numpy as np

    if kernel is None:
        kernel = [1, 2, 1]  # Assign a higher weight to the center frame
    kernel = np.array(kernel, dtype=np.float32)
    kernel = kernel / kernel.sum()

    N, C = logits.shape
    K = len(kernel)
    pad = K // 2

    # Use edge padding at the boundaries
    padded = np.pad(logits, ((pad, pad), (0, 0)), mode="edge")

    smoothed = np.zeros_like(logits, dtype=np.float32)
    for t in range(N):
        window = padded[t:t+K]          # (K, C)
        # Compute a weighted sum along the temporal dimension
        smoothed[t] = (window * kernel[:, None]).sum(axis=0)

    return smoothed


# ======================================================
# Temporal majority-vote smoothing (majority filter)
# ======================================================
def majority_filter_1d(labels, k: int = 2):
    """
    Apply sliding-window majority-vote smoothing to a 1D label sequence.

    Parameters:
        labels: 1D array-like, predicted label sequence, e.g., [0, 1, 1, 2, 1, ...]
        k: half window size, window length = 2*k + 1
        k = 2 -> window length 5
        k = 3 -> window length 7

    Returns:
        out: smoothed label sequence (np.ndarray, same length)
    """
    import numpy as np

    labels = np.asarray(labels)
    T = len(labels)
    if T == 0:
        return labels

    out = labels.copy()

    for t in range(T):
        left = max(0, t - k)
        right = min(T, t + k + 1)  
        window = labels[left:right]


        vals, counts = np.unique(window, return_counts=True)
        max_count = counts.max()
        # Label candidates with the highest occurrence count
        majority_candidates = vals[counts == max_count]

        if len(majority_candidates) == 1:
            # If there is a clear majority, replace with it
            out[t] = majority_candidates[0]
        else:
            # If there is a tie for the majority, keep the original label to avoid over-smoothing
            out[t] = labels[t]

    return out





def get_state_logits(outputs):

    if isinstance(outputs, dict):
        for key in ["state_logits", "logits", "state"]:
            if key in outputs:
                return outputs[key]
        raise ValueError(
            f"In Model forward can't find state logits, keys = {list(outputs.keys())}"
        )
    return outputs


def eval_collect_logits(model, loader, device):
    """
    Return：
        all_logits: torch.Tensor, shape [N, C]，CPU
        all_targets_raw: torch.Tensor, shape [N], CPU， 1..K
    """
    import torch

    model.eval()
    all_logits = []
    all_targets_raw = []

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict):
                x = batch["x"]
                y = batch["y"]
            elif isinstance(batch, (list, tuple)):
                if len(batch) >= 2:
                    x, y = batch[0], batch[1]
                else:
                    raise ValueError(
                        f"batch is list/tuple, but length < 2, -> (x, y) failed. batch = {batch}"
                    )
            else:

                raise TypeError(
                    f"Unknown batch type: {type(batch)}, content example: {batch}"
                )

            y = y.long()
            if y.ndim > 1:
                y = y.argmax(dim=1)

            mask_valid = (y > 0)
            if mask_valid.sum() == 0:
                continue

            x = x[mask_valid]
            y_raw = y[mask_valid]      

            x = x.to(device)
            y_raw = y_raw.to(device)

            outputs = model(x)
            logits = get_state_logits(outputs)  # [B_valid, C]
            all_logits.append(logits.cpu())
            all_targets_raw.append(y_raw.cpu())



    if len(all_logits) == 0:
        return torch.empty(0, 0), torch.empty(0, dtype=torch.long)

    all_logits = torch.cat(all_logits, dim=0)        # (N, C)
    all_targets_raw = torch.cat(all_targets_raw, dim=0)  # (N,)

    return all_logits, all_targets_raw



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

            try:
                out = model(x, return_feat=True, return_phys_loss=False)
            except TypeError:
                out = model(x)

            feat = None

            # Case A：dict（include feat）
            if isinstance(out, dict):
                if "feat" in out:
                    feat = out["feat"]
                elif "logits" in out:
                    feat = out["logits"]
                else:
                    raise RuntimeError("dict output without 'feat' or 'logits', t-SNE failed.")

            # Case B：tuple / list
            elif isinstance(out, (tuple, list)):
                feat = out[-1]

            # Case C：Tensor
            elif isinstance(out, torch.Tensor):
                
                feat = out

            else:
                raise RuntimeError(
                    f"model(...) returned {type(out)}, cannot be used for t-SNE."
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

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=200,
        init="pca",
        random_state=0,
        verbose=1,
    )
    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, s=3)
    plt.colorbar(scatter)
    plt.title("t-SNE of Model Features")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"[t-SNE] Saved to {save_path}")
    else:
        plt.show()




def load_base_cfg():
    """
    As in optuna_runner, construct the full cfg using Hydra:
    - Equivalent to @hydra.main(config_path='config', config_name='default')
    """
    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name="default")
    return cfg

