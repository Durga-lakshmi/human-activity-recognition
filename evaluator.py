import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from hydra.utils import instantiate
import torch.nn.functional as F
# ============================
# = Bayesian prior relevant ==
# ============================
from module_training.transition import bayesian_reweight_learned

def get_state_logits(outputs):
    """
    Unified extraction of logits used for state classification from model outputs.
    - If outputs is a tensor, return it directly;
    - If it is a dict, search for logits using common keys.
    """
    if isinstance(outputs, dict):
        for key in ["state_logits", "logits", "state"]:
            if key in outputs:
                return outputs[key]
        raise ValueError(
            f"The state logits cannot be found in the dictionary returned by the model’s forward pass. keys = {list(outputs.keys())}"
        )
    return outputs


class Evaluator_HAPT:
    """
    Label conventions are aligned with the Trainer:

    - Raw labels y_raw:
        0 or -1  → ignored
        1..K     → valid classes (e.g., 1..12)

    - Only for cross-entropy loss and metric computation,
        construct y_ce = y_raw - 1 ∈ 0..K-1.
    """

    def __init__(self, cfg, eval_loader, model, device, transition_prior=None, use_bayes_seq: bool = False, lambda_bayes_seq: float = 0.0):
        self.cfg = cfg
        self.eval_loader = eval_loader
        self.model = model
        self.device = device

        self.criterion = instantiate(cfg.train_module.loss).to(self.device)
        self.use_bayes_prior = getattr(cfg.train, "use_bayes_prior", False)
        self.transition_prior = transition_prior

        # Whether to use sequential Bayesian inference during inference.
        self.use_bayes_seq = bool(use_bayes_seq)
        self.lambda_bayes_seq = float(lambda_bayes_seq)

        print("===== EVAL BAYES DEBUG =====")
        print(f"use_bayes_prior      = {self.use_bayes_prior}")
        print(f"has_transition_prior = {self.transition_prior is not None}\n")
        print(f"use_bayes_seq        = {self.use_bayes_seq}")
        print(f"lambda_bayes_seq     = {self.lambda_bayes_seq}")

    @torch.no_grad()
    def eval(self):
        self.model.eval()

        total_loss = 0.0
        total_count = 0

        all_preds = []    # Save 0-based predictions for CE / metric computation
        all_targets = []  # Save 0-based target labels

        for batch in self.eval_loader:
            # Compatible with DataLoader output formats: (x, y) or (x, y, ...)
            x = batch[0]
            y = batch[1]

            # 1) Convert labels to index form
            y = y.long()
            if y.ndim > 1:
                y = y.argmax(dim=1)  # one-hot → index

            # Raw labels: 0 / -1 / 1..K
            y_raw = y

            # 2) Keep only valid labels (> 0)
            mask_valid = (y_raw > 0)
            if mask_valid.sum() == 0:
                continue

            x = x[mask_valid]
            y_raw = y_raw[mask_valid]        # 1..K

            # 3) Construct 0-based labels
            y_ce = y_raw - 1                 # 0..K-1

            x = x.to(self.device)
            y_ce = y_ce.to(self.device)

            # 4) forward
            outputs = self.model(x)
            logits = get_state_logits(outputs)   # [B_valid, K]

            # ====================================
            # === Bayesian reweight (batch) ======
            # ====================================
            if self.use_bayes_prior and self.transition_prior is not None:
                # Use the “previous frame ground-truth labels” as prev_labels
                # (consistent with the loss construction in the Trainer)
                prev_labels = torch.roll(y_ce, shifts=1, dims=0)  # [B_valid]
                if prev_labels.numel() > 1:
                    prev_labels[0] = prev_labels[1]

                log_post = bayesian_reweight_learned(
                    logits,
                    prev_labels=prev_labels,
                    transition_prior=self.transition_prior,
                )
                used_logits = log_post  # 后验 log-prob，用于 argmax
            else:
                used_logits = logits
            # ============================

            # 5) Loss (Evaluator loss is for logging only): still apply CE on raw logits
            loss = self.criterion(logits, y_ce)

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size



            # 6) Predictions
            # =====================================================
            # >>> Add a “sequential Bayesian decoding” layer on top of used_logits
            #     - When use_bayes_seq is disabled, the behavior is exactly equivalent
            #       to the original argmax(used_logits)
            # =====================================================
            if (
                self.use_bayes_seq
                and self.transition_prior is not None
                and self.lambda_bayes_seq > 0.0
            ):
                # First normalize used_logits into log p_NN(y | x)
                log_p_nn = F.log_softmax(used_logits, dim=-1)  # [B_valid, K]
                B_valid, K = log_p_nn.shape

                preds_seq = torch.zeros(B_valid, dtype=torch.long, device=self.device)

                # t = 0: no previous frame, directly use the NN prediction
                preds_seq[0] = log_p_nn[0].argmax(dim=-1)

                # t >= 1: Bayesian reweighting with previous prediction and transition prior
                for t in range(1, B_valid):
                    prev_label = preds_seq[t - 1].unsqueeze(0)        # [1]
                    log_prior = self.transition_prior(prev_label)     # [1, K]，log p(y_t|y_{t-1})

                    log_unnorm = log_p_nn[t] + self.lambda_bayes_seq * log_prior.squeeze(0)  # [K]
                    log_post = F.log_softmax(log_unnorm, dim=-1)                              # [K]

                    preds_seq[t] = log_post.argmax(dim=-1)

                preds = preds_seq  # [B_valid]
            else:
                # Keep original behavior: argmax over used_logits
                preds = torch.argmax(used_logits, dim=1)  # 0..K-1  

            #preds = torch.argmax(used_logits, dim=1)  # 0..K-1

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_ce.cpu().numpy())

        # Extreme case: no valid samples
        if total_count == 0:
            print("Evaluator: no valid samples found (all labels <= 0).")
            dummy_cm = np.zeros((1, 1), dtype=int)
            return {
                "loss": 0.0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "confusion_matrix": dummy_cm,
            }

        avg_loss = total_loss / total_count

        all_targets_np = np.array(all_targets)
        all_preds_np   = np.array(all_preds)

        accuracy = accuracy_score(all_targets_np, all_preds_np)
        precision = precision_score(all_targets_np, all_preds_np, average="macro", zero_division=0)
        recall = recall_score(all_targets_np, all_preds_np, average="macro", zero_division=0)
        f1 = f1_score(all_targets_np, all_preds_np, average="macro", zero_division=0)

        cm = confusion_matrix(all_targets_np, all_preds_np)

        print("Evaluation Results:")
        print(f"  • Average Loss: {avg_loss:.4f}")
        print(f"  • Accuracy: {accuracy * 100:.2f}%")
        print(f"  • Precision (macro): {precision:.4f}")
        print(f"  • Recall (macro): {recall:.4f}")
        print(f"  • F1-score (macro): {f1:.4f}")

        print("\nConfusion Matrix (labels 0..K-1):")
        print(cm)
        print("\n")

        return {
            "all_targets": all_targets_np,
            "all_preds": all_preds_np,
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm,
        }

class Evaluator_RW:
    def __init__(self, cfg, eval_loader, model, device):
        self.cfg = cfg
        self.eval_loader = eval_loader
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    @torch.no_grad()
    def eval(self):
        self.model.eval()

        total_loss = 0.0
        total_count = 0

        all_preds = []
        all_targets = []

        for x, y in self.eval_loader:
            if isinstance(x, dict):  # multi-position
                x = {pos: data.to(self.device).float() for pos, data in x.items()}
            else:
                x = x.to(self.device).float()   # [B, T, 6]
            y = y.to(self.device)

            logits = self.model(x)       
            if self.cfg.dataset.name.lower() == "hapt":                    # [B, num_classes]
                targets = torch.argmax(y, dim=1)       # [B]
            else:
                targets = y

            # Loss
            loss_per_sample = self.criterion(logits, targets)  # [B]
            loss = loss_per_sample.mean()
            # total_loss += loss.item() * x.size(0)  # sum over batch
            # total_count += x.size(0)
            batch_size = next(iter(x.values())).size(0) if isinstance(x, dict) else x.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

            
             # Predictions
            preds = torch.argmax(logits, dim=1) 

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())


        avg_loss = total_loss / total_count

        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average="macro", zero_division=0)
        recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        cm = confusion_matrix(all_targets, all_preds)

        print("Evaluation Results:")
        print(f"  • Average Loss: {avg_loss:.4f}")
        print(f"  • Accuracy: {accuracy * 100:.2f}%")
        print(f"  • Precision (macro): {precision:.4f}")
        print(f"  • Recall (macro): {recall:.4f}")
        print(f"  • F1-score (macro): {f1:.4f}")

        print("\nConfusion Matrix:")
        print(cm)
        print("\n")

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm
        }