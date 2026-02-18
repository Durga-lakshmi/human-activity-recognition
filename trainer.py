import os
import torch
import torch.nn as nn
import numpy as np
import wandb
from collections import deque
import torch.nn.functional as F
import torch.optim as optim
import torch.amp as amp

from sklearn.metrics import f1_score, accuracy_score
from evaluator import Evaluator_HAPT

from hydra.utils import instantiate
from module_training.optim import build_optimizer
from module_training.sched import build_scheduler
from module_training.losses import AsymmetricStateLoss
# ==== Bayesian prior + asymmetric penalty related ====
from module_training.transition import (
    build_transition_mask,
    #build_fixed_transition_logits,
    #build_fixed_transition_logits_prob,
    build_fixed_transition_logits_full,
    build_fixed_transition_logits_from_data,
    #LearnableTransitionPrior,
    FixedTransitionPrior,
    bayesian_reweight_learned,
    plot_transition_prior
)
# ====================================================
# utility function to freeze the backbone
# ====================================================
def freeze_backbone(model):
    # Freeze every module inside backbone
    if hasattr(model, "backbone"):
        for m in model.backbone:          
            for p in m.parameters():
                p.requires_grad = False

    # Ensure classifier is trainable
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True

    if hasattr(model, "head_trans"):
        for p in model.head_trans.parameters():
            p.requires_grad = True

    if hasattr(model, "head_34"):
        for p in model.head_34.parameters():
            p.requires_grad = True

    if hasattr(model, "head_phys"):
        for p in model.head_phys.parameters():
            p.requires_grad = True



# ====================================================
# optimizer only for heads (LR reduced to 1/5)
# ====================================================
def build_finetune_optimizer(model, cfg):
    """
    Fine-tune only classifier / auxiliary heads.
    Assume original learning rate is cfg.train.lr and weight decay is cfg.train.weight_decay.
    """
    base_lr = cfg.train.lr
    wd = getattr(cfg.train, "weight_decay", 0.0)

    param_groups = []

    # Main classification head
    if hasattr(model, "classifier"):
        param_groups.append({"params": model.classifier.parameters()})

    # Transition auxiliary head
    if hasattr(model, "head_trans"):
        param_groups.append({"params": model.head_trans.parameters()})

    # Binary auxiliary head for class 3/4
    if hasattr(model, "head_34"):
        param_groups.append({"params": model.head_34.parameters()})

    # Physical head
    if hasattr(model, "head_phys"):
        param_groups.append({"params": model.head_phys.parameters()})

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=base_lr * 0.2,        # learning rate = 1/5 of the original
        weight_decay=wd,
    )
    return optimizer


class Trainer_HAPT:
    """
    Generic Trainer (compatible with current label conventions) + Bayesian prior + asymmetric penalty.

    Label convention:
      - y_state from Dataset / DataLoader:
            0 or -1  → ignored / invalid label
            1..K     → valid classes (e.g., 1..12)

      - Inside Trainer:
            y_raw keeps original semantics: 0 / -1 / 1..K
            y_ce is constructed temporarily for CrossEntropy + metrics:
                        y_ce = y_raw - 1 → 0..K-1

    - Supports dict outputs from model:
        - Must contain state_logits (or logits/state)
        - Optional trans_logit (transition head, Stage2)
        - Optional phys_loss (physical regularization term)
        - Optional logits_trans (CNN_TCN 6-class transition auxiliary head)

    - Additional components:
        - Bayesian prior (0..K-1):
            cfg.train.use_bayes_prior: bool
            cfg.train.lambda_bayes:    float
        - Asymmetric penalty AsymmetricStateLoss:
            cfg.train.use_asym_penalty: bool
            cfg.train.asym_penalty:      float (>1 effective)
    """

    def __init__(self, cfg, train_loader, val_loader, model, device, max_checkpoints=5,minor_loader=None):
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.to(device)
        
        self.device = device
        self.minor_loader = minor_loader

        # -----------------------------------------------
        # Loss weights (transition + physical + temporal)
        # -----------------------------------------------
        self.lambda_trans = cfg.train.lambda_trans
        self.lambda_phys = cfg.train.lambda_phys
        self.lambda_temp = cfg.train.lambda_temp

        # ==================================
        # ====== Bayesian prior switch =====
        # ==================================
        self.use_bayes_prior = cfg.train.use_bayes_prior
        self.lambda_bayes = cfg.train.lambda_bayes

        if self.use_bayes_prior:
            num_states = getattr(getattr(cfg, "model", None), "num_classes", None)
            if num_states is None:
                num_states = getattr(cfg.train, "num_states", 12)
            mask = build_transition_mask(num_states).to(device)
            init_logits = build_fixed_transition_logits_full(mask)
            #init_logits = build_fixed_transition_logits_from_data(mask)
            self.transition_prior = FixedTransitionPrior(mask, init_logits)
        else:
            self.transition_prior = None

        # -------------------------
        # Training params
        # -------------------------
        self.epochs = cfg.epochs
        self.log_interval = cfg.log_interval

        # state loss（CE）
        self.criterion_state = instantiate(cfg.train_module.loss).to(device)

        # ===================================
        # ==== asymmetric penalty switch ====
        # ===================================
        self.use_asym_penalty = getattr(cfg.train, "use_asym_penalty", False)
        self.asym_penalty_factor = getattr(cfg.train, "asym_penalty", 1.0)

        if self.use_asym_penalty and self.asym_penalty_factor > 1.0:
            # Default [3,4,5] static classes, with asym penalties for transitions between 6..11
            static_classes = getattr(cfg.train, "asym_static_classes", [3, 4, 5])
            transition_range = getattr(cfg.train, "asym_transition_range", (6, 11))
            self.criterion_state = AsymmetricStateLoss(
                base_criterion=self.criterion_state,
                static_classes=static_classes,
                transition_range=transition_range,
                penalty_factor=self.asym_penalty_factor,
                enabled=True,
            )

        # transition loss
        self.criterion_trans = nn.BCEWithLogitsLoss()

        # ==== If Bayes prior is enabled, add the transition_prior parameter to the optimizer. ====
        #params = list(self.model.parameters())
        #if self.use_bayes_prior and self.transition_prior is not None:
        #    params += list(self.transition_prior.parameters())
        #self.optimizer = build_optimizer(cfg, params)
        #self.scheduler = build_scheduler(cfg, self.optimizer)

        # ====================================================
        # Fine_tune switch:
        #   - fine_tune = True : freeze backbone, train heads only, lower LR
        #   - fine_tune = False: original full-model + Bayesian training
        # ====================================================
        if getattr(cfg.train, "fine_tune", False):
            print("[INFO] Fine-tune mode: freeze backbone, train heads only.")
            # 1) Freeze backbone
            freeze_backbone(self.model)
            # 2) Collect head parameters only
            params = []
            if hasattr(self.model, "classifier"):
                params += list(self.model.classifier.parameters())
            if hasattr(self.model, "head_trans"):
                params += list(self.model.head_trans.parameters())
            if hasattr(self.model, "head_34"):
                params += list(self.model.head_34.parameters())
            if hasattr(self.model, "head_phys"):
                params += list(self.model.head_phys.parameters())
            if self.use_bayes_prior and self.transition_prior is not None:
                params += list(self.transition_prior.parameters())

            self.optimizer = build_optimizer(cfg, params)
            for g in self.optimizer.param_groups:
                g["lr"] *= 0.2
            self.scheduler = build_scheduler(cfg, self.optimizer)

        else:
            # --------------- Full Model Training  ----------------------
            params = list(self.model.parameters())
            if self.use_bayes_prior and self.transition_prior is not None:
                params += list(self.transition_prior.parameters())
            self.optimizer = build_optimizer(cfg, params)
            self.scheduler = build_scheduler(cfg, self.optimizer)


        self.evaluator = Evaluator_HAPT(
            cfg=self.cfg,
            eval_loader=self.val_loader,
            model=self.model,
            device=self.device,
            transition_prior=self.transition_prior,
        )

        self.best_val_f1 = -1.0
        self.best_val_acc = 0.0

        # -------------------------
        # Checkpoints
        # -------------------------
        self.max_checkpoints = max_checkpoints
        self.checkpoint_queue = deque()
        self.checkpoint_folder = "artifacts/checkpoints"
        os.makedirs(self.checkpoint_folder, exist_ok=True)

        self.save_path = cfg.save.path
        self.check_path = cfg.check_path
        self.start_epoch = 1

        self.early_stopping = EarlyStopping(
            patience=50,
            min_delta=1e-4,
            verbose=True,
        )

        # ===== Static Class CE Gate Hyperparameters =====
        # (Retains original logic; no longer used if AsymmetricStateLoss is enabled) 
        self.static_ce_scale = getattr(cfg.train, "static_ce_scale", None)
        self.static_classes_raw = getattr(cfg.train, "static_classes", [])


        print("===== BAYES DEBUG =====")
        print(f"use_bayes_prior = {self.use_bayes_prior}")
        print(f"lambda_bayes_prior   = {self.lambda_bayes}")
        print(f"has_transition_prior = {self.transition_prior is not None}")

        # === DEBUG: Check if minor_loader was passed in. ===
        print(f"\n[TRAINER INIT] minor_loader = {self.minor_loader}")
    # ------------------------------
    # TRAIN LOOP
    # ------------------------------
    def train(self):

        for epoch in range(1, self.epochs + 1):
            self.model.train()

        
            self._reset_minor_iter()

            total_loss = 0.0
            all_preds = []
            all_targets = []  # Stores y_ce (0..K-1), used only for metric computation

            print(f"\n\n=============================================")
            print(f"Epoch [{epoch}/{self.epochs}]")

            for step, batch in enumerate(self.train_loader):
                # -------------------------
                # Compatible with multiple DataLoader output formats:
                #   - (x, y_state)
                #   - (x, y_state, y_trans)
                #   - (x, y_state, y_trans, subj)
                # -------------------------
                if len(batch) == 4:
                    x, y_state, y_trans, subj = batch
                    has_subj = True
                elif len(batch) == 3:
                    x, y_state, y_trans = batch
                    has_subj = False
                else:
                    x, y_state = batch
                    # If no transition labels are provided, fill with zeros
                    # (they will not contribute to the loss or will have zero weight)
                    y_trans = torch.zeros_like(y_state)
                    has_subj = False

                x = x.to(self.device)               # [B, T, C]
                y_state = y_state.to(self.device)   # [B], Original labels: 0 or -1 for invalid/ignored, 1..K for valid classes
                y_trans = y_trans.to(self.device)   # [B], 0/1

                if has_subj:
                    subj = subj.to(self.device)     # [B], Used only to check whether samples belong to the same subject

                # y_state may be one-hot encoded and needs to be converted to class indices
                if y_state.ndim > 1:
                    y_state = y_state.argmax(dim=1)
                y_state = y_state.long()

                # Original labels y_raw: 0 / -1 / 1..K
                y_raw = y_state

                # Valid labels: 1..K
                mask_valid = (y_raw > 0)

                # -------------------------
                # Forward pass: supports dictionary outputs
                # -------------------------
                # Prefer calling the model with return_phys_loss enabled
                try:
                    out = self.model(x, return_phys_loss=True)
                except TypeError:
                    out = self.model(x)

                trans_aux_logits = None

                if isinstance(out, dict):
                    if "state_logits" in out:
                        state_logits = out["state_logits"]            # [B, K]
                        trans_logit = out.get("trans_logit", None)
                        if trans_logit is not None:
                            trans_logit = trans_logit.squeeze(-1)     # [B]
                        trans_aux_logits = None
                    elif "logits" in out:
                        state_logits = out["logits"]                  # [B, K]
                        trans_logit = None
                        trans_aux_logits = out.get("logits_trans", None)
                    else:
                        raise RuntimeError("Unexpected dict keys from model output")

                    phys_loss = out.get("phys_loss", None)
                    if phys_loss is None:
                        phys_loss = 0.0 * state_logits.mean()
                else:
                    state_logits = out
                    trans_logit = None
                    trans_aux_logits = None
                    phys_loss = 0.0 * state_logits.mean()

                # -------------------------
                # state loss
                # -------------------------
                if mask_valid.any():
                    logits_state_valid = state_logits[mask_valid]     # [B_valid, K]

                    # Construct 0-based labels for CE loss and metric computation only at this point
                    y_ce = y_raw.clone()
                    y_ce[mask_valid] = y_ce[mask_valid] - 1          # 1..K → 0..K-1
                    targets_state_valid = y_ce[mask_valid]           # [B_valid], 0..K-1

                    # ==== If AsymmetricStateLoss is enabled, use it directly to compute the loss ====
                    if self.use_asym_penalty and isinstance(self.criterion_state, AsymmetricStateLoss):
                        loss_state = self.criterion_state(
                            logits_state_valid,
                            targets_state_valid,
                        )
                    else:
                        # Original logic: F.cross_entropy with optional static-class scaling
                        weight = None
                        if isinstance(self.criterion_state, nn.CrossEntropyLoss):
                            weight = self.criterion_state.weight

                        ce_all = nn.functional.cross_entropy(
                            logits_state_valid,
                            targets_state_valid,
                            weight=weight,
                            reduction="none",     # [B_valid]
                        )

                        # Optional: apply cross-entropy scaling to specific classes (e.g., static classes)
                        if (
                            self.static_ce_scale is not None
                            and len(self.static_classes_raw) > 0
                            and not self.use_asym_penalty    # Automatically disabled when AsymmetricStateLoss is enabled
                        ):
                            static_mask = torch.zeros_like(
                                targets_state_valid,
                                dtype=torch.bool,
                            )
                            # static_classes_raw contains original labels (1..K); convert to 0-based indices for comparison
                            for c_raw in self.static_classes_raw:
                                c0 = int(c_raw) - 1
                                static_mask |= (targets_state_valid == c0)

                            ce_all = torch.where(
                                static_mask,
                                ce_all * self.static_ce_scale,
                                ce_all,
                            )

                        loss_state = ce_all.mean()
                else:
                    loss_state = 0.0 * state_logits.mean()

                # -------------------------------
                # ==== NEW: Bayesian CE loss ====
                # -------------------------------
                loss_bayes = 0.0 * loss_state
                #if self.use_bayes_prior and self.transition_prior is not None and mask_valid.any():
                #    logits_v = state_logits[mask_valid]                # [N_valid, K]
                #    targets_bayes = (y_raw[mask_valid] - 1).long()     # 0..K-1

                #  Construct the "previous frame" labels by simply following batch order (subject identity is ignored)
                #    prev = torch.roll(targets_bayes, shifts=1, dims=0)
                #    if prev.numel() > 1:
                #        prev[0] = prev[1]

                #    log_post = bayesian_reweight_learned(
                #        logits_v,
                #        prev_labels=prev,
                #        transition_prior=self.transition_prior,
                #    )
                #    loss_bayes = nn.functional.nll_loss(log_post, targets_bayes)

                if self.use_bayes_prior and self.transition_prior is not None and mask_valid.any():
                    logits_v = state_logits[mask_valid]
                    targets_bayes = (y_raw[mask_valid] - 1).long()  # 0..K-1

                    # prev_labels
                    prev = torch.roll(targets_bayes, shifts=1, dims=0)
                    if prev.numel() > 1:
                        prev[0] = prev[1]

                    # >>> Apply Bayes only for prev ∈ {3,4,5} (0-based: 3,4,5)
                    # !! Ablation needed to decide whether to also include transition classes, dynamic classes, or all classes
                    bayes_mask = (prev >= 3) & (prev <= 5) 
                    #bayes_mask = (prev >= 0) & (prev <= 11)

                    if bayes_mask.any():
                        log_post = bayesian_reweight_learned(
                            logits_v[bayes_mask],
                            prev_labels=prev[bayes_mask],
                            transition_prior=self.transition_prior,
                        )
                        loss_bayes = F.nll_loss(log_post, targets_bayes[bayes_mask])
                    else:
                        loss_bayes = 0.0 * logits_v.mean()

                    
                    if epoch == 1 and step < 3:
                        print("[BAYES DEBUG] mask_valid =", mask_valid.sum().item())
                        print("[BAYES DEBUG] loss_bayes =", float(loss_bayes.detach().cpu()))
                        print("[BAYES DEBUG] log_post min/max =",
                                float(log_post.min().detach().cpu()),
                                float(log_post.max().detach().cpu()))

                # -------------------------
                # temporal consistency loss
                # -------------------------
                if self.lambda_temp > 0 and mask_valid.sum() > 1:
                    probs_valid = torch.softmax(state_logits[mask_valid], dim=1)  # [B_valid, K]

                    if has_subj:
                        subj_valid = subj[mask_valid]                # [B_valid]
                        same_subj = (subj_valid[1:] == subj_valid[:-1])  # [B_valid-1]
                    else:
                        same_subj = torch.ones(
                            probs_valid.size(0) - 1,
                            dtype=torch.bool,
                            device=probs_valid.device,
                        )

                    if same_subj.any():
                        diff = probs_valid[1:] - probs_valid[:-1]    # [B_valid-1, K]
                        diff = diff[same_subj]                       # [N_pairs, K]
                        temp_loss = (diff ** 2).mean()
                    else:
                        temp_loss = 0.0 * state_logits.mean()
                else:
                    temp_loss = 0.0 * state_logits.mean()

                # -------------------------
                # transition  loss
                # -------------------------
                loss_trans = 0.0 * state_logits.mean()

                # Case 1: Stage2 binary transition head (trans_logit + y_trans)
                if trans_logit is not None and self.lambda_trans > 0:
                    loss_trans_bce = self.criterion_trans(
                        trans_logit,
                        y_trans.float(),
                    )
                    loss_trans = loss_trans + loss_trans_bce

                # Case 2: CNN_TCN 6-class transition auxiliary head (logits_trans)
                if trans_aux_logits is not None and self.lambda_trans > 0:
                    mask_tail = mask_valid & (y_raw >= 7)
                    if mask_tail.any():
                        y_tail = (y_raw[mask_tail] - 7).long()
                        loss_trans_aux = nn.functional.cross_entropy(
                            trans_aux_logits[mask_tail],   # [N_tail, 6]
                            y_tail,                        # [N_tail], 0..5
                            reduction="mean",
                        )
                        loss_trans = loss_trans + loss_trans_aux

                # ------------------------------------------------------
                # Total loss composition (including the Bayesian term)
                # ------------------------------------------------------
                loss = (
                    loss_state
                    + self.lambda_bayes * loss_bayes
                    + self.lambda_phys * phys_loss
                    + self.lambda_trans * loss_trans
                    + self.lambda_temp * temp_loss
                )

                # == Optionally add an additional cross-entropy loss from a minor-class batch ===
                loss = self._apply_minor_oversampling(loss, step, epoch)

                # -------------------------
                # backward + step
                # -------------------------
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                # Training metrics: evaluate state predictions only on valid samples (internal 0-based labels)
                if mask_valid.any():
                    logits_valid = state_logits[mask_valid]              # [B_valid, K]
                    preds = torch.argmax(logits_valid, dim=1)            # [B_valid], 0..K-1

                    y_ce_metric = (y_raw[mask_valid] - 1)                # 0..K-1

                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(y_ce_metric.cpu().numpy())

                if step % self.log_interval == 0:
                    print(
                        f"Epoch [{epoch}/{self.epochs}], "
                        f"Step [{step}/{len(self.train_loader)}], "
                        f"Loss: {loss.item():.4f} "
                        f"(state={loss_state.item():.4f}, "
                        f"bayes={loss_bayes.item():.4f}, "
                        f"trans={loss_trans.item():.4f}, "
                        f"phys={phys_loss.item():.4f}, "
                        f"temporal={temp_loss.item():.4f})"
                    )

            # -------------------------
            # Train metrics
            # -------------------------
            if len(all_targets) > 0:
                train_f1 = f1_score(all_targets, all_preds, average="macro")
                train_acc = accuracy_score(all_targets, all_preds)
            else:
                train_f1, train_acc = 0.0, 0.0

            avg_train_loss = total_loss / len(self.train_loader)
            print(f"\n------------ Epoch [{epoch}/{self.epochs}] ------------")
            print(
                f"Epoch [{epoch}/{self.epochs}] \n"
                f"Train Loss: {avg_train_loss:.4f} \n"
                f"Train F1 (macro, 0-based): {train_f1:.4f} \n"
                f"Train Accuracy (0-based): {train_acc:.4f} \n"
            )


            print(f"[EPOCH {epoch}] minor batches used = {getattr(self, '_minor_used_count', 0)}")

            wandb.log(
                {
                    "train/loss": avg_train_loss,
                    "train/f1_macro": train_f1,
                    "train/accuracy": train_acc,
                    "train/loss_state": loss_state.item(),
                    "train/loss_bayes": loss_bayes.item(),
                    "train/loss_trans": loss_trans.item(),
                    "train/loss_phys": phys_loss.item(),
                    "train/loss_temp": temp_loss.item(),
                },
                step=epoch,
            )

            # -------------------------
            # Validation
            # -------------------------
            val_metrics = self.evaluator.eval()

            if self.cfg.train_module.scheduler.name == "plateau":
                self.scheduler.step(val_metrics["loss"])

            if self.cfg.train.metric == "f1":
                score = val_metrics["f1"]
                
                if score >= self.best_val_f1:
                    self.best_val_f1 = score
                    self.best_val_acc = val_metrics["accuracy"]
                    self.save_checkpoint_best(epoch)
                    print(f"✅ New best F1: {self.best_val_f1:.4f}")

                    wandb.log({
                        "best_Model/epoch": epoch,
                        "best_Model/val_acc": val_metrics["accuracy"],
                        "best_Model/val_loss": val_metrics["loss"],
                        "best_Model/best_val_F1": val_metrics["f1"],
                        "best_Model/val_Precision": val_metrics["precision"],
                        "best_Model/val_Recall": val_metrics["recall"],
                        "best_Model/ConfusionMatrix": val_metrics["confusion_matrix"],
                    })

                    #if self.cfg.train.use_bayes_prior and epoch == 1:
                    #    class_names = [str(i) for i in range(self.cfg.num_classes)]
                    #    plot_transition_prior(prior_module=self.transition_prior,class_names=class_names)

            else:
                score = val_metrics["accuracy"]
                if score >= self.best_val_acc:
                    self.best_val_f1 = val_metrics["f1"]
                    self.best_val_acc = score
                    self.save_checkpoint_best(epoch)
                    print(f"✅ New best ACC: {self.best_val_acc:.4f}")

                    wandb.log({
                        "best_Model/epoch": epoch,
                        "best_Model/best_val_acc": val_metrics["accuracy"],
                        "best_Model/val_loss": val_metrics["loss"],
                        "best_Model/val_F1": val_metrics["f1"],
                        "best_Model/val_Precision": val_metrics["precision"],
                        "best_Model/val_Recall": val_metrics["recall"],
                        "best_Model/ConfusionMatrix": val_metrics["confusion_matrix"],
                    })
                    if self.cfg.train.use_bayes_prior:
                        class_names = [str(i) for i in range(self.cfg.num_classes)]
                        #plot_transition_prior(prior_module=self.transition_prior,class_names=class_names)



            self.early_stopping.step(score, self.model)

            if self.early_stopping.should_stop:
                print(f"Early stopped at epoch {epoch}")
                self.early_stopping.restore_best(self.model, self.device)
                break

            self.save_checkpoint(epoch)

    # ------------------------------
    # SAVE CHECKPOINT
    # ------------------------------
    def save_checkpoint(self, epoch):
        path = os.path.join(
            self.checkpoint_folder,
            f"checkpoint_epoch{epoch}.pth"
        )

        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            # "best_val_loss": self.best_val_loss,
            "best_val_f1": self.best_val_f1,
        }, path)

        self.checkpoint_queue.append(path)
        if len(self.checkpoint_queue) > self.max_checkpoints:
            old = self.checkpoint_queue.popleft()
            if os.path.exists(old):
                os.remove(old)

    def save_checkpoint_best(self, epoch):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save({
            "epoch": epoch,
            "epochs": self.epochs,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_f1": self.best_val_f1,
            "best_acc": self.best_val_acc,
        }, self.save_path)
        wandb.save(self.save_path)
        print(f"Checkpoint saved at {self.save_path}")


    # =========================================
    # === NEW: helper functions for minor-class oversampling ===
    # =========================================
    def _reset_minor_iter(self):
        """
        Reset the iterator of the minor-class DataLoader at the beginning of each epoch.
        If self.minor_loader is not provided, do nothing.
        """
        if hasattr(self, "minor_loader") and self.minor_loader is not None:
            self._minor_iter = iter(self.minor_loader)
        else:
            self._minor_iter = None

        # === DEBUG: count how many minor-class batches are used in this epoch ===
        self._minor_used_count = 0


    def _apply_minor_oversampling(self, loss, step: int, epoch: int):
        """
        Add an extra cross-entropy loss from a minor-class batch on top of the existing loss.

        Note:
          - Does not modify the original state / bayes / phys / trans / temporal logic
          - Only performs an additional forward pass on one minor-class batch and applies
            cross-entropy loss as strengthened supervision
        """
        # If no minor-class loader is available, return the original loss
        if not hasattr(self, "_minor_iter") or self._minor_iter is None:
            return loss

        # Insert one minor-class batch every N steps (default: 3; can be configured via cfg)
        every_n = getattr(self, "minor_every_n_steps", 3)
        alpha = getattr(self, "minor_alpha", 1.5)

        if every_n <= 0 or (step + 1) % every_n != 0:
            return loss

        # Fetch one minor-class batch
        try:
            batch_minor = next(self._minor_iter)
        except StopIteration:
            # Restart the minor-class loader after one full pass in this epoch
            self._minor_iter = iter(self.minor_loader)
            batch_minor = next(self._minor_iter)

        # Compatible with multiple DataLoader output formats
        if len(batch_minor) == 4:
            x_m, y_state_m, y_trans_m, subj_m = batch_minor
        elif len(batch_minor) == 3:
            x_m, y_state_m, y_trans_m = batch_minor
        else:
            x_m, y_state_m = batch_minor

        x_m = x_m.to(self.device)
        y_state_m = y_state_m.to(self.device)

        # May be one-hot encoded
        if y_state_m.ndim > 1:
            y_state_m = y_state_m.argmax(dim=1)
        y_state_m = y_state_m.long()

        y_raw_m = y_state_m                   # 0 / -1 / 1..K
        mask_valid_m = (y_raw_m > 0)         # Train only on valid labels

        if not mask_valid_m.any():
            return loss  # No valid labels, do not add extra loss

        # Forward pass (physical loss is not needed for minor-class batch,
        # so return_phys_loss is disabled)
        try:
            out_m = self.model(x_m, return_phys_loss=False)
        except TypeError:
            out_m = self.model(x_m)

        if isinstance(out_m, dict):
            if "state_logits" in out_m:
                state_logits_m = out_m["state_logits"]
            elif "logits" in out_m:
                state_logits_m = out_m["logits"]
            else:
                raise RuntimeError("Unexpected dict keys from model output (minor batch)")
        else:
            state_logits_m = out_m

        logits_state_valid_m = state_logits_m[mask_valid_m]

        # 1..K → 0..K-1
        y_ce_m = y_raw_m.clone()
        y_ce_m[mask_valid_m] = y_ce_m[mask_valid_m] - 1
        targets_state_valid_m = y_ce_m[mask_valid_m]

        # Use a "clean" cross-entropy loss for minor-class reinforcement,
        # without asymmetric penalty or static-class scaling
        loss_minor_state = nn.functional.cross_entropy(
            logits_state_valid_m,
            targets_state_valid_m,
            reduction="mean",
        )

        # Add a weighted minor-class loss term to the original loss
        loss = loss + alpha * loss_minor_state

        # Counter: number of minor-class batches used in this epoch
        if hasattr(self, "_minor_used_count"):
            self._minor_used_count += 1

        if epoch == 1 and step < 3:
            print(
                f"[MINOR DEBUG] epoch={epoch}, step={step}, "
                f"minor_batch_valid={mask_valid_m.sum().item()}, "
                f"loss_minor_state={loss_minor_state.item():.4f}, alpha={alpha}"
            )

        return loss


class EarlyStopping:
    """
    Early stopping based on a monitored metric (the higher the better).
    """
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        verbose: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose

        self.best_score = None
        self.counter = 0
        self.should_stop = False
        self.best_state = None

    def step(self, score: float, model: nn.Module):
        """
        score: monitored metric (e.g. val macro F1)
        """
        if self.best_score is None:
            self.best_score = score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return

        if score >= self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if self.verbose:
                print(f"[EarlyStopping] New best score: {score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] No improvement ({self.counter}/{self.patience})")

            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print("[EarlyStopping] Patience exceeded. Stop training.")

    def restore_best(self, model: nn.Module, device):
        if self.best_state is not None:
            model.load_state_dict(
                {k: v.to(device) for k, v in self.best_state.items()}
            )
    
class Trainer_RW:
    def __init__(self,cfg,train_loader,val_loader,model,evaluator,device,max_checkpoints=5):
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.to(device)
        self.evaluator = evaluator
        self.device = device

        # -------------------------
        # Training params
        # -------------------------
        self.epochs = cfg.epochs
        self.log_interval = cfg.log_interval

        # IMPORTANT: reduction="none" for masking
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = build_optimizer(cfg, self.model)
        self.scaler = amp.GradScaler(device=self.device)

        self.best_val_f1 = -1.0


        # -------------------------
        # Checkpoints
        # -------------------------
        self.max_checkpoints = max_checkpoints
        self.checkpoint_queue = deque()
        self.checkpoint_folder = "artifacts/checkpoints"
        os.makedirs(self.checkpoint_folder, exist_ok=True)

        self.save_path = cfg.save.path
        self.check_path = cfg.check_path  
        self.start_epoch = 1  

    # ------------------------------
    # SAVE CHECKPOINT
    # ------------------------------
    def save_checkpoint(self, epoch):
        path = os.path.join(
            self.checkpoint_folder,
            f"checkpoint_epoch{epoch}.pth"
        )

        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            # "best_val_loss": self.best_val_loss,
            "best_val_f1": self.best_val_f1,
        }, path)

        self.checkpoint_queue.append(path)
        if len(self.checkpoint_queue) > self.max_checkpoints:
            old = self.checkpoint_queue.popleft()
            if os.path.exists(old):
                os.remove(old)

    # ------------------------------
    # LOAD CHECKPOINT
    # ------------------------------


    def load_checkpoint(self, path=None):
        path = path or self.check_path
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.start_epoch = checkpoint["epoch"] + 1
            self.best_val_f1 = checkpoint["best_val_f1"]
            print(f"Loaded checkpoint from {path}, starting at epoch {self.start_epoch}")
        else:
            print(f"No checkpoint found at {path}, starting from scratch.")
        return self.start_epoch

    # ------------------------------
    # TRAINING LOOP
    # ------------------------------
    def train(self, resume=False, checkpoint_path=None):
        start_epoch = 1
        if resume and checkpoint_path is not None:
            start_epoch = self.load_checkpoint(checkpoint_path)

        
        #Training
        for epoch in range(start_epoch, self.epochs + 1):
            self.model.train()

            total_loss = 0.0
            all_preds = []
            all_targets = []

            print(f"\n\n=============================================")
            print(f"Epoch [{epoch}/{self.epochs}]")

            for step, (x, y) in enumerate(self.train_loader):
                if isinstance(x, dict):  # multi-position
                    x = {pos: data.to(self.device).float() for pos, data in x.items()}
                else:
                    x = x.to(self.device).float()  

                if self.cfg.dataset.name.lower() == "hapt":
                    mask = y.sum(dim=1) > 0   # mask unlabeled
                    targets = y.argmax(dim=1)
                    targets = targets[mask]
                    with amp.autocast(device_type=self.device.type):
                        logits = self.model(x)
                        loss = self.criterion(logits, targets)
                    if mask.sum() > 0:
                        loss = loss[mask].mean()
                    else:
                        loss = torch.tensor(0.0, device=self.device)
                else:
                    y = y.to(self.device)
                    logits = self.model(x)  
                    loss = self.criterion(logits, y)


                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                    
                    



                # ----------------------------------
                # Accumulate metrics
                # ----------------------------------
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                if self.cfg.dataset.name.lower() == "hapt":
                    preds = preds[mask]
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())

                if step % self.log_interval == 0:
                    print(
                        f"Epoch [{epoch}/{self.epochs}], "
                        f"Step [{step}/{len(self.train_loader)}], "
                        f"Loss: {loss.item():.4f}"
                    )


            # -------------------------
            # Train metrics
            # -------------------------
            avg_train_loss = total_loss / len(self.train_loader)
            train_f1 = f1_score(all_targets, all_preds, average="macro")
            train_acc = accuracy_score(all_targets, all_preds)
            print(f"\n------------ Epoch [{epoch}/{self.epochs}] ------------") 
            print(
                f"Epoch [{epoch}/{self.epochs}] \n"
                f"Train Loss: {avg_train_loss:.4f} \n"
                f"Train F1 (macro): {train_f1:.4f} \n"
                f"Train Accuracy: {train_acc:.4f} \n"
            )

            wandb.log(
                {
                    "train/loss": avg_train_loss,
                    "train/f1_macro": train_f1,
                    "train/accuracy": train_acc,
                },
                step=epoch,
            )

            # -------------------------
            # Validation
            # -------------------------
            val_metrics = self.evaluator.eval()

            wandb.log(
                {
                    "val/loss": val_metrics["loss"],
                    "val/f1_macro": val_metrics["f1"],
                    "val/accuracy": val_metrics["accuracy"],
                },
                step=epoch,
            )

            # -------------------------
            # Best model
            # -------------------------
            if val_metrics["f1"] >= self.best_val_f1:
                self.best_val_f1 = val_metrics["f1"]
                best_path = f"artifacts/models/{self.cfg.dataset.name.lower()}_{self.cfg.model.name}_bestmodel.pth"
                if not os.path.exists(os.path.dirname(best_path)):
                    os.makedirs(os.path.dirname(best_path))
                torch.save(self.model.state_dict(), best_path)
                print(f"New best model saved: {best_path}")
                
                # self.save_checkpoint(epoch)
                # print(f"✅ F1: {self.best_val_f1:.4f}")

                #best_path = (
                #    f"artifacts/models/hapt_{self.cfg.model.name}_best_model.pth"
                #)
                #torch.save(self.model.state_dict(), best_path)
                #print(f"✅ New best model saved: {best_path}")


            # Save checkpoint every epoch #not necessary! because we only care about the best model
            self.save_checkpoint(epoch)
