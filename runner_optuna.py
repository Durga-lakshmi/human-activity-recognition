import copy
import torch
import optuna
from hydra import initialize, compose
from omegaconf import OmegaConf

from run_optuna_train_val import run_train_val


# ====== 1. Load base config ======
def load_base_cfg():
    """
    Compose the full Hydra config (including dataset / model / train sub-configs).
    This is equivalent to the original
    @hydra.main(config_path='config', config_name='default.yaml')
    """
    # Note: config_path is relative to the current working directory.
    # Make sure you run this script from the project root: human_activity_Wu/
    with initialize(version_base=None, config_path="config"):
        # config_name is usually the filename without .yaml
        cfg = compose(config_name="default")
    return cfg



# ====== 2. Define search space + objective ======
def objective(trial: optuna.trial.Trial) -> float: #
    base_cfg = load_base_cfg()
    cfg = copy.deepcopy(base_cfg)

    # ---------- ① Learning rate & weight decay ----------
    # In your original wandb.config, both cfg.lr and
    # cfg.train_module.optimizer.name exist.
    # We update both branches to ensure build_optimizer
    # can always access the learning rate.

    # Top-level lr (if exists)
    if hasattr(cfg, "lr"):
        cfg.lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)

    # train_module.optimizer.lr (if exists)
    if hasattr(cfg, "train_module") and hasattr(cfg.train_module, "optimizer"):
        cfg.train_module.optimizer.lr = trial.suggest_float(
            "optim_lr", 1e-4, 3e-3, log=True
        )
        # Alternatively, force it to be identical to top-level lr:
        # cfg.train_module.optimizer.lr = cfg.lr

        # Weight decay follows the same logic
        cfg.train_module.optimizer.weight_decay = trial.suggest_float(
            "weight_decay", 1e-6, 1e-3, log=True
        )
    else:
        # Fallback if train_module.optimizer does not exist
        cfg.lr = trial.suggest_float("lr_simple", 1e-4, 3e-3, log=True)
        cfg.weight_decay = trial.suggest_float(
            "weight_decay_simple", 1e-6, 1e-3, log=True
        )

    # ---------- ② CNN_TCN architectural hyperparameters ----------
    if hasattr(cfg, "model") and getattr(cfg.model, "name", "") == "CNN_TCN":
        # 1) CNN channels: controls local feature capacity
        cfg.model.cnn_channels = trial.suggest_categorical(
            "cnn_channels", [32, 64, 96]
        )

        # 2) TCN channels: controls temporal feature capacity
        cfg.model.tcn_channels = trial.suggest_categorical(
            "tcn_channels", [128, 256, 384]
        )

        # 3) TCN dropout: applied only to TCN layers
        cfg.model.tcn_dropout = trial.suggest_float(
            "tcn_dropout", 0.0, 0.5
        )

        # 4) Dilation patterns: choose from common structures
        # First let Optuna select a pattern name (string)
        pattern_name = trial.suggest_categorical(
            "tcn_dilations_pattern",
            ["1_2_4", "1_2_4_8", "2_4", "2_4_8"],
        )

        # Then map it to the actual dilation list
        pattern_map = {
            "1_2_4":    [1, 2, 4],
            "1_2_4_8":  [1, 2, 4, 8],
            "2_4":      [2, 4],
            "2_4_8":    [2, 4, 8],
        }

        cfg.model.tcn_dilations = pattern_map[pattern_name]

    # ---------- ③ Loss weight coefficients ----------
    # These fields are expected in Trainer:
    # self.lambda_trans / self.lambda_phys / self.lambda_temp / self.lambda_bayes
    # Comment out any weights you do not want Optuna to tune.

    if hasattr(cfg, "train") and hasattr(cfg.train, "lambda_bayes"):
        cfg.train.lambda_bayes = trial.suggest_float(
            "lambda_bayes", 1e-4, 0.5, log=True
        )

    if hasattr(cfg, "train") and hasattr(cfg.train, "lambda_phys"):
        cfg.train.lambda_phys = trial.suggest_float(
            "lambda_phys", 1e-4, 1.0, log=True
        )

    if hasattr(cfg, "train") and hasattr(cfg.train, "lambda_trans"):
        cfg.train.lambda_trans = trial.suggest_float(
            "lambda_trans", 1e-4, 1.0, log=True
        )

    if hasattr(cfg, "train") and hasattr(cfg.train, "lambda_temp"):
        cfg.train.lambda_temp = trial.suggest_float(
            "lambda_temp", 1e-4, 1.0, log=True
        )

    # ---------- ④ Fix number of epochs, rely on EarlyStopping ----------
    cfg.epochs = getattr(cfg, "epochs", 60)

    # Fix random seed to reduce variance across trials
    seed = 42 + trial.number
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ---------- ⑤ Call the unified training & validation entry ----------
    val_macro_f1 = run_train_val(cfg, trial=trial)

    # Optuna maximizes this metric
    return float(val_macro_f1)


# ====== 3. Launch study ======
def main():
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=0),
    )

    # Recommended: start with 20–30 trials
    study.optimize(objective, n_trials=30)

    print("===== Best Trial =====")
    print("Value (val F1 macro) =", study.best_value)
    print("Best Params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    OmegaConf.save(
        OmegaConf.create(study.best_params),
        "optuna/best_optuna_params.yaml"
    )


if __name__ == "__main__":
    main()
