import optuna
import torch
import torch.nn as nn
import hydra
from datasets import get_dataset
from datasets.hapt.dataset import HAPT, compute_train_stats
from models.util import get_model
import copy


def make_objective(cfg, device, mean, std):
    def objective(trial):
        cfg_trial = copy.deepcopy(cfg)

        cfg_trial.model.hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

        model = get_model(cfg_trial).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        criterion = nn.CrossEntropyLoss()

        # ⬇⬇ THIS WAS YOUR BUG ⬇⬇
        train_ds, train_loader, mean, std = get_dataset(cfg_trial, "train")
        val_ds, val_loader = get_dataset(cfg_trial, "val", mean, std)

        # short training
        for _ in range(5):
            model.train()
            for x, y in train_loader:
                x = x.to(device)
                y = y.argmax(dim=1).to(device)

                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()

        # evaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.argmax(dim=1).to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        return correct / total


    return objective

@hydra.main(version_base=None, config_path="config", config_name="default.yaml")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean, std = compute_train_stats(cfg)

    objective = make_objective(cfg, device, mean, std)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("Best params:", study.best_params)
    print("Best val accuracy:", study.best_value)


if __name__ == "__main__":
    main()
