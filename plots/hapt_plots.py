from datasets.hapt.dataset import HAPT
import hydra
import numpy as np

def coarse_class_stats(dataset):
    labels = dataset.y.numpy()

    static = np.isin(labels, [1, 2, 3])
    dynamic = np.isin(labels, [4, 5, 6])
    transition = np.isin(labels, [7, 8, 9, 10, 11, 12])
    unlabeled = labels == 0

    total = len(labels)

    print("\n=== Coarse class breakdown ===")
    print(f"Static     : {static.sum():6d} ({static.mean()*100:5.2f}%)")
    print(f"Dynamic    : {dynamic.sum():6d} ({dynamic.mean()*100:5.2f}%)")
    print(f"Transition : {transition.sum():6d} ({transition.mean()*100:5.2f}%)")
    print(f"Unlabeled  : {unlabeled.sum():6d} ({unlabeled.mean()*100:5.2f}%)")

def window_energy(X):
    # X: (T, C)
    return np.mean(X**2)

def window_variance(X):
    return np.mean(np.var(X, axis=0))

def energy_variance_by_class(dataset):
    X = dataset.X.numpy()
    y = dataset.y.numpy()

    stats = {}

    for cls in range(1, 13):
        idx = np.where(y == cls)[0]
        if len(idx) == 0:
            continue

        E = [window_energy(X[i]) for i in idx]
        V = [window_variance(X[i]) for i in idx]

        stats[cls] = {
            "E_mean": np.mean(E),
            "E_std":  np.std(E),
            "V_mean": np.mean(V),
            "V_std":  np.std(V),
            "n": len(idx)
        }

    print("\n=== Energy / Variance by class ===")
    for cls, s in stats.items():
        print(
            f"class {cls:02d} | "
            f"E={s['E_mean']:.4f}±{s['E_std']:.4f} | "
            f"V={s['V_mean']:.4f}±{s['V_std']:.4f} | "
            f"n={s['n']}"
        )

    return stats
def transition_purity(dataset):
    X = dataset.X.numpy()
    y = dataset.y.numpy()
    ws = X.shape[1]

    purity = []

    for i in range(len(y)):
        if y[i] < 7:
            continue

        # Recover original per-sample labels if you stored them
        # If not: approximate purity by variance (proxy)
        V = window_variance(X[i])
        purity.append(V)

    purity = np.array(purity)

    print("\n=== Transition window variance stats ===")
    print(f"Mean V : {purity.mean():.4f}")
    print(f"Std  V : {purity.std():.4f}")
    print(f"Min  V : {purity.min():.4f}")
    print(f"Max  V : {purity.max():.4f}")

    return purity

def acc_gyro_energy(dataset):
    X = dataset.X.numpy()
    acc = X[:, :, :3]
    gyro = X[:, :, 3:]

    E_acc = np.mean(acc**2)
    E_gyro = np.mean(gyro**2)

    print("\n=== Sensor energy comparison ===")
    print(f"ACC  energy: {E_acc:.4f}")
    print(f"GYRO energy: {E_gyro:.4f}")

def per_subject_energy(dataset, users):
    print("\n=== Per-subject mean energy ===")
    for u in users:
        X, _ = dataset._load_user(u)
        E = np.mean(X**2)
        print(f"user {u:02d}: E={E:.4f}")

@hydra.main(version_base=None, config_path='config', config_name='default.yaml')
def main(cfg):

    train_ds = HAPT(
        root=cfg.dataset.path,
        users=cfg.dataset.train_users,
        window_size=cfg.sliding_window.window_size,
        window_shift=cfg.sliding_window.window_shift,
    )

    coarse_class_stats(train_ds)
    energy_variance_by_class(train_ds)
    transition_purity(train_ds)
    acc_gyro_energy(train_ds)
    per_subject_energy(train_ds, cfg.dataset.train_users)


if __name__ == "__main__":
    main()
