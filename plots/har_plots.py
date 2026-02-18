import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import zipfile
from human_activity.datasets import get_dataset
from human_activity.models import get_model
import hydra
import torch
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
    

@hydra.main(version_base=None, config_path='config', config_name='default.yaml')
def main(cfg):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset, train_loader, mean, std = get_dataset(cfg, split="train")
    val_dataset, val_loader = get_dataset(cfg, split="val", mean=mean, std=std)
    test_dataset, test_loader = get_dataset(cfg, split="test", mean=mean, std=std)
    POSITIONS = ["chest", "head", "shin", "thigh", "upperarm", "waist", "forearm"]
    ACTIVITIES = ["walking", "running", "sitting", "standing","lying", "climbingup", "climbingdown", "jumping"]

    #--------------------------------
    # Raw Singal Exploration
    #--------------------------------

    # Example for acc_climbingdown_chest.csv
    path = "/home/data/realworld2016_dataset/proband1/data/acc_climbingdown_chest.csv"
    df = pd.read_csv(path)
    print(df.head())
    print(df.shape)

    # -------------------------------
    # Plot raw data samples for one user
    # -------------------------------

    base = Path("/home/data/realworld2016_dataset")
    subject = "proband2"

    files = sorted((base / subject / "data").glob("acc_*_chest.csv"))

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["file"] = f.name
        dfs.append(df)
    print(f"Number of files for {subject}: {len(dfs)}")

    raw = pd.concat(dfs, ignore_index=True)
    print("Combined raw data head:")
    print(raw.head())
    print("Columns:")
    print(raw.columns)
    
    print("dt describe:")
    raw["dt"] = raw.groupby("file")["attr_time"].diff()
    print(raw["dt"].describe())

    raw["activity"] = raw["file"].str.replace("acc_", "").str.replace("_chest.csv", "")
    print("Activity value counts:")
    print(raw["activity"].value_counts())

    # activities = ["sitting", "climbingup", "standing", "walking", "lying", "running", "climbingdown", "jumping"]
    for activity in ACTIVITIES:
        subset = raw[raw["activity"] == activity].iloc[:2000]
        plt.figure(figsize=(12,4))
        plt.plot(subset["attr_x"], label="x")
        plt.plot(subset["attr_y"], label="y")
        plt.plot(subset["attr_z"], label="z")
        plt.title(f"{subject} – {activity}")
        plt.legend()
        plt.show()
        plt.savefig(f"artifacts/plots/HAR/user2/{subject}_{activity}_acc.png", dpi=300)

    # -------------------------------
    # Class distribution
    # -------------------------------

    def check_class_distribution(loader, dataset_name):
        labels = []

        for _, y in loader:
            if dataset_name.lower() == "hapt":
                # one-hot → class index
                y = torch.argmax(y, dim=1)
            labels.extend(y.cpu().tolist())

        return Counter(labels)
    
    train_dist = check_class_distribution(train_loader, cfg.dataset.name)
    print("Train class distribution:")
    for k in sorted(train_dist):
        print(f"Class {k}: {train_dist[k]}")

    val_dist = check_class_distribution(val_loader, cfg.dataset.name)
    print("\nValidation class distribution:")
    for k in sorted(val_dist):
        print(f"Class {k}: {val_dist[k]}")
    
    # -------------------------------
    # Class distribution - Bar Chart
    # -------------------------------

    def plot_and_save_class_distribution(dist, title, save_path):
        classes = sorted(dist.keys())
        print(classes)
        counts = [dist[c] for c in classes]

        
        # Map class indices to activity names
        # class_labels = [ACTIVITIES[c] for c in classes]
        plt.figure(figsize=(8, 4))
        bars = plt.bar(classes, counts)

        plt.xlabel("Class label")
        plt.ylabel("Number of samples")
        plt.title(title)
        plt.xticks(classes, rotation=45)

        # 🔹 Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=9
            )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
    
    
    plot_and_save_class_distribution(
    train_dist,
    "Train Class Distribution",
    "artifacts/plots/HAR/train_class_distribution.png"
    )

    plot_and_save_class_distribution(
        val_dist,
        "Validation Class Distribution",
        "artifacts/plots/HAR/val_class_distribution.png"
    )

    # -------------------------------
    # Repreesentation analysis: single-position linear probing
    # -------------------------------

    def evaluate_single_position(cfg, position, device):
        model = get_model(cfg)   #single-position model
        model.to(device)

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        # ---- training ----
        for epoch in range(10):
            model.train()
            for x_dict, y in train_loader:
                x = x_dict[position].to(device)
                y = y.to(device)

                opt.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                opt.step()

        # ---- validation ----
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x_dict, y in test_loader:
                x = x_dict[position].to(device)
                y = y.to(device)

                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        return correct / total
    
    results = {}
    for position in POSITIONS:
        acc = evaluate_single_position(cfg, position, device)
        results[position] = acc
        print(f"{position:10s}: {acc:.3f}")

    # -------------------------------
    # Single-position linear probing - Bar Chart
    # -------------------------------

    def linearprobing_results_barchart():
        positions = POSITIONS
        # accuracies = [0.885, 0.774, 0.729, 0.726, 0.629, 0.378, 0.330] validation
        accuracies = [0.745, 0.719, 0.787, 0.654, 0.708, 0.833, 0.815]  # test

        # ---- sort by accuracy (descending) ----
        data = sorted(zip(positions, accuracies), key=lambda x: x[1], reverse=True)
        positions, accuracies = zip(*data)

        # Color lower-body vs upper-body sensors
        norm = mcolors.Normalize(vmin=min(accuracies), vmax=max(accuracies))
        cmap = cm.Blues
        colors = [cmap(norm(a)) for a in accuracies]

        plt.figure(figsize=(8,5))
        bars = plt.bar(positions, accuracies, color=colors)
        plt.ylim(0,1)
        plt.ylabel("Test Accuracy")
        plt.title("Single-Position Linear Probing of Shared Encoder")

        # Annotate exact numbers on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.02, f"{acc:.3f}", ha="center")

        save_path = "../human_activity/artifacts/plots/HAR/single_position_accuracy.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
    
    linearprobing_results_barchart()
    
    # -------------------------------
    # t-SNE visualization of embeddings
    # -------------------------------

    def tsne():
        model = get_model(cfg)   # MUST be a single-position model
        model.load_state_dict(torch.load("artifacts/models/har_HARLSTM_bestmodel93.pth", map_location=device))
        model.to(device)
        model.eval()

        # Optional: add helper to extract fused embeddings
        def extract_fused_embedding(model, x_dict):
            embeddings = []
            for pos in POSITIONS:
                emb = model.encoder(x_dict[pos])
                embeddings.append(emb)
            fused = torch.cat(embeddings, dim=1)  # [B, 448]
            return fused

        # --- 2. Collect embeddings & labels from test set ---
        X_list, y_list = [], []

        with torch.no_grad():
            fused_embeddings = []
            labels_list = []

            with torch.no_grad():
                for x_dict, y in test_loader:
                    # Move all positions to device
                    x_dict = {pos: x_dict[pos].to(device) for pos in POSITIONS}

                    # Compute embeddings efficiently
                    embs = torch.cat([model.encoder(x_dict[pos]) for pos in POSITIONS], dim=1)  # [B, hidden*pos]
                    fused_embeddings.append(embs.cpu())
                    labels_list.append(y.cpu())

        X = torch.cat(fused_embeddings, dim=0).numpy()
        y = torch.cat(labels_list, dim=0).numpy()

        if len(X) > 2000:
            idx = np.random.choice(len(X), 2000, replace=False)
            X, y = X[idx], y[idx]
        # --- 3. Reduce dimensionality for t-SNE ---
        # Step 1: PCA
        X_pca = PCA(n_components=50, random_state=42).fit_transform(X)

        # Step 2: t-SNE
        X_tsne = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate=200,
            # n_iter=1000,
            random_state=42
        ).fit_transform(X_pca)

        # --- 4. Plot ---
        plt.figure(figsize=(8,6))
        scatter = plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y, cmap="tab10", s=8, alpha=0.7)
        handles = []
        for i, activity in enumerate(ACTIVITIES):
            handles.append(
                plt.Line2D(
                    [], [], marker="o", linestyle="",
                    color=scatter.cmap(scatter.norm(i)),
                    label=activity
                )
            )

        plt.legend(handles=handles, title="Activity", fontsize=9)
        plt.title("t-SNE - HAR")
        plt.xlabel("t-SNE Dim 1")
        plt.ylabel("t-SNE Dim 2")
        plt.tight_layout()
        plt.show()
        plt.savefig("../human_activity/artifacts/plots/HAR/tsne.png", dpi=300)
        plt.close()

    tsne()

if __name__ == "__main__":
    main()
