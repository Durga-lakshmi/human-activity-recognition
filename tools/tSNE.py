from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def tsne_plot(feats, labels, num_classes=12):
    # 1. standardize features
    feats = StandardScaler().fit_transform(feats)

    # 2. t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=300,
        n_iter=1000,
        random_state=42,
        init="pca"
    )
    z = tsne.fit_transform(feats)

    # 3. plot figure
    plt.figure(figsize=(7, 6))
    for c in range(num_classes):
        idx = labels == c
        plt.scatter(z[idx, 0], z[idx, 1], s=8, label=str(c), alpha=0.7)

    plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1))
    plt.title("t-SNE of Validation Embeddings")
    plt.tight_layout()
    plt.show()
