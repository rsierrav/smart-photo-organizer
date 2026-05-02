import os
from tkinter.font import names
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


def _ensure_figures_dir():
    os.makedirs("figures", exist_ok=True)


# Similarity histogram
def plot_similarity_histogram(features, out_path="figures/similarity_histogram.png"):
    _ensure_figures_dir()

    if features is None or len(features) < 2:
        plt.figure()
        plt.text(0.5, 0.5, "Not enough data to plot histogram", ha='center', va='center')
        plt.axis('off')
        plt.savefig(out_path)
        plt.close()
        return

    sims = cosine_similarity(features)
    triu = sims[np.triu_indices_from(sims, k=1)]

    plt.figure()
    plt.hist(triu, bins=30, color='C0', edgecolor='k')
    plt.title('Pairwise similarity histogram')
    plt.xlabel('Cosine similarity')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# PCA Cluster Plot
def plot_pca_clusters(features, labels, out_path="figures/pca_clusters.png"):
    _ensure_figures_dir()

    if features is None or len(features) < 2:
        plt.figure()
        plt.text(0.5, 0.5, "Not enough data to plot clusters", ha='center', va='center')
        plt.axis('off')
        plt.savefig(out_path)
        plt.close()
        return

    pca = PCA(n_components=2)
    pts = pca.fit_transform(features)

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(pts[:, 0], pts[:, 1], c=labels, cmap='tab10', s=30)
    plt.title('PCA of image features (clusters)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# Blur Score Graph
def plot_blur_scores(images, names, is_blurry_fn, out_path="figures/blur_scores.png"):
    _ensure_figures_dir()

    scores = []
    labels = []

    for img, name in zip(images, names):
        try:
            # Check if valid numpy image
            if img is None or not isinstance(img, np.ndarray):
                print(f"Skipping invalid image: {name}")
                continue

            if img.size == 0:
                print(f"Skipping empty image: {name}")
                continue

            _, score = is_blurry_fn(img)

            scores.append(score)
            labels.append(name)

        except Exception as e:
            print(f"Skipping {name}: {e}")
            continue

    if len(scores) == 0:
        plt.figure()
        plt.text(0.5, 0.5, "No images to score", ha='center', va='center')
        plt.axis('off')
        plt.savefig(out_path)
        plt.close()
        return

    order = np.argsort(scores)
    sorted_scores = np.array(scores)[order]
    sorted_labels = np.array(labels)[order]

    plt.figure(figsize=(8, max(2, len(scores) * 0.25)))
    plt.barh(sorted_labels, sorted_scores, color='C1')
    plt.title('Blur scores (lower = blurrier)')
    plt.xlabel('Laplacian variance')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()