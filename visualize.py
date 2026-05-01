import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# Similarity histogram
def plot_similarity_histogram(features):
    sim_matrix = cosine_similarity(features)

    # get upper triangle
    sims = []
    for i in range(len(sim_matrix)):
        for j in range(i + 1, len(sim_matrix)):
            sims.append(sim_matrix[i][j])

    plt.figure()
    plt.hist(sims, bins=20)
    plt.title("Similarity Distribution")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")

    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/similarity_histogram.png")
    plt.close()

# PCA Cluster Plot
from sklearn.decomposition import PCA

def plot_pca_clusters(features, labels):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features)

    plt.figure()

    for label in set(labels):
        indices = [i for i in range(len(labels)) if labels[i] == label]
        plt.scatter(
            reduced[indices, 0],
            reduced[indices, 1],
            label=f"Cluster {label}"
        )

    plt.title("PCA Cluster Visualization")
    plt.legend()

    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/pca_clusters.png")
    plt.close()

# Blur Score Graph
def plot_blur_scores(imgs, names, is_blurry):
    scores = []

    for img in imgs:
        _, score = is_blurry(img)
        scores.append(score)

    plt.figure()
    plt.bar(range(len(scores)), scores)

    plt.title("Blur Scores")
    plt.xlabel("Image Index")
    plt.ylabel("Laplacian Variance")

    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/blur_scores.png")
    plt.close()