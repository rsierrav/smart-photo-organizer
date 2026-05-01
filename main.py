import cv2
import os
from features import extract_features
from similarity import find_duplicates
from blur import is_blurry
from cluster import cluster_images
from visualize import plot_similarity_histogram, plot_pca_clusters, plot_blur_scores

def load_images(folder):
    images = []
    filenames = []

    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = cv2.imread(path)

        if img is not None:
            img = cv2.resize(img, (224, 224))
            images.append(img)
            filenames.append(file)

    return images, filenames


if __name__ == "__main__":
    imgs, names = load_images("images")

    print(f"Loaded {len(imgs)} images")

    # STEP 1: Feature extraction
    features = extract_features(imgs)
    print("Features extracted")

    # STEP 2: Duplicate detection
    duplicates = find_duplicates(features)

    print("\n=== DUPLICATES ===")
    for i, j, score in duplicates:
        print(f"{names[i]} <--> {names[j]} | similarity: {score:.2f}")

    # STEP 3: Blur detection
    print("\n=== BLURRY IMAGES ===")
    for i, img in enumerate(imgs):
        blurry, score = is_blurry(img)
        if blurry:
            print(f"{names[i]} is blurry (score: {score:.2f})")

    # STEP 4: Image clustering
    print("\n=== CLUSTERS ===")
    labels = cluster_images(features, k=7)

    clusters = {}
    for i, label in enumerate(labels):
        clusters.setdefault(label, []).append(names[i])

    for label, imgs in clusters.items():
        print(f"\nCluster {label}:")
        for img in imgs:
            print(f"  {img}")

    # VISUALIZATIONS
    plot_similarity_histogram(features)
    plot_pca_clusters(features, labels)
    plot_blur_scores(imgs, names, is_blurry)