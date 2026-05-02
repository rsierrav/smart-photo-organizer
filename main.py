import cv2
import os
from collections import Counter
from features import extract_features, categories
from similarity import find_duplicates
from blur import is_blurry
from cluster import cluster_images, find_best_k
from visualize import plot_similarity_histogram, plot_pca_clusters, plot_blur_scores
from organize import create_output_dirs, save_duplicates, save_blurry, save_clusters

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


def label_clusters_by_prediction(cluster_labels, predictions, categories):
    cluster_names = {}

    for cluster_id in set(cluster_labels):
        indices = [i for i, l in enumerate(cluster_labels) if l == cluster_id]
        cluster_preds = [predictions[i] for i in indices]
        
        # Get the most common prediction
        most_common = Counter(cluster_preds).most_common(1)[0][0]
        cluster_names[cluster_id] = categories[most_common]

    return cluster_names


if __name__ == "__main__":
    imgs, names = load_images("images")

    print(f"Loaded {len(imgs)} images")

    # STEP 1: Feature extraction
    features, predictions = extract_features(imgs)
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
    best_k = find_best_k(features, k_range=(5, 10))
    labels = cluster_images(features, k=best_k)
    
    # Label clusters by most common ImageNet prediction
    cluster_names = label_clusters_by_prediction(labels, predictions, categories)

    clusters = {}
    for i, label in enumerate(labels):
        clusters.setdefault(label, []).append(names[i])

    for label, cluster_imgs in clusters.items():
        imagenet_label = cluster_names[label]
        print(f"\nGroup {label} – Similar Images (Predicted: {imagenet_label}):")
        for img in cluster_imgs:
            print(f"  {img}")

    # VISUALIZATIONS
    plot_similarity_histogram(features)
    plot_pca_clusters(features, labels)
    plot_blur_scores(imgs, names, is_blurry)

    # ORGANIZATION
    print("\nOrganizing files into output folder...")

    create_output_dirs()
    save_duplicates(duplicates, names, imgs, is_blurry)
    save_blurry(imgs, names, is_blurry)
    save_clusters(labels, names)

    print("Done! Check the 'output' folder.")