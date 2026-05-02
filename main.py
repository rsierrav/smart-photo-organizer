import cv2
import os
from collections import Counter
from features import extract_features, categories
from similarity import find_duplicates
from blur import is_blurry
from cluster import cluster_images, find_best_k
from features import get_cluster_representatives, generate_caption, summarize_captions
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
        
        # Get the most common CLIP prediction
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
    
    # Label clusters using BLIP captions on representatives
    reps = get_cluster_representatives(features, labels, top_k=3)

    cluster_names = {}
    clusters = {}
    for i, label in enumerate(labels):
        clusters.setdefault(label, []).append(names[i])

    for label, indices in reps.items():
        captions = []
        for idx in indices:
            img = imgs[idx]
            try:
                cap = generate_caption(img)
            except Exception:
                cap = "image"
            captions.append(cap)

        summary = summarize_captions(captions)
        cluster_names[label] = summary

    for label, cluster_imgs in clusters.items():
        predicted_label = cluster_names.get(label, "Miscellaneous")
        print(f"\nGroup {label} – Predicted Label: {predicted_label}")
        for img in cluster_imgs:
            print(f"  {img}")

    # VISUALIZATIONS
    plot_similarity_histogram(features)
    plot_pca_clusters(features, labels)
    plot_blur_scores(imgs, names, is_blurry)

    # ORGANIZATION
    print("\nOrganizing files into output folder...")

    # Create organized output
    create_output_dirs()
    from organize import save_organized
    save_organized(duplicates, names, imgs, is_blurry, labels, cluster_names)

    print("Done! Check the 'output' folder.")