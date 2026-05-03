import cv2
import os
from collections import Counter
from features import extract_features, categories, generate_caption, summarize_captions
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
    # Prefer non-blurry images for representative selection
    import numpy as np
    reps = {}
    for cluster_id in set(labels):
        cluster_indices = [i for i, l in enumerate(labels) if l == cluster_id]
        non_blurry_indices = [i for i in cluster_indices if not is_blurry(imgs[i])[0]]
        
        # use non-blurry images if available, else fall back to all
        representative_pool = non_blurry_indices if non_blurry_indices else cluster_indices
        
        # select top_k closest to centroid
        cluster_feats = features[representative_pool]
        centroid = cluster_feats.mean(axis=0)
        dists = np.linalg.norm(cluster_feats - centroid, axis=1)
        top_indices = np.argsort(dists)[:min(3, len(dists))]
        reps[cluster_id] = [representative_pool[j] for j in top_indices]

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

        # Check if ALL images in the cluster are blurry
        cluster_indices = [i for i, l in enumerate(labels) if l == label]
        all_blurry = all(is_blurry(imgs[ci])[0] for ci in cluster_indices)

        if all_blurry:
            cluster_names[label] = "Blurry Images"
        else:
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
    blurry_moved = save_organized(duplicates, names, imgs, is_blurry, labels, cluster_names)

    print(f"Done! Check the 'output' folder.")
    print(f"Summary: {blurry_moved} blurry images moved to trash.")