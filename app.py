import streamlit as st
import os
from collections import Counter

from main import load_images
from features import extract_features, categories, get_cluster_representatives, generate_caption, summarize_captions
from similarity import find_duplicates
from blur import is_blurry
from cluster import cluster_images, find_best_k
from visualize import plot_similarity_histogram, plot_pca_clusters, plot_blur_scores
from organize import create_output_dirs, save_duplicates, save_blurry, save_clusters, zip_output

IMAGE_FOLDER = "images"

@st.cache_data
def get_optimal_k(features_array):
    return find_best_k(features_array, k_range=(5, 10))

def label_clusters_by_prediction(cluster_labels, predictions, categories):
    cluster_names = {}
    for cluster_id in set(cluster_labels):
        indices = [i for i, l in enumerate(cluster_labels) if l == cluster_id]
        cluster_preds = [predictions[i] for i in indices]
        most_common = Counter(cluster_preds).most_common(1)[0][0]
        cluster_names[cluster_id] = categories[most_common]
    return cluster_names

st.title("📸 Smart Photo Organizer")
st.write("Upload or analyze a folder of images to detect duplicates, blurry photos, and clusters.")
st.success("Click the button to analyze your photo collection and clean it up automatically.")

# Button
if st.button("Run Analysis"):

    imgs, names = load_images(IMAGE_FOLDER)

    st.subheader("Loaded Images")
    st.write(f"{len(imgs)} images found")

    # Features (extract both features and predictions)
    features, predictions = extract_features(imgs)

    # Visualizations
    try:
        plot_similarity_histogram(features)
    except Exception as e:
        st.warning(f"Similarity histogram failed: {e}")

    # DUPLICATES
    st.subheader("Duplicates")

    duplicates = find_duplicates(features)

    if duplicates:
        for i, j, score in duplicates:
            st.write(f"Similarity: {score:.2f}")

            col1, col2 = st.columns(2)
            col1.image(imgs[i], caption=names[i])
            col2.image(imgs[j], caption=names[j])
    else:
        st.write("No duplicates found")

    # BLURRY
    st.subheader("Blurry Images")

    blurry_found = False
    for i, img in enumerate(imgs):
        blurry, score = is_blurry(img)
        if blurry:
            st.image(img, caption=f"{names[i]} (Score: {score:.2f})")
            blurry_found = True

    if not blurry_found:
        st.write("No blurry images detected")

    try:
        plot_blur_scores(imgs, names, is_blurry)
    except Exception as e:
        st.warning(f"Blur graph failed: {e}")

    # CLUSTERS
    st.subheader("Clusters")

    # Find optimal K automatically
    with st.spinner("Finding optimal number of clusters..."):
        best_k = get_optimal_k(features)

    st.success(f"Optimal cluster count: **{best_k}**")

    labels = cluster_images(features, k=best_k)

    try:
        plot_pca_clusters(features, labels)
    except Exception as e:
        st.warning(f"PCA plot failed: {e}")

    # Label clusters using BLIP captions on representatives
    reps = get_cluster_representatives(features, labels, top_k=3)

    cluster_names = {}
    clusters = {}
    for i, label in enumerate(labels):
        clusters.setdefault(label, []).append(i)

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

    for label, indices in clusters.items():
        predicted_label = cluster_names.get(label, "Miscellaneous")
        st.write(f"**Group {label}** – Predicted Label: {predicted_label})")

        cols = st.columns(4)
        for idx, i in enumerate(indices):
            cols[idx % 4].image(imgs[i], caption=names[i])
    
    # ORGANIZATION
    st.subheader("Organize Photos")

    if st.button("Save Organized Photos"):
        create_output_dirs()
        from organize import save_organized
        save_organized(duplicates, names, imgs, is_blurry, labels, cluster_names)

        st.success("Photos organized! Check the output folder.")

    # Zip for download
    st.subheader("Download Organized Photos")

    if st.button("Download ZIP"):
        zip_path = zip_output()

        with open(zip_path, "rb") as f:
            st.download_button(
                "Download Organized Photos",
                f,
                file_name="organized_photos.zip"
            )

    # Show visualizations
    st.subheader("Visualizations")

    if os.path.exists("figures/similarity_histogram.png"):
        st.image("figures/similarity_histogram.png")

    if os.path.exists("figures/pca_clusters.png"):
        st.image("figures/pca_clusters.png")

    if os.path.exists("figures/blur_scores.png"):
        st.image("figures/blur_scores.png")