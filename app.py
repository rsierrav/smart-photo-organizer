import streamlit as st
import os
import shutil
import numpy as np

from main import load_images
from features import extract_features, generate_caption, summarize_captions
from similarity import find_duplicates
from blur import is_blurry
from cluster import cluster_images, find_best_k
from visualize import plot_similarity_histogram, plot_pca_clusters, plot_blur_scores
from organize import create_output_dirs, zip_output

IMAGE_FOLDER = "images"
UPLOAD_FOLDER = "uploaded_images"


def save_uploaded_files(uploaded_files, folder=UPLOAD_FOLDER):
    if os.path.exists(folder):
        shutil.rmtree(folder)

    os.makedirs(folder, exist_ok=True)

    saved_names = set()

    for uploaded_file in uploaded_files:
        original_name = os.path.basename(uploaded_file.name)
        name, ext = os.path.splitext(original_name)

        safe_name = original_name
        counter = 1

        # Avoid overwriting files with the same name
        while safe_name in saved_names or os.path.exists(os.path.join(folder, safe_name)):
            safe_name = f"{name}_{counter}{ext}"
            counter += 1

        saved_names.add(safe_name)

        save_path = os.path.join(folder, safe_name)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    return folder

@st.cache_data
def get_optimal_k(features_array):
    return find_best_k(features_array, k_range=(5, 10))

st.title("📸 Smart Photo Organizer")
st.write("Upload or analyze a folder of images to detect duplicates, blurry photos, and clusters.")
st.success("Click the button to analyze your photo collection and clean it up automatically.")

# Image source selection
source_option = st.radio(
    "Choose image source",
    ["Use images folder", "Upload my own images"]
)

uploaded_files = []

if source_option == "Upload my own images":
    uploaded_files = st.file_uploader(
        "Upload images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True
    )

# Button
if st.button("Run Analysis"):

    if source_option == "Upload my own images":
        if not uploaded_files:
            st.warning("Please wait for images to upload before running the analysis.")
            st.stop()

        source_folder = save_uploaded_files(uploaded_files)
    else:
        source_folder = IMAGE_FOLDER

    imgs, names = load_images(source_folder)

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
    # Prefer non-blurry images for representative selection
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

        # Check if ALL images in the cluster are blurry
        cluster_indices = clusters.get(label, [])
        all_blurry = all(is_blurry(imgs[ci])[0] for ci in cluster_indices)

        if all_blurry:
            cluster_names[label] = "Low Detail / Possibly Blurry"
        else:
            summary = summarize_captions(captions)
            cluster_names[label] = summary

    for label, indices in clusters.items():
        predicted_label = cluster_names.get(label, "Miscellaneous")
        st.write(f"**Group {label}** – Predicted Label: {predicted_label})")

        cols = st.columns(4)
        for idx, i in enumerate(indices):
            cols[idx % 4].image(imgs[i], caption=names[i])
    
    # ORGANIZATION, Automatic save after analysis
    st.subheader("Organize Photos")

    create_output_dirs()
    from organize import save_organized
    blurry_flags, cluster_blurry_flags = save_organized(
        duplicates,
        names,
        imgs,
        is_blurry,
        labels,
        cluster_names,
        source_folder=source_folder
    )

    true_blur_count = sum(blurry_flags)
    cluster_blur_count = max(0, sum(cluster_blurry_flags) - true_blur_count)

    st.success("Photos organized! You can now download the ZIP.")
    st.info(f"Summary: {true_blur_count} images removed due to blur detection.")
    st.info(f"Summary: {cluster_blur_count} images removed due to blurry cluster grouping.")

    # Zip for download
    st.subheader("Download Organized Photos")

    zip_path = zip_output()

    with open(zip_path, "rb") as f:
        st.download_button(
            label="Download ZIP",
            data=f,
            file_name="organized_photos.zip",
            mime="application/zip"
        )

    # Show visualizations
    st.subheader("Visualizations")

    if os.path.exists("figures/similarity_histogram.png"):
        st.image("figures/similarity_histogram.png")

    if os.path.exists("figures/pca_clusters.png"):
        st.image("figures/pca_clusters.png")

    if os.path.exists("figures/blur_scores.png"):
        st.image("figures/blur_scores.png")