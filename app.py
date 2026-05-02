import streamlit as st
import os

from main import load_images
from features import extract_features
from similarity import find_duplicates
from blur import is_blurry
from cluster import cluster_images
from visualize import plot_similarity_histogram, plot_pca_clusters, plot_blur_scores
from organize import create_output_dirs, save_duplicates, save_blurry, save_clusters

IMAGE_FOLDER = "images"

st.title("📸 Smart Photo Organizer")
st.write("Upload or analyze a folder of images to detect duplicates, blurry photos, and clusters.")
st.success("Click the button to analyze your photo collection and clean it up automatically.")

# Button
if st.button("Run Analysis"):

    imgs, names = load_images(IMAGE_FOLDER)

    st.subheader("Loaded Images")
    st.write(f"{len(imgs)} images found")

    # Features
    features = extract_features(imgs)

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

    labels = cluster_images(features, k=7)

    try:
        plot_pca_clusters(features, labels)
    except Exception as e:
        st.warning(f"PCA plot failed: {e}")

    clusters = {}
    for i, label in enumerate(labels):
        clusters.setdefault(label, []).append(i)

    for label, indices in clusters.items():
        st.write(f"Group {label} (Similar Images)")

        cols = st.columns(4)
        for idx, i in enumerate(indices):
            cols[idx % 4].image(imgs[i], caption=names[i])
    
    # ORGANIZATION
    st.subheader("Organize Photos")

    if st.button("Save Organized Photos"):
        create_output_dirs()
        save_duplicates(duplicates, names)
        save_blurry(imgs, names, is_blurry)
        save_clusters(labels, names)

        st.success("Photos organized! Check the output folder.")

    # Show visualizations
    st.subheader("Visualizations")

    if os.path.exists("figures/similarity_histogram.png"):
        st.image("figures/similarity_histogram.png")

    if os.path.exists("figures/pca_clusters.png"):
        st.image("figures/pca_clusters.png")

    if os.path.exists("figures/blur_scores.png"):
        st.image("figures/blur_scores.png")