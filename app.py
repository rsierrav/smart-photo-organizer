import streamlit as st
from main import load_images
from features import extract_features
from similarity import find_duplicates
from blur import is_blurry
from cluster import cluster_images

st.title("📸 Smart Photo Organizer")

st.write("Upload or analyze a folder of images to detect duplicates, blurry photos, and clusters.")

st.success("Click the button to analyze your photo collection and clean it up automatically.")

# Button
if st.button("Run Analysis"):

    imgs, names = load_images("images")

    st.subheader("Loaded Images")
    st.write(f"{len(imgs)} images found")

    # Features
    features = extract_features(imgs)

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

    # CLUSTERS
    st.subheader("Clusters")

    labels = cluster_images(features, k=7)

    clusters = {}
    for i, label in enumerate(labels):
        clusters.setdefault(label, []).append(i)

    for label, indices in clusters.items():
        st.write(f"Cluster {label}")

        cols = st.columns(4)
        for idx, i in enumerate(indices):
            cols[idx % 4].image(imgs[i], caption=names[i])

    # VISUALIZATIONS
    st.subheader("Visualizations")

    st.image("figures/similarity_histogram.png")
    st.image("figures/pca_clusters.png")
    st.image("figures/blur_scores.png")