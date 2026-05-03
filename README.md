Photo AI - Smart Photo Organizer

A smart photo organization tool that analyzes image collections to detect duplicates, find blurry or low-quality photos, and group similar images into clusters. Includes both a command-line version and an interactive Streamlit web app.

Live Demo

Streamlit app:
https://smart-photo-organizer-zkkjubpzfruxlrz7yedxxq.streamlit.app/

Features
- Duplicate detection using image embeddings and cosine similarity
- Blur detection for low-quality photos (Laplacian variance)
- Image clustering to group similar pictures (KMeans)
- Automatic cluster labeling using image captions (BLIP)
- Interactive Streamlit UI (app.py)
- Upload your own images or use default dataset

Visualizations:
- similarity histogram
- PCA cluster plot
- blur score graph

Organized output:
- output/keep
- output/trash
- output/organized

Download results as a ZIP file (Streamlit app)

Requirements
- Python 3.8+
- See requirements.txt for dependencies

Setup
Create a virtual environment:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1   # PowerShell (Windows)
# or: venv\Scripts\activate.bat  # CMD
# or: source venv/bin/activate   # macOS / Linux
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run
Streamlit App

```bash
streamlit run app.py
```

Command-line version

```bash
python main.py
```

Project Layout
- app.py — Streamlit interface
- main.py — CLI runner and image loader
- features.py — CLIP feature extraction + BLIP captioning
- similarity.py — duplicate detection logic
- blur.py — blur detection helper
- cluster.py — clustering + optimal k selection
- visualize.py — graphs and plots
- organize.py — output organization + ZIP creation
- images/ — input images
- output/ — organized results
- figures/ — generated visualizations

How It Works
- Images are loaded from a folder or uploaded by the user
- CLIP generates feature embeddings for each image
- Cosine similarity detects duplicates
- Laplacian variance detects blur
- KMeans clusters similar images
- BLIP generates captions for representative images
- Results are organized into keep, trash, and grouped folders

Datasets Used
- Labeled Faces in the Wild (LFW): https://www.kaggle.com/datasets/jessicali9530/lfw-dataset
- Flickr: https://www.flickr.com/search/
- Unsplash: https://unsplash.com/

Resources
- CLIP model: https://huggingface.co/openai/clip-vit-base-patch32
- CLIP GitHub: https://github.com/openai/CLIP
- BLIP (image captioning model)

Notes
- The app copies images into output folders instead of deleting originals
- Similarity and blur thresholds can be adjusted for different datasets

This is a prototype and can be extended with:
- better labeling
- face/person grouping
- larger datasets
- improved UI
