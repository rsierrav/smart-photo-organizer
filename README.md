# Photo AI — Smart Photo Organizer

A small collection of utilities and a Streamlit app to analyze photo collections: detect duplicates, find blurry images, and group similar photos into clusters.

## Features
- Duplicate detection using image features
- Blur detection for low-quality photos
- Image clustering to group similar pictures
- Streamlit UI for interactive analysis (`app.py`)

## Requirements
- Python 3.8+
- See `requirements.txt` for exact dependencies

## Setup
1. Create a virtual environment:

	 ```powershell
	 python -m venv venv
	 venv\Scripts\Activate.ps1   # PowerShell (Windows)
	 # or: venv\Scripts\activate.bat  # CMD (Windows)
	 # or: source venv/bin/activate     # macOS / Linux
	 ```

2. Install dependencies:

	 ```bash
	 pip install -r requirements.txt
	 ```

## Run
- Launch the Streamlit UI (recommended):

	```bash
	streamlit run app.py
	```

- Or run the command-line analyzer:

	```bash
	python main.py
	```

## Project layout
- `app.py` — Streamlit interface that runs the analysis
- `main.py` — CLI-style runner and image loader
- `features.py` — feature extraction utilities
- `similarity.py` — duplicate detection logic
- `blur.py` — blur detection helper
- `cluster.py` — image clustering wrapper
- `images/` — sample or input images folder

## Notes
- The Streamlit app expects images in the `images/` folder by default.
- Tweak clustering `k` and other parameters in `app.py` or `main.py` as needed.
