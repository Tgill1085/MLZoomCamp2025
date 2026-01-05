# Hardware Based Steam Games Recommender

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://steamgameshardwarerec.streamlit.app)  
*A machine learning-powered tool to recommend Steam games based on your PC hardware.*

![App Screenshot](app-screenshot.png)

## Problem & Motivation

In early 2026, PC gamers face several challenges:
- **Hardware Pricing Crisis**: GPUs, CPUs, and RAM have become prohibitively expensive due to commercial supply outranking consumers, making PC upgrades and gaming difficult.
- **Lack of Non-Invasive Hardware-Based Game Recommendations**: Most tools require a file download, system scan, or suggest games by genre or popularity, ignoring whether your actual PC can run them smoothly.
- **Inconsistent & Messy Data**: Steam system requirements are often vague, outdated, missing, or poorly formatted. Benchmark data is scattered across sites.

This project solves these by building a **personalized "Can I Run It?" recommender** using:
- Real 2026 Tom's Hardware CPU/GPU benchmarks
- Filtered and Enriched Steam dataset (50,000 games)
- Fuzzy matching + rule-based imputation for unmatched hardware
- Synthetic data + Random Forest ML for accurate compatibility prediction

It demonstrates end-to-end handling of **real-world messy data**, **imputation**, **fuzzy string matching**, and **machine learning** to deliver practical, hardware-aware game recommendations.

## Features

- Enter your CPU, GPU, and RAM and get 0–100 component performance scores
- Paste any Steam game URL, return "Yes/No" prediction with the model confidence %
- View minimum & recommended specs side-by-side (if available in data)
- Listing of the Top 20 most popular games you can actually run
- Catalog highlights: Top 10 most & least demanding games
- Fully reproducible pipeline, and available as a live demo

## Live Demo

Try it now — no installation needed!  
**https://steamgameshardwarerec.streamlit.app**  


## Quick Start

### Option 1: Virtual Environment (Recommended)

```bash
git clone https://github.com/Tgill1085/MLZoomCamp2025.git
cd MLZoomCamp2025/steamhardwarerec

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate data & train model (first time only)
python train.py

# Launch the app
streamlit run predict.py
```
Open http://localhost:8501 in your browser.


### Option 2: Docker
```bash
git clone https://github.com/Tgill1085/MLZoomCamp2025.git
cd MLZoomCamp2025/steamhardwarerec

# Build and run
docker build -t steam-recommender .
docker run -p 8501:8501 steam-recommender
```
Open http://localhost:8501
Note: The first run (inside Docker) will automatically train the model if missing.

### Option 3: Just Use the Web App
No setup needed!
Go directly to: https://steamgameshardwarerec.streamlit.app

## Deployment

This app is designed for easy deployment on **Streamlit Community Cloud** (free tier available).  
The trained model (`can_run_model_final.pkl` ≈ 150 MB) is too large for direct Git commit, so it is hosted as a **GitHub Release asset** and automatically downloaded by `predict.py` on first run.

### 1. Upload the Model as a GitHub Release

1. Train your model locally using `train.py` or `notebook.ipynb` to generate `can_run_model_final.pkl`.
2. Go to your repository: https://github.com/YOUR_USERNAME/YOUR_REPOSITORY
3. Click **Releases** → **Draft a new release**.
4. Create a new tag (e.g., `model-v1`).
5. Add a title like **"Model v1"** and optional description.
6. Drag and drop `can_run_model_final.pkl` into the assets section.
7. Click **Publish release**.

> **Important**: The download URL in `predict.py` is currently:  
> `https://github.com/Tgill1085/MLZoomCamp2025/releases/download/model-v1/can_run_model_final.pkl`  
> If you change the tag or filename, update the `model_release_url` variable in `predict.py`.

### 2. Deploy to Streamlit Community Cloud

1. Sign up or log in at [https://share.streamlit.io](https://share.streamlit.io)
2. Click **New app**.
3. Connect your GitHub repository: `YOUR_USERNAME/YOUR_REPOSITORY`
4. Set:
   - **Branch**: `main` (or your preferred branch)
   - **Main file path**: `YOUR_REPOSITORY/predict.py`
5. Click **Deploy**.

Streamlit will:
- Automatically install dependencies from `requirements.txt`
- Pull the CSV and benchmark files from your repo
- Download the `.pkl` model from your GitHub Release on first launch (may take ~30–60 seconds)

Your live app URL will look like: `https://your-app-name.streamlit.app`

### Optional: Local Development & Docker

- **Run locally**:  
  ```bash
  streamlit run steamhardwarerec/predict.py

### Project Structure
``` text
.
├── data_gathering.ipynb                  # Scrapes benchmark data + processes/enriches Steam games
├── notebook.ipynb                        # Exploratory analysis, feature engineering, and modeling
├── train.py                              # Standalone script: trains and saves the ML model
├── predict.py                            # Streamlit web app (main deployed application)
├── requirements.txt                      # Python dependencies
├── Dockerfile                            # For containerized deployment (e.g., Streamlit Cloud, Docker)
├── .dockerignore                         # Files to exclude from Docker build
├── cpu_benchmarks_2026.csv               # Raw CPU benchmark data from scraping
├── cpu_benchmarks_2026_extended.csv      # Extended/cleaned CPU dataset used in app
├── gpu_benchmarks_2026.csv               # Raw GPU benchmark data from scraping
├── gpu_benchmarks_2026_extended.csv      # Extended/cleaned GPU dataset used in app
├── steam_games_final.csv                 # Final enriched Steam dataset with parsed specs and intensity scores
├── steam_games_slim.csv                  # Slimmed-down version of Steam application data
├── can_run_model_final.pkl               # Trained ML model (downloaded from GitHub Release on first run if missing)
└── app-screenshot.png                    # Screenshot of the deployed Streamlit app
```

## How It Works

1. **Data Gathering**  
   Auto-downloads the Steam dataset from Zenodo and scrapes 2026 CPU/GPU benchmarks from Tom's Hardware.

2. **Enrichment**  
   Uses fuzzy matching (RapidFuzz) + hand-crafted rules to handle unmatched and low-end GPUs/CPUs for maximum coverage.

3. **Intensity Scoring**  
   Computes a weighted demand score for each game:  
   **70% GPU** + **20% CPU** + **10% RAM**  
   (Higher Scoring = more demanding game)

4. **Synthetic Training**  
   Generates 100,000 balanced user-game pairs to train the model on realistic "can run / cannot run" scenarios.

5. **Random Forest Model**  
   Predicts compatibility probability using hardware deltas (user vs game requirements).

6. **Streamlit App**  
   Interactive web UI for entering hardware, checking any Steam game, and getting personalized recommendations.

## Notes for Reproducibility

- The Large model file is either generated automatically on the first run, or can be found under releases:
    - [Trained Model v1](https://github.com/Tgill1085/MLZoomCamp2025/releases/tag/model-v1).
- `train.py` checks for required data and trains a fresh model if needed.
- All scraping and downloading steps are fully automated — no manual file uploads required.

## Built With

- **Python** / **Pandas** / **Scikit-learn** – Core data processing and modeling
- **RapidFuzz** – High-performance fuzzy string matching
- **Streamlit** – Interactive web app
- **Docker** – Containerized deployment option







