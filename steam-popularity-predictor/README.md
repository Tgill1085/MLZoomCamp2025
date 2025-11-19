# Steam Game Popularity Predictor

## Project Overview

### Problem Description
This project predicts whether a Steam game will achieve **high popularity** (defined as >90% positive reviews from users) based solely on **pre-release metadata**. This is crucial for game developers, publishers, and investors to assess market potential before launch, avoiding costly flops. 

Inspired by research like Hu et al. (2024) on Steam dynamics, we use features like price, platforms (Windows/Mac/Linux), genres count, achievements, supported languages, release year, DLC count, and engineered proxies (e.g., `big_ip` for franchise boosts like Valve/EA titles, `price_ip_adjust` to discount high prices for IPs, `f2p_ip_boost` for free-to-play hits like CS2).

- **Dataset**: ~89k Steam games (March 2025 snapshot) with review counts, metadata. Target imbalance (~27% "High") addressed via SMOTE oversampling.
- **Model**: XGBoost classifier (AUC 0.743 on test set) after tuning (RandomizedSearchCV on n_estimators, max_depth, etc.). Handles biases like legacy F2P underprediction.
- **Deployment**: FastAPI backend for predictions + Streamlit UI for URL-based game analysis (fetches Steam API metadata).
- **Use Case**: Input a Steam store URL (e.g., for an upcoming indie game); get probability, feature breakdown, and "High Potential" verdict.

**Why It Matters**: Steam's ~50k games/year see ~70% fail <90% positivity. This tool flags winners early, e.g., predicting SpongeBob (IP-adjusted) at 74% (real: 87%) or CS2 F2P-IP at 89% (real: 86%).

## Quick Start
1. **Clone & Setup**:

   ```
   git clone https://github.com/yourusername/steam-popularity-predictor.git
    cd steam-popularity-predictor
    pip install -r requirements.txt
   ```

2. **Download Data** (if not committed):
- Dataset: `games_march2025_cleaned.csv` (89k rows, ~10MB). Download from [Kaggle Steam Games Dataset](https://www.kaggle.com/datasets/nikdavis/steam-store-games) or place in `./archive/`.

3. **Train Model**:

   ```
   python train.py  # Outputs models/full_pipeline.pkl (XGBoost + scaler)
   ```

4. **Run API (Prediction Service)**:

   ```
   uvicorn app2:app --reload --host 0.0.0.0 --port 8000
   ```

- Test: POST to `http://localhost:8000/predict` with JSON (e.g., `{"price": 0.0, "achievements": 51}`).
- Docs: http://localhost:8000/docs

5. **Run Streamlit UI**:
   ```
   streamlit run streamlit_app.py
   ```

- Open http://localhost:8501. Paste Steam URL (e.g., https://store.steampowered.com/app/730/CounterStrike_2/) for auto-fetch + predict.

6. **Docker (Local Deployment)**:
- Build/Run (API + UI): `docker-compose up --build`
  - API: http://localhost:8000/docs
  - UI: http://localhost:8501
- Single API: `docker build -t steam-predictor . && docker run -p 8000:8000 steam-predictor`
- Single UI: `docker run -p 8501:8501 steam-predictor streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0`

**Notes**:
- Model favors low-price/indie (e.g., free + 50 achievements + multi-platform → 85%+ prob).
- Limitations: No post-launch leaks (e.g., owners); proxies for unreleased (e.g., achievements=25).
- Eval: Test AUC ~0.74; confusion matrix in notebook.

## File Structure
- `ml_project.ipynb`: Full EDA (imbalance viz, feature eng like `price_ip_adjust`), tuning, plots (importances, IP interactions).
- `train.py`: Trains/saves XGBoost pipeline (SMOTE, scaling, RandomizedSearchCV).
- `app2.py`: FastAPI for `/predict` (loads pickle, scales, returns proba).
- `streamlit_app.py`: UI fetches Steam data, calls API, shows metrics/breakdown.
- `serialize.py`: Verification script (mock predict, resave pipeline).
- `archive/games_march2025_cleaned.csv`: Data (committed; or download instr.).
- `models/full_pipeline.pkl`: Trained model (generated on train).
- `requirements.txt`: Dependencies.
- `Dockerfile`: Builds container (trains model on build).
- `docker-compose.yml`: Runs API + UI services.

## Reproducibility
- Random seeds: 42 (splits, SMOTE, XGBoost).
- Env: Python 3.10+; virtualenv recommended (`python -m venv env; source env/bin/activate`).
- No GPU needed; ~2min train on CPU.
- Docker ensures isolation: Builds with data/code, trains inside.

