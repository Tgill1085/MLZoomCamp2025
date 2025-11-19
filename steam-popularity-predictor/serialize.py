# serialize.py
import pickle
import joblib
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore", message="InconsistentVersionWarning")

# Load from pipeline (no train.csv needed)
pipeline_path = 'models/full_pipeline.pkl'
with open(pipeline_path, 'rb') as f:
    pipeline = pickle.load(f)
best_model = pipeline['model']
scaler = pipeline['scaler']
feature_cols = pipeline['features']
continuous_cols = pipeline['continuous_cols']

print(f"Loaded pipeline: {len(feature_cols)} features, continuous: {continuous_cols}")

# Mock test: High example (Portal 2-like: indie puzzle, low price, achievements, multi-platform)
# Limit to exact training feats (no genre dummies)
data_dict = {col: 0 for col in feature_cols}
data_dict.update({
    'price': 0.0,  # Top import—free boosts
    'achievements': 51,  # Engagement
    'supported_languages': 10,  # Reach
    'genre_count': 2,  # e.g., Indie + Adventure (neutral variety)
    'tag_count': 25,  # Rich tags (top import 0.164—higher for High)
    'platforms_count': 3,  # Multi
    'release_year': 2025,  # New game sim (import 0.095—recent favors high)
    'developers_count': 1,  # Indie
    'dlc_count': 0  # Low
})

X_sample = pd.DataFrame([data_dict])
print(f"Mock shape: {X_sample.shape}, Sample cols: {list(X_sample.columns[:5])}...")

# Scale continuous (skip owners_numeric if not in feats)
avail_cont = [c for c in continuous_cols if c in X_sample.columns]
print(f"Scaling {len(avail_cont)} continuous cols: {avail_cont}")
X_sample_scaled = X_sample.copy()
if avail_cont:
    X_sample_scaled[avail_cont] = scaler.transform(X_sample[avail_cont])

# Predict
pred = best_model.predict(X_sample_scaled)[0]
proba = best_model.predict_proba(X_sample_scaled)[0][1]
print(f"Portal 2-like: {'High' if pred == 1 else 'Low'} | Proba: {proba:.3f} (Expected High; recent/indie/tags/low-price)")

# Save updated pipeline
pipeline = {'model': best_model, 'scaler': scaler, 'features': feature_cols, 'continuous_cols': continuous_cols}
with open('models/full_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
print("Pipeline resaved (verified)! Now test API/UI with `uvicorn app2:app --reload` and `streamlit run streamlit_app.py`.")