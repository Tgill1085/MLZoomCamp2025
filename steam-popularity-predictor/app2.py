# app2.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import os
import numpy as np

app = FastAPI(title="Steam Game Popularity Predictor API")

# Load pipeline
pipeline_path = 'models/full_pipeline.pkl'
with open(pipeline_path, 'rb') as f:
    pipeline = pickle.load(f)
model = pipeline['model']
scaler = pipeline['scaler']
feature_cols = pipeline['features']
continuous_cols = pipeline['continuous_cols']

class GameData(BaseModel):
    price: float = 0.0
    dlc_count: int = 0
    achievements: int = 0
    supported_languages: int = 1
    developers_count: int = 1
    platforms_count: int = 1
    genre_count: int = 1
    tag_count: int = 5
    release_year: int = 2025
    big_ip: int = 0
    price_tier: int = 0
    price_ip_adjust: float = 0.0
    f2p_ip_boost: int = 0

@app.post("/predict")
async def predict_popularity(data: GameData):
    try:
        df = pd.DataFrame([data.model_dump()]).reindex(columns=feature_cols, fill_value=0)
        avail_cont = [c for c in continuous_cols if c in df.columns]
        df[avail_cont] = scaler.transform(df[avail_cont])
        pred = model.predict(df)[0]
        proba = float(model.predict_proba(df)[0][1])
        return {
            "popularity": "High (>90% positive reviews)" if pred == 1 else "Low",
            "probability": round(proba, 4),
            "features_used": feature_cols
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)