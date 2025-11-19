# streamlit_app.py
import streamlit as st
import requests
import json
import pandas as pd
import pickle
import os
import re
import time  # For potential delays
import os

API_URL = os.getenv('API_URL', 'http://localhost:8000')  # Then use API_URL in requests.post


@st.cache_resource
def load_pipeline_info():
    with open('models/full_pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    model = pipeline['model']
    feature_cols = pipeline['features']
    continuous_cols = pipeline['continuous_cols']
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    return feature_cols, continuous_cols, importances

feature_cols, continuous_cols, importances = load_pipeline_info()

# Binary cols: Genres (match train: genre__*) - Now empty since no dummies
binary_cols = [col for col in feature_cols if col.startswith('genre__')]

def fetch_steam_data(url):
    if '/app/' not in url:
        return None
    appid = url.split('/app/')[1].split('/')[0]
    # Add cc=US for consistent USD pricing
    response = requests.get(f'https://store.steampowered.com/api/appdetails?appids={appid}&cc=US')
    if response.status_code != 200:
        return None
    data = response.json().get(appid, {}).get('data', {})
    
    # Price
    is_free = data.get('is_free', False)
    game_type = data.get('type', '')
    if is_free or 'free' in game_type.lower() or 'demo' in game_type.lower():
        price = 0.0
    else:
        price_overview = data.get('price_overview', {})
        final_cents = price_overview.get('final', 999)
        price = final_cents / 100
    
    platforms = data.get('platforms', {})
    platforms_count = sum([platforms.get(p, False) for p in ['windows', 'mac', 'linux']])
    genre_count = len(data.get('genres', []))
    tag_count = len(data.get('categories', []))  # Proxy
    achievements = data.get('achievements', {}).get('total', 0) if data.get('achievements') else 0
    dlc_count = len(data.get('dlc', [])) if 'dlc' in data else 0
    release_str = data.get('release_date', {}).get('date', 'Jan 1, 2025')
    try:
        release_date = pd.to_datetime(release_str, dayfirst=True, errors='coerce')
        release_year = release_date.year if pd.notna(release_date) else 2025
    except:
        release_year = 2025
    release_year = min(release_year, 2025)
    
    # Bump legacy/evolved games to effective recent year
    name = data.get('name', '').lower()
    if release_year < 2020 and any(keyword in name for keyword in ['counter-strike', 'cs2', 'dota', 'team fortress', 'left 4 dead']):
        release_year = 2023  # Valve relaunches (add more as needed)
    elif release_year < 2020 and 'upcoming' in data.get('short_description', '').lower():
        release_year = 2025  # Proxy for announced but undated
    
    # Impute achievements for new/low
    if achievements == 0 or achievements < 5:
        achievements = 25  # Proxy for upcoming/early
    
    # No genre dummies - rely on genre_count only
    
    # Supported languages count
    lang_str = data.get('supported_languages', '')
    supported_languages = len(re.split(r',\s*', lang_str)) if lang_str else 1
    developers_count = len(data.get('developers', []))
    
    # Franchise IP proxy
    publishers = data.get('publishers', [])
    big_ip = 1 if any('Nickelodeon' in p or 'THQ' in p or 'Paramount' in p or 'EA' in p or 'Ubisoft' in p or 'Valve' in p for p in publishers) else 0
    
    # Price tier with IP adjust
    price_tier = 0 if price == 0 else 1 if price < 5 else 2 if price < 15 else 2 if price < 30 else 3  # Simplified bins
    price_ip_adjust = price_tier * (1 - big_ip * 0.5)
    
    # F2P-IP boost
    f2p_ip_boost = 1 if price == 0 and big_ip == 1 else 0
    
    feats = {
        'price': price,
        'dlc_count': dlc_count,
        'achievements': achievements,
        'supported_languages': supported_languages,
        'genre_count': genre_count,
        'platforms_count': platforms_count,
        'release_year': release_year,
        'tag_count': tag_count,
        'developers_count': developers_count,
        'big_ip': big_ip,
        'price_tier': price_tier,
        'price_ip_adjust': price_ip_adjust,
        'f2p_ip_boost': f2p_ip_boost
    }
    
    # Fill missing to feature_cols
    for col in feature_cols:
        if col not in feats:
            feats[col] = 0
    
    # DEBUG: Log feats to console
    print(f"DEBUG: Feats for appid {appid}: {feats}")
    
    return feats

def format_value(val, col):
    if col in binary_cols:
        return '✅' if val == 1 else '❌'
    elif col == 'price':
        return "Free" if val == 0 else f"${val:.2f}"
    elif isinstance(val, (int, float)) and abs(val - round(val)) < 1e-6:
        return int(round(val))
    else:
        return round(val, 2) if isinstance(val, float) else val

# UI
st.title("Steam Game Popularity Predictor")
st.markdown("Enter a Steam URL to fetch metadata and predict >90% positive reviews chance.")

url = st.text_input("Steam Game URL:")
threshold = st.slider("High Threshold", 0.1, 0.6, 0.3)

if st.button("Predict"):
    if url:
        feats = fetch_steam_data(url)
        if feats:
            response = requests.post(API_URL, json=feats)
            if response.status_code == 200:
                result = response.json()
                proba = result['probability']
                is_high = proba > threshold
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric("Popularity Chance (>90% Positive)", f"{proba:.1%}", delta="High" if is_high else "Low")
                with col2:
                    st.progress(proba)
                
                if is_high:
                    st.success("🚀 High Potential!")
                else:
                    st.warning("📉 Moderate/Riskier Bet")
                
                st.info("Owners: Baseline (5K est. for unreleased) – not predictive. Release year from metadata (capped 2025).")
                
                with st.expander("Why this prediction? (Feature Breakdown)"):
                    st.write("Model favors indie/casual genres, high achievements, free/low-price. See top influences below.")
                    st.write("**Raw Features:**")  # NEW: Display raw feats for debug
                    st.json(feats)  # JSON for easy read
                    
                    exclude = ['release_year']  # Baselines
                    display_feats = {k: v for k, v in feats.items() if k not in exclude}
                    sorted_feats = sorted(display_feats.items(), key=lambda x: importances.get(x[0], 0), reverse=True)
                    display_df = pd.DataFrame(sorted_feats[:12], columns=['Feature', 'Value'])
                    display_df['Value'] = display_df.apply(lambda row: format_value(row['Value'], row['Feature']), axis=1)
                    display_df['Value'] = display_df['Value'].astype(str)  # FIX: Ensure str for Arrow
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                st.info("All features from training (e.g., genres, platforms). Proxies enhance for new games.")
            else:
                st.error(f"API Error: {response.text}")
        else:
            st.error("Invalid URL.")
    else:
        st.info("Paste a URL!")

st.markdown("---\nBuilt with Streamlit | XGBoost (AUC 0.742) | [Notebook](ml_project.ipynb)")