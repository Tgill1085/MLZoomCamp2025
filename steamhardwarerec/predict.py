import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import os
import requests
import urllib.request

# === Get script directory (safe, no chdir) ===
script_dir = os.path.dirname(os.path.abspath(__file__))

# === Load data and model ===
@st.cache_data(show_spinner="Loading data and model...")
def load_data():
    # All paths relative to script location
    csv_path = os.path.join(script_dir, 'steam_games_final.csv')
    
    if not os.path.exists(csv_path):
        st.error(f"Data file 'steam_games_final.csv' not found!")
        st.info("""
        This file should be in the repository.
        Please ensure it is committed and pushed.
        """)
        st.stop()
    
    df = pd.read_csv(csv_path)
    
    # Model with download from GitHub Release
    model_path = os.path.join(script_dir, 'can_run_model_final.pkl')
    model_release_url = "https://github.com/Tgill1085/MLZoomCamp2025/releases/download/model-v1/can_run_model_final.pkl"  # UPDATE AFTER RELEASE
    
    model = None
    
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            st.success("Pre-trained model loaded!")
        except Exception as e:
            st.warning(f"Local model failed: {e}")
    
    if model is None:
        st.info("Downloading model from GitHub (~150 MB)...")
        with st.spinner("Downloading..."):
            try:
                urllib.request.urlretrieve(model_release_url, model_path)
                model = joblib.load(model_path)
                st.success("Model downloaded and loaded!")
            except Exception as e:
                st.error(f"Download failed: {e}")
                st.stop()
    
    # Benchmark files
    cpu_csv = os.path.join(script_dir, 'cpu_benchmarks_2026_extended.csv')
    gpu_csv = os.path.join(script_dir, 'gpu_benchmarks_2026_extended.csv')
    
    if not os.path.exists(cpu_csv) or not os.path.exists(gpu_csv):
        st.error("Benchmark files missing.")
        st.stop()
    
    df_cpu = pd.read_csv(cpu_csv)
    df_gpu = pd.read_csv(gpu_csv)
    
    cpu_lookup = {row['cpu_name'].lower().strip(): row['perf_score'] for _, row in df_cpu.iterrows()}
    gpu_lookup = {row['gpu_name'].lower().strip(): row['perf_score'] for _, row in df_gpu.iterrows()}
    
    cpu_manufacturers = sorted(df_cpu['manufacturer'].dropna().str.strip().str.title().unique())
    gpu_manufacturers = sorted(df_gpu['manufacturer'].dropna().str.strip().str.title().unique())
    
    return df, model, cpu_lookup, gpu_lookup, cpu_manufacturers, gpu_manufacturers, df_cpu, df_gpu

df_steam, model, cpu_lookup, gpu_lookup, cpu_manufacturers, gpu_manufacturers, df_cpu, df_gpu = load_data()

df_valid = df_steam[df_steam['intensity'] > 10].copy()
if len(df_valid) == 0:
    df_valid = df_steam.copy()

# === Header image helper (cached) ===
@st.cache_data(ttl=3600)
def get_header_url(appid):
    api_url = f"https://store.steampowered.com/api/appdetails?appids={appid}"
    try:
        response = requests.get(api_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            app_data = data.get(str(appid), {}).get('data', {})
            if app_data and 'header_image' in app_data:
                return app_data['header_image']
    except:
        pass
    return f"https://steamcdn-a.akamaihd.net/steam/apps/{appid}/header.jpg"

# === Fuzzy matching ===
def get_perf_score(user_input, lookup):
    if not user_input or not user_input.strip():
        return 0.0
    query = user_input.lower().strip()
    try:
        from rapidfuzz import process, fuzz
        result = process.extractOne(query, lookup.keys(), scorer=fuzz.partial_token_sort_ratio)
        if result and result[1] >= 85:
            return lookup[result[0]]
    except:
        pass
    return 0.0

# === Steam API helpers ===
def fetch_game_details(appid):
    url = f"https://store.steampowered.com/api/appdetails?appids={appid}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return None, None, {}, {}
        data = response.json().get(str(appid), {}).get('data', {})
        if not data:
            return None, None, {}, {}
        name = data.get('name', 'Unknown Game')
        thumb_url = data.get('header_image', f"https://steamcdn-a.akamaihd.net/steam/apps/{appid}/header.jpg")
        min_html = data.get('pc_requirements', {}).get('minimum', '')
        rec_html = data.get('pc_requirements', {}).get('recommended', '')
        
        def parse_req(raw_html):
            if not raw_html:
                return {'processor': '', 'graphics': '', 'memory': ''}
            processor = re.search(r'Processor:</strong>\s*(.+?)(?=<br|<strong|$)', raw_html, re.I | re.S)
            graphics = re.search(r'Graphics:</strong>\s*(.+?)(?=<br|<strong|$)', raw_html, re.I | re.S)
            memory = re.search(r'Memory:</strong>\s*(.+?)(?=<br|<strong|$)', raw_html, re.I | re.S)
            return {
                'processor': processor.group(1).strip() if processor else '',
                'graphics': graphics.group(1).strip() if graphics else '',
                'memory': memory.group(1).strip() if memory else ''
            }
        
        return name, thumb_url, parse_req(min_html), parse_req(rec_html)
    except:
        return None, None, {}, {}

def extract_ram_gb(text):
    if not text or not isinstance(text, str):
        return 0.0
    text = re.sub(r'\\bMB\\b', 'GB', text, flags=re.IGNORECASE)
    match = re.search(r'(\\d+(?:\\.\\d+)?)\\s*GB', text, re.IGNORECASE)
    return float(match.group(1)) if match else 0.0

# === Game check logic ===
def check_specific_game(appid, user_cpu_score, user_gpu_score, user_ram_gb):
    game_row = df_steam[df_steam['appid'] == appid]
    if not game_row.empty:
        game = game_row.iloc[0]
        name = game['name']
        thumb_url = get_header_url(appid)
        min_cpu_score = game['cpu_score']
        min_gpu_score = game['gpu_score']
        min_ram_gb = game['ram_gb_final']
        
        rec_cpu = '' if pd.isna(game.get('mat_pc_processor_rec')) else game['mat_pc_processor_rec']
        rec_gpu = '' if pd.isna(game.get('mat_pc_graphics_rec')) else game['mat_pc_graphics_rec']
        rec_ram_text = '' if pd.isna(game.get('mat_pc_memory_rec')) else game['mat_pc_memory_rec']
        rec_ram_gb = extract_ram_gb(rec_ram_text) or min_ram_gb
        
        rec_cpu_score = get_perf_score(rec_cpu, cpu_lookup) or min_cpu_score
        rec_gpu_score = get_perf_score(rec_gpu, gpu_lookup) or min_gpu_score
        
        features_min = np.array([[
            user_cpu_score - min_cpu_score,
            user_gpu_score - min_gpu_score,
            user_ram_gb - min_ram_gb,
            user_cpu_score,
            user_gpu_score,
            user_ram_gb
        ]])
        conf_min = model.predict_proba(features_min)[0, 1]
        
        features_rec = np.array([[
            user_cpu_score - rec_cpu_score,
            user_gpu_score - rec_gpu_score,
            user_ram_gb - rec_ram_gb,
            user_cpu_score,
            user_gpu_score,
            user_ram_gb
        ]])
        conf_rec = model.predict_proba(features_rec)[0, 1]
        
        return (name, thumb_url, conf_min, conf_rec,
                game.get('mat_pc_processor_min', ''),
                game.get('mat_pc_graphics_min', ''),
                f"{min_ram_gb:.0f} GB" if min_ram_gb > 0 else '',
                rec_cpu, rec_gpu, rec_ram_text or f"{min_ram_gb:.0f} GB")
    
    name, thumb_url, min_req, rec_req = fetch_game_details(appid)
    if not name:
        return None
    
    return (name, thumb_url, None, None,
            min_req['processor'], min_req['graphics'], min_req['memory'],
            rec_req['processor'], rec_req['graphics'], rec_req['memory'])

# =============================================================================
# UI
# =============================================================================
st.set_page_config(page_title="Steam Games based on Hardware Recommender", layout="wide")

st.title("🖥️ Steam Games based on Hardware Recommender")
st.markdown("Enter your PC specs to discover what games you can run — powered by benchmark data and machine learning.")

st.markdown("### Your CPU, GPU and RAM")

col1, col2 = st.columns(2)

with col1:
    st.subheader("CPU")
    cpu_manufacturer = st.selectbox("Manufacturer (optional)", ["Any"] + cpu_manufacturers, key="cpu_manu")
    if cpu_manufacturer != "Any":
        filtered = df_cpu[df_cpu['manufacturer'].str.title() == cpu_manufacturer.title()]
        models = filtered['cpu_name'].tolist()
        clean_models = []
        for m in models:
            clean = m
            if cpu_manufacturer.lower() == 'amd':
                clean = re.sub(r'^amd\\s+', '', m, flags=re.IGNORECASE)
            elif cpu_manufacturer.lower() == 'intel':
                clean = re.sub(r'^intel\\s*(core\\s*)?', '', m, flags=re.IGNORECASE)
            clean_models.append(clean.title())
        model_map = dict(zip(clean_models, models))
        selected = st.selectbox("Model", ["Type or select..."] + clean_models, key="cpu_select")
        cpu_input = model_map.get(selected, selected) if selected != "Type or select..." else st.text_input("Custom CPU", key="cpu_custom")
    else:
        cpu_input = st.text_input("Your CPU", placeholder="e.g., Ryzen 7 7800X3D", key="cpu_text")

with col2:
    st.subheader("GPU")
    gpu_manufacturer = st.selectbox("Manufacturer (optional)", ["Any"] + gpu_manufacturers, key="gpu_manu")
    if gpu_manufacturer != "Any":
        filtered = df_gpu[df_gpu['manufacturer'].str.title() == gpu_manufacturer.title()]
        models = filtered['gpu_name'].tolist()
        clean_models = []
        for m in models:
            clean = m
            if gpu_manufacturer.lower() == 'nvidia':
                clean = re.sub(r'^geforce\\s+', '', m, flags=re.IGNORECASE)
            elif gpu_manufacturer.lower() == 'amd':
                clean = re.sub(r'^radeon\\s+', '', m, flags=re.IGNORECASE)
            clean_models.append(clean.title())
        model_map = dict(zip(clean_models, models))
        selected = st.selectbox("Model", ["Type or select..."] + clean_models, key="gpu_select")
        gpu_input = model_map.get(selected, selected) if selected != "Type or select..." else st.text_input("Custom GPU", key="gpu_custom")
    else:
        gpu_input = st.text_input("Your GPU", placeholder="e.g., RTX 4070", key="gpu_text")

ram_gb = st.slider("RAM (GB)", 4, 128, 32, 4)

st.markdown("---")

game_url = st.text_input("Steam Game URL (optional)", placeholder="https://store.steampowered.com/app/730/CounterStrike_2/")

col1, col2, _ = st.columns([1, 1, 2])
with col1:
    check_pressed = st.button("Check Game Compatibility", type="primary", use_container_width=True)
with col2:
    req_pressed = st.button("Show Requirements Only", use_container_width=True)

# =============================================================================
# Results
# =============================================================================
if check_pressed or req_pressed:
    has_hardware = bool(cpu_input.strip() or gpu_input.strip())
    has_url = bool(game_url.strip())
    
    if not has_hardware and not has_url:
        st.error("Please enter your hardware and/or a game URL.")
        st.stop()
    
    user_cpu_score = get_perf_score(cpu_input, cpu_lookup)
    user_gpu_score = get_perf_score(gpu_input, gpu_lookup)
    
    if user_cpu_score == 0 and cpu_input:
        st.warning(f"CPU '{cpu_input}' not recognized — treated as low-end.")
    if user_gpu_score == 0 and gpu_input:
        st.warning(f"GPU '{gpu_input}' not recognized — treated as low-end.")
    
    # Sticky specs bar
    st.markdown(f"""
    <div style="background: #0e1117; padding: 12px; border-radius: 8px; margin: 10px 0; font-size: 1.1em;">
        <strong>Your PC:</strong> CPU: {cpu_input or '—'} ({user_cpu_score:.0f}/100) | 
        GPU: {gpu_input or '—'} ({user_gpu_score:.0f}/100) | 
        RAM: {ram_gb} GB
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("ℹ️ How confidence scores work"):
        st.markdown("""
        - **Confidence** = model's predicted probability you can run the game smoothly
        - ≥70% = Very likely
        - 50–70% = Possible (may need lower settings)
        - <50% = Unlikely
        """)
    
    if has_url:
        match = re.search(r'/app/(\\d+)', game_url)
        if match:
            appid = int(match.group(1))
            result = check_specific_game(appid, user_cpu_score, user_gpu_score, ram_gb)
            if result:
                name, thumb, conf_min, conf_rec, min_cpu, min_gpu, min_ram, rec_cpu, rec_gpu, rec_ram = result
                
                st.markdown(f"### Can you run **{name}**?")
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(thumb, width=300)
                with col2:
                    if has_hardware and conf_min is not None:
                        st.markdown(f"**Minimum specs:** {'🟢 YES' if conf_min >= 0.5 else '🔴 NO'} (Confidence: {conf_min*100:.1f}%)")
                        st.markdown(f"**Recommended specs:** {'🟢 YES' if conf_rec >= 0.5 else '🔴 NO'} (Confidence: {conf_rec*100:.1f}%)")
                    st.markdown("**Minimum:**")
                    st.write(f"- CPU: {min_cpu or '—'}")
                    st.write(f"- GPU: {min_gpu or '—'}")
                    st.write(f"- RAM: {min_ram or '—'}")
                    st.markdown("**Recommended:**")
                    st.write(f"- CPU: {rec_cpu or '—'}")
                    st.write(f"- GPU: {rec_gpu or '—'}")
                    st.write(f"- RAM: {rec_ram or '—'}")
        else:
            st.error("Invalid Steam URL — please check and try again.")
    
    if check_pressed and has_hardware:
        # Top 20 runnable games
        features = pd.DataFrame({
            'delta_cpu': user_cpu_score - df_valid['cpu_score'].values,
            'delta_gpu': user_gpu_score - df_valid['gpu_score'].values,
            'delta_ram': ram_gb - df_valid['ram_gb_final'].values,
            'user_cpu': user_cpu_score,
            'user_gpu': user_gpu_score,
            'user_ram': ram_gb
        })
        df_valid['confidence'] = model.predict_proba(features.values)[:, 1]
        runnable = df_valid[df_valid['confidence'] >= 0.5].copy()
        runnable = runnable.sort_values('recommendations_total', ascending=False).head(20)
        
        st.success(f"You can likely run **{len(runnable):,}** of the top games!")
        st.subheader("Top 20 Most Popular Games You Can Run")
        
        cols = st.columns(2)
        for i, row in runnable.iterrows():
            col = cols[i % 2]
            with col:
                thumb = get_header_url(row['appid'])
                st.image(thumb, use_column_width=True)
                st.markdown(f"**[{row['name']}](https://store.steampowered.com/app/{row['appid']}/)**")
                st.caption(f"Confidence: {row['confidence']*100:.1f}%")

st.caption("Built with Tom's Hardware 2026 benchmarks • Steam data • scikit-learn • Streamlit")
