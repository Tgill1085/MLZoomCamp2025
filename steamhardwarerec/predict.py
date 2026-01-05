import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import os
import requests
import urllib.request

# === Get script directory safely ===
script_dir = os.path.dirname(os.path.abspath(__file__))  # Fixed: __file__ not **file**

# === Load data and model ===
@st.cache_data(show_spinner="Loading data and model...")
def load_data():
    # Main data file
    csv_path = os.path.join(script_dir, 'steam_games_final.csv')
    
    if not os.path.exists(csv_path):
        st.error("Data file 'steam_games_final.csv' not found!")
        st.info("This file should be in the repository. Run data_gathering.ipynb locally and push it.")
        st.stop()
    
    df = pd.read_csv(csv_path)
    
    # Model loading with GitHub Release fallback
    model_path = os.path.join(script_dir, 'can_run_model_final.pkl')
    model_release_url = "https://github.com/Tgill1085/MLZoomCamp2025/releases/download/model-v1/can_run_model_final.pkl"
    
    model = None
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            st.spinner("Pre-trained model loaded!")
        except Exception as e:
            st.warning(f"Local model failed: {e}")
    
    if model is None:
        st.spinner("Model not found — downloading from GitHub (~150 MB)...")
        with st.spinner("Downloading model..."):
            try:
                urllib.request.urlretrieve(model_release_url, model_path)
                model = joblib.load(model_path)
                st.spinner("Model downloaded and loaded!")
            except Exception as e:
                st.error(f"Download failed: {e}")
                st.error("Check the release URL in the code.")
                st.stop()
    
    # === FIXED: Load benchmark CSVs using script_dir ===
    cpu_csv_path = os.path.join(script_dir, 'cpu_benchmarks_2026_extended.csv')
    gpu_csv_path = os.path.join(script_dir, 'gpu_benchmarks_2026_extended.csv')
    
    if not os.path.exists(cpu_csv_path):
        st.error(f"Benchmark file not found: {cpu_csv_path}")
        st.info("Make sure 'cpu_benchmarks_2026_extended.csv' is in your GitHub repository root.")
        st.stop()
    
    if not os.path.exists(gpu_csv_path):
        st.error(f"Benchmark file not found: {gpu_csv_path}")
        st.info("Make sure 'gpu_benchmarks_2026_extended.csv' is in your GitHub repository root.")
        st.stop()
    
    df_cpu = pd.read_csv(cpu_csv_path)
    df_gpu = pd.read_csv(gpu_csv_path)
    
    # Build lookups
    cpu_lookup = {row['cpu_name'].lower().strip(): row['perf_score'] for _, row in df_cpu.iterrows()}
    gpu_lookup = {row['gpu_name'].lower().strip(): row['perf_score'] for _, row in df_gpu.iterrows()}
    
    cpu_manufacturers = sorted(df_cpu['manufacturer'].dropna().str.strip().str.title().unique())
    gpu_manufacturers = sorted(df_gpu['manufacturer'].dropna().str.strip().str.title().unique())
    
    return df, model, cpu_lookup, gpu_lookup, cpu_manufacturers, gpu_manufacturers, df_cpu, df_gpu

# Load everything
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

def get_perf_score(user_input, lookup):
    if not user_input or not user_input.strip():
        return 0.0
    query = user_input.lower().strip()
    try:
        from rapidfuzz import process, fuzz
        result = process.extractOne(query, lookup.keys(), scorer=fuzz.partial_token_sort_ratio)
        if result and result[1] >= 85:
            return lookup[result[0]]
    except ImportError:
        pass
    return 0.0

def fetch_game_details(appid):
    url = f"https://store.steampowered.com/api/appdetails?appids={appid}"
    response = requests.get(url)
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

def extract_ram_gb(text):
    if not text or not isinstance(text, str):
        return 0.0
    text = re.sub(r'\bMB\b', 'GB', text, flags=re.IGNORECASE)
    match = re.search(r'(\d+(?:\.\d+)?)\s*GB', text, re.IGNORECASE)
    return float(match.group(1)) if match else 0.0

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
        
        features_min = pd.DataFrame([{
            'delta_cpu': user_cpu_score - min_cpu_score,
            'delta_gpu': user_gpu_score - min_gpu_score,
            'delta_ram': user_ram_gb - min_ram_gb,
            'user_cpu': user_cpu_score,
            'user_gpu': user_gpu_score,
            'user_ram': user_ram_gb
        }])
        conf_min = model.predict_proba(features_min.values)[0, 1]
        
        features_rec = pd.DataFrame([{
            'delta_cpu': user_cpu_score - rec_cpu_score,
            'delta_gpu': user_gpu_score - rec_gpu_score,
            'delta_ram': user_ram_gb - rec_ram_gb,
            'user_cpu': user_cpu_score,
            'user_gpu': user_gpu_score,
            'user_ram': user_ram_gb
        }])
        conf_rec = model.predict_proba(features_rec.values)[0, 1]
        
        return (name, thumb_url, conf_min, conf_rec,
                '' if pd.isna(game.get('mat_pc_processor_min')) else game['mat_pc_processor_min'],
                '' if pd.isna(game.get('mat_pc_graphics_min')) else game['mat_pc_graphics_min'],
                f"{min_ram_gb:.0f} GB" if pd.notna(min_ram_gb) and min_ram_gb > 0 else '',
                rec_cpu,
                rec_gpu,
                rec_ram_text if rec_ram_text else (f"{min_ram_gb:.0f} GB" if pd.notna(min_ram_gb) else ''))
    
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

st.title("Steam Games based on Hardware Recommender")
st.markdown("Enter your PC specs to discover what games you can run — powered by benchmark data and machine learning.")

# === Define the dialogs FIRST (before any buttons try to call them) ===

@st.dialog("How to Find Your PC Specs on Windows")
def show_components_help():
    st.markdown("""
    ### How to Find Your CPU, GPU, and RAM on Windows
    
    Follow these steps to quickly locate your hardware details:
    
    **Best Method – System Information:**
    1. Press **Windows key + R** 
       - type `msinfo32` and press Enter.
    2. Under **System Summary** (default view on left panel) check the right panel for:
       - `Processor` - for your full CPU model (e.g., "AMD Ryzen 7 7800X3D").
       - `Installed Physical Memory (RAM)` -  for your Total RAM (e.g., "32.0 GB").
    3. In the left panel list, expand **Components** and click on **Display**:
       - `Name` - For your GPU model (e.g., "NVIDIA GeForce RTX 4070").
    
    **Alternative for GPU:**
    1. Right-click Desktop and click **Display settings** 
       - Scroll down in this window and click on **Advanced display settings**
       - Click on **Display adapter properties**, which opens another window 
       - Find `Adapter Type` displaying the full GPU Model at the top of the window.
    2. Search Start Menu or Settings for **Device Manager**
       - In the left panel, expand `Display adapters`
       - Find Full GPU Model name in list of adapters
    """)
    
    if st.button("Close", type="primary", use_container_width=True):
        st.rerun()

@st.dialog("How to Use This Site")
def show_usage_help():
    st.markdown("""
    ### Four Ways to Use This Recommender
    
    1. **Get General Recommendations**  
       - Enter your CPU, GPU, and RAM.  
       - Click `Get Single Steam Game Analysis and/or Steam Games Recommendations`.  
       - You'll see the top 20 most popular games your PC can likely run (plus a total from the catalogue count out of 50,000 games).
    
    2. **Check a Specific Game + Get Recommendations**  
       - Enter your hardware specs.  
       - Paste a **Steam store URL** (e.g., https://store.steampowered.com/app/...).  
       - Click `Get Single Steam Game Analysis and/or Steam Games Recommendations`.  
       - See if your PC can run it (with confidence scores) **plus** your system's top 20 list.
    
    3. **Just View a Game's System Requirements**  
       - Paste a Steam store URL (hardware entry optional).  
       - Click `Just Get Single Steam Game Requirements`.  
       - Displays the official minimum and recommended specs (no prediction needed).

    4. **Browse the Top 10 and Bottom 10 Games**
       - When you visit the site, you can find the list of the most system intensive games and the least intensive games listed below the input form. 
    
    **Tip:** Confidence ≥70% = very likely to run well · 50–70% = probably with lower settings, see more under `How to read the results?`.
    """)
    
    if st.button("Close", type="primary", use_container_width=True):
        st.rerun()

@st.dialog("How to Read the Results")
def show_results_help():
    st.markdown("""
    # Understanding the Results
    
    1. ### **Performance Score (0–100)**  
    Your CPU and GPU get a relative gaming performance score based on 2026 benchmarks. 
    #### GPU Examples: 
       - RTX 4090 ≈ `95–100` 
       - RTX 3070 ≈ `70–80`  
       - GTX 1050 ≈ `20–30`
    
    2. ### **Intensity Score** (shown in catalog highlights) 
    #### How demanding a game's requirements are overall (Higher scores means it needs better hardware).    
       - GPU Requirement:     **50%** Weight
       - CPU Requirement:     **30%** Weight
       - RAM Requirement:     **20%** Weight
    
    
    3. ### **Confidence Score** (when checking a game or recommendations)  
    #### Machine learning prediction of how well your PC can run the game smoothly:  
       - `≥70%` : Very likely to run well at decent settings  
       - `50–70%` : Probably yes, but may need lower settings/resolution  
       - `<50%` : Unlikely to run smoothly
    
    ### **Top 20 List**  
    Sorted by Steam user recommendations (most popular first) — only games your system is predicted to run.
    """)
    
    if st.button("Close", type="primary", use_container_width=True):
        st.rerun()

# === Three help buttons side-by-side ===
help_col1, help_col2, help_col3, help_col4 = st.columns([1, 1, 1, 1])

with help_col1:
    if st.button("How to find my components?", use_container_width=True):
        show_components_help()

with help_col2:
    if st.button("How to use this site?", use_container_width=True):
        show_usage_help()

with help_col3:
    if st.button("How to read the results?", use_container_width=True):
        show_results_help()

# Optional: empty column for spacing if you want them more centered
with help_col4:
    st.write("")  # Keeps symmetric spacing

# === Now the main hardware input section ===
st.markdown("### Your CPU, GPU and RAM")

hw_col1, hw_col2 = st.columns(2)

with hw_col1:
    st.subheader("CPU")
    cpu_manufacturer = st.selectbox("Manufacturer (optional)", ["Any"] + cpu_manufacturers, key="cpu_manu")
    if cpu_manufacturer != "Any":
        filtered_df = df_cpu[df_cpu['manufacturer'].str.title() == cpu_manufacturer.title()]
        filtered_cpu_models = filtered_df['cpu_name'].tolist()
        display_models = []
        for cpu_model_name in filtered_cpu_models:
            clean_model = cpu_model_name
            if cpu_manufacturer.lower() == 'amd':
                clean_model = re.sub(r'^amd\s+', '', cpu_model_name, flags=re.IGNORECASE)
            elif cpu_manufacturer.lower() == 'intel':
                clean_model = re.sub(r'^intel\s*(core\s*)?', '', cpu_model_name, flags=re.IGNORECASE)
            display_models.append(clean_model.title())
        model_to_key = dict(zip(display_models, filtered_cpu_models))
        selected_display = st.selectbox("Select or type model", options=["Suggested Models List..."] + display_models, key="cpu_model_select")
        cpu_input = model_to_key.get(selected_display, selected_display) if selected_display != "Suggested Models List..." else st.text_input("Enter your CPU model", placeholder="e.g., Ryzen 7 7800X3D", key="cpu_custom")
    else:
        cpu_input = st.text_input("Your CPU", placeholder="e.g., Ryzen 7 7800X3D, i7-13700K", key="cpu_text")

with hw_col2:
    st.subheader("GPU")
    gpu_manufacturer = st.selectbox("Manufacturer (optional)", ["Any"] + gpu_manufacturers, key="gpu_manu")
    if gpu_manufacturer != "Any":
        filtered_df = df_gpu[df_gpu['manufacturer'].str.title() == gpu_manufacturer.title()]
        filtered_gpu_models = filtered_df['gpu_name'].tolist()
        display_models = []
        for gpu_model_name in filtered_gpu_models:
            clean_model = gpu_model_name
            if gpu_manufacturer.lower() == 'nvidia':
                clean_model = re.sub(r'^geforce\s+', '', gpu_model_name, flags=re.IGNORECASE)
            elif gpu_manufacturer.lower() == 'amd':
                clean_model = re.sub(r'^radeon\s+', '', gpu_model_name, flags=re.IGNORECASE)
            display_models.append(clean_model.title())
        model_to_key = dict(zip(display_models, filtered_gpu_models))
        selected_display = st.selectbox("Select or type model", options=["Suggested Models List..."] + display_models, key="gpu_model_select")
        gpu_input = model_to_key.get(selected_display, selected_display) if selected_display != "Suggested Models List..." else st.text_input("Enter your GPU model", placeholder="e.g., GeForce RTX 4090", key="gpu_custom")
    else:
        gpu_input = st.text_input("Your GPU", placeholder="e.g., GeForce RTX 4070, GTX 1660 Super", key="gpu_text")

st.markdown("---")
ram_gb = st.slider("RAM (GB)", min_value=4, max_value=128, value=32, step=4, format="%d GB")
st.markdown(
    """
    <style>
    .stSlider [data-testid="stTickBar"] > div {display: flex !important; justify-content: space-between;}
    .stSlider [data-testid="stTickBar"] > div > div {font-size: 12px; color: #888;}
    </style>
    """, unsafe_allow_html=True
)
st.markdown("---")

st.subheader("Can my system run this Steam Game?")
game_url = st.text_input("Enter Steam Game URL", placeholder="e.g., https://store.steampowered.com/app/1903340/Clair_Obscur_Expedition_33/", label_visibility="collapsed")
st.markdown("---")

btn_col1, btn_col2, _ = st.columns([1, 1, 2])
with btn_col1:
    recommend_pressed = st.button("Get Single Steam Game Analysis and/or Steam Games Recommendations", type="primary", use_container_width=True)
with btn_col2:
    req_pressed = st.button("Just Get Single Steam Game Requirements", use_container_width=True)

# =============================================================================
# Results Section
# =============================================================================
if recommend_pressed or req_pressed:
    has_hardware = bool(cpu_input.strip() or gpu_input.strip())
    has_url = bool(game_url.strip())
    
    if not has_hardware and not has_url:
        st.error("Please enter your hardware specs and/or a Steam game URL.")
        st.stop()
    
    user_cpu = get_perf_score(cpu_input, cpu_lookup)
    user_gpu = get_perf_score(gpu_input, gpu_lookup)
    
    if user_cpu == 0 and cpu_input:
        st.warning(f"CPU '{cpu_input}' not recognized — treated as low-end.")
    if user_gpu == 0 and gpu_input:
        st.warning(f"GPU '{gpu_input}' not recognized — treated as low-end.")
    
    # Sticky specs bar
    spec_col1, spec_col2 = st.columns([5, 1])
    with spec_col1:
        st.markdown(
            f"""
            <div style="color: #ffffff; font-size: 1.15em; padding: 12px 0;">
                <strong>Your Specs:</strong>&nbsp;&nbsp;
                CPU: <strong>{cpu_input or 'Not specified'}</strong> ({user_cpu:.1f}/100)&nbsp;&nbsp;|&nbsp;&nbsp;
                GPU: <strong>{gpu_input or 'Not specified'}</strong> ({user_gpu:.1f}/100)&nbsp;&nbsp;|&nbsp;&nbsp;
                RAM: <strong>{ram_gb} GB</strong>
            </div>
            """,
            unsafe_allow_html=True
        )
    with spec_col2:
        if st.button("Change Specs", key="change_specs"):
            st.experimental_rerun()
    
    st.markdown(
        """
        <style>
        div[data-testid="column"]:nth-child(1) > div:first-child {
            position: sticky;
            top: 0;
            background: rgba(14, 17, 23, 0.95);
            backdrop-filter: blur(10px);
            z-index: 999;
            padding: 10px 0;
            border-bottom: 3px solid #333;
            box-shadow: 0 4px 20px rgba(0,0,0,0.6);
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    
    st.markdown("---")
    
    # Single game check
    if has_url:
        match = re.search(r'/app/(\d+)/', game_url)
        if match:
            appid = int(match.group(1))
            if has_hardware:
                result = check_specific_game(appid, user_cpu, user_gpu, ram_gb)
                if result and result[0]:
                    name, thumb_url, conf_min, conf_rec, min_cpu, min_gpu, min_ram, rec_cpu, rec_gpu, rec_ram = result
                    st.subheader(f"Can my system run {name}?")
                    conf_min_pct = conf_min * 100
                    conf_rec_pct = conf_rec * 100
                    yes_no_min = "Yes" if conf_min >= 0.5 else "No"
                    yes_no_rec = "Yes" if conf_rec >= 0.5 else "No"
                    html = f"""
                    <div style="padding: 20px; border: 1px solid #444; border-radius: 12px; background-color: #1e1e1e; color: #e0e0e0;">
                        <div style="display: flex; gap: 20px; align-items: start;">
                            <img src="{thumb_url}" width="250" style="border-radius: 8px;">
                            <div style="flex: 1;">
                                <h3 style="color: #ffffff; margin-top: 0;"><a href="{game_url}" target="_blank" style="color: #58a6ff; text-decoration: none;">{name}</a></h3>
                                <p><strong>Minimum Specs:</strong> <span style="color: {'#0f9' if conf_min >= 0.5 else '#f66'}; font-weight: bold;">{yes_no_min}</span>
                                (Confidence: {conf_min_pct:.1f}%)</p>
                                <ul style="color: #cccccc;">
                                    <li>CPU: {min_cpu or '—'}</li>
                                    <li>GPU: {min_gpu or '—'}</li>
                                    <li>RAM: {min_ram or '—'}</li>
                                </ul>
                                <p><strong>Recommended Specs:</strong> <span style="color: {'#0f9' if conf_rec >= 0.5 else '#f66'}; font-weight: bold;">{yes_no_rec}</span>
                                (Confidence: {conf_rec_pct:.1f}%)</p>
                                <ul style="color: #cccccc;">
                                    <li>CPU: {rec_cpu or '—'}</li>
                                    <li>GPU: {rec_gpu or '—'}</li>
                                    <li>RAM: {rec_ram or '—'}</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                    """
                    st.markdown(html, unsafe_allow_html=True)
            else:
                name, thumb_url, min_req, rec_req = fetch_game_details(appid)
                if name:
                    st.subheader(f"{name} System Requirements")
                    html = f"""
                    <div style="padding: 20px; border: 1px solid #444; border-radius: 12px; background-color: #1e1e1e; color: #e0e0e0;">
                        <div style="display: flex; gap: 20px; align-items: start;">
                            <img src="{thumb_url}" width="250" style="border-radius: 8px;">
                            <div style="flex: 1;">
                                <h3 style="color: #ffffff; margin-top: 0;"><a href="{game_url}" target="_blank" style="color: #58a6ff; text-decoration: none;">{name}</a></h3>
                                <h4 style="color: #bbbbbb;">Minimum Requirements</h4>
                                <ul style="color: #cccccc;">
                                    <li><strong>CPU:</strong> {min_req['processor'] or '—'}</li>
                                    <li><strong>GPU:</strong> {min_req['graphics'] or '—'}</li>
                                    <li><strong>RAM:</strong> {min_req['memory'] or '—'}</li>
                                </ul>
                                <h4 style="color: #bbbbbb;">Recommended Requirements</h4>
                                <ul style="color: #cccccc;">
                                    <li><strong>CPU:</strong> {rec_req['processor'] or '—'}</li>
                                    <li><strong>GPU:</strong> {rec_req['graphics'] or '—'}</li>
                                    <li><strong>RAM:</strong> {rec_req['memory'] or '—'}</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                    """
                    st.markdown(html, unsafe_allow_html=True)
        else:
            st.error("Sorry, that's an invalid Steam store URL. Please try again.")
        st.markdown("---")
    
    # Top 20 Most Popular Games You Can Run
    if has_hardware and recommend_pressed:
        features = pd.DataFrame({
            'delta_cpu': user_cpu - df_valid['cpu_score'].values,
            'delta_gpu': user_gpu - df_valid['gpu_score'].values,
            'delta_ram': ram_gb - df_valid['ram_gb_final'].values,
            'user_cpu': np.full(len(df_valid), user_cpu),
            'user_gpu': np.full(len(df_valid), user_gpu),
            'user_ram': np.full(len(df_valid), ram_gb)
        })
        df_valid['can_run'] = model.predict(features.values)
        df_valid['confidence'] = model.predict_proba(features.values)[:, 1]
        
        runnable = df_valid[df_valid['can_run'] == 1].copy()
        
        # Sort by popularity (recommendations_total) descending, take top 20
        runnable = runnable.sort_values('recommendations_total', ascending=False, na_position='last').head(20)
        
        total_games = len(df_valid[df_valid['can_run'] == 1])
        st.success(f"You can likely run **{total_games:,}** games from the Steam catalog!")
        
        st.subheader("Top 20 Most Popular Games You Can Run")
        
        st.info("Sorted by Steam user recommendations (most popular first). This helps surface real, well-known titles over obscure or joke entries.")
        
        left_col, right_col = st.columns(2)
        for i, (_, row) in enumerate(runnable.iterrows()):
            col = left_col if i % 2 == 0 else right_col
            with col:
                appid = row['appid']
                name = row['name']
                url = f"https://store.steampowered.com/app/{appid}/"
                thumb = get_header_url(appid)
                conf_pct = row['confidence'] * 100
                
                # Safe handling of recommendations_total
                if pd.isna(row['recommendations_total']):
                    recs_display = "Unknown"
                else:
                    recs_display = f"{int(row['recommendations_total']):,} players"
                
                min_cpu = '' if pd.isna(row.get('mat_pc_processor_min')) else row['mat_pc_processor_min']
                min_gpu = '' if pd.isna(row.get('mat_pc_graphics_min')) else row['mat_pc_graphics_min']
                min_ram = f"{row['ram_gb_final']:.0f} GB" if pd.notna(row['ram_gb_final']) and row['ram_gb_final'] > 0 else ''
                
                rec_cpu = '' if pd.isna(row.get('mat_pc_processor_rec')) else row['mat_pc_processor_rec']
                rec_gpu = '' if pd.isna(row.get('mat_pc_graphics_rec')) else row['mat_pc_graphics_rec']
                rec_ram = '' if pd.isna(row.get('mat_pc_memory_rec')) else row['mat_pc_memory_rec']
                if not rec_ram and pd.notna(row['ram_gb_final']) and row['ram_gb_final'] > 0:
                    rec_ram = f"{row['ram_gb_final']:.0f} GB"
                
                card = f"""
                <div style="background: #1e1e1e; border: 1px solid #444; border-radius: 12px; padding: 16px; margin-bottom: 20px; display: flex; gap: 18px;">
                    <img src="{thumb}" width="250" height="125" style="border-radius: 8px; object-fit: cover;">
                    <div style="flex: 1; color: #e0e0e0;">
                        <a href="{url}" target="_blank" style="font-size: 1.3em; color: #58a6ff; text-decoration: none;">{name}</a><br><br>
                        <strong>Recommendations:</strong> {recs_display}&nbsp;&nbsp;|&nbsp;&nbsp;<strong>Confidence:</strong> {conf_pct:.1f}%<br><br>
                        <strong>Minimum:</strong><br>
                        • CPU: {min_cpu or '—'}<br>
                        • GPU: {min_gpu or '—'}<br>
                        • RAM: {min_ram or '—'}<br><br>
                        <strong>Recommended:</strong><br>
                        • CPU: {rec_cpu or '—'}<br>
                        • GPU: {rec_gpu or '—'}<br>
                        • RAM: {rec_ram or '—'}
                    </div>
                </div>
                """
                st.markdown(card, unsafe_allow_html=True)
        
        st.markdown("---")

# Collapsible global lists — starts expanded on load, collapses after results
with st.expander("Catalog Highlights (Top 10 Most & Least Demanding Games)", expanded=not (recommend_pressed or req_pressed)):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 10 Most Demanding Steam Games (Overall)")
        for _, row in df_valid.nlargest(10, 'intensity').iterrows():
            appid = row['appid']
            name = row['name']
            url = f"https://store.steampowered.com/app/{appid}/"
            thumb = get_header_url(appid)
            card = f"""
            <div style="display: flex; gap: 15px; margin-bottom: 15px; background: #1e1e1e; padding: 12px; border-radius: 8px;">
                <img src="{thumb}" width="140" style="border-radius: 6px;">
                <div style="color: #e0e0e0;">
                    <a href="{url}" target="_blank" style="color: #58a6ff; text-decoration: none;">{name}</a><br>
                    Intensity: <strong>{row['intensity']:.1f}</strong>
                </div>
            </div>
            """
            st.markdown(card, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Top 10 Least Demanding Steam Games")
        for _, row in df_valid.nsmallest(10, 'intensity').iterrows():
            appid = row['appid']
            name = row['name']
            url = f"https://store.steampowered.com/app/{appid}/"
            thumb = get_header_url(appid)
            card = f"""
            <div style="display: flex; gap: 15px; margin-bottom: 15px; background: #1e1e1e; padding: 12px; border-radius: 8px;">
                <img src="{thumb}" width="140" style="border-radius: 6px;">
                <div style="color: #e0e0e0;">
                    <a href="{url}" target="_blank" style="color: #58a6ff; text-decoration: none;">{name}</a><br>
                    Intensity: <strong>{row['intensity']:.1f}</strong>
                </div>
            </div>
            """
            st.markdown(card, unsafe_allow_html=True)

st.caption("Built with Tom's Hardware 2026 benchmarks • vintagedon Steam dataset • scikit-learn • Streamlit")
st.caption("Full project README at: https://github.com/Tgill1085/MLZoomCamp2025/tree/main/steamhardwarerec#readme")
