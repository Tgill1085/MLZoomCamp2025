# train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE  # For imbalance
import pickle

# Load data
df = pd.read_csv('archive/games_march2025_cleaned.csv')  # Or your data path
print(f"Loaded shape: {df.shape}")
print("Columns:", df.columns.tolist())

# Engineer target (as in notebook)
drop_leaks = ['user_score', 'score_rank', 'pct_pos_total', 'pct_pos_recent', 'num_reviews_total', 'num_reviews_recent']
df = df.drop(columns=[col for col in drop_leaks if col in df.columns])
if 'positive' in df.columns and 'negative' in df.columns:
    total_reviews = df['positive'] + df['negative']
    df['positive_ratio'] = df['positive'] / total_reviews.replace(0, np.nan)
    df['positive_ratio'] = df['positive_ratio'].fillna(0)
    df['positive'] = (df['positive_ratio'] > 0.9).astype(int)
    print("Re-engineered binary 'positive' from counts.")

# Drop low-importance: required_age
if 'required_age' in df.columns:
    df = df.drop('required_age', axis=1)
    print("Dropped required_age (low impact).")

# CRITICAL FIX: Drop leaks post-engineering (from notebook logic)
leak_cols = ['positive_ratio', 'negative', 'appid', 'name', 'recommendations', 'notes', 'estimated_owners', 
             'average_playtime_forever', 'average_playtime_2weeks', 'median_playtime_forever', 'median_playtime_2weeks',
             'peak_ccu', 'discount', 'metacritic_score', 'metacritic_url', 'score_rank']  # Post-release or ID leaks
df = df.drop(columns=[col for col in leak_cols if col in df.columns])
print(f"Dropped leaks: {len([col for col in leak_cols if col in df.columns])} columns (e.g., positive_ratio, negative, appid)")

# Feature eng: Skip genre dummies (use genre_count only for neutrality)
if 'genres' in df.columns:
    df['genre_count'] = df['genres'].str.split(',').str.len().fillna(1)
    print("Added genre_count (neutral variety proxy—no dummies to avoid bias).")
else:
    df['genre_count'] = 1
    print("Default genre_count=1.")

# Explicitly drop any existing genre dummies to ensure neutrality
genre_dummies = [col for col in df.columns if col.startswith('genre__')]
if genre_dummies:
    df = df.drop(columns=genre_dummies)
    print(f"Dropped existing genre dummies: {genre_dummies}")

# Add supported_languages and developers_count if available (parse from data)
if 'languages' in df.columns or 'supported_languages' in df.columns:
    lang_col = 'languages' if 'languages' in df.columns else 'supported_languages'
    # FIX: Parse string list safely (e.g., "['English', 'French']" -> len=2)
    def parse_lang_count(s):
        try:
            # Strip brackets/quotes, split on ', '
            cleaned = str(s).strip("[]'").replace("'", "").split(", ")
            return len(cleaned)
        except:
            return 1  # Fallback
    df['supported_languages'] = df[lang_col].apply(parse_lang_count).fillna(1)
    print("Added supported_languages count (broader reach proxy; parsed lists).")
elif 'supported_languages' in df.columns:
    df['supported_languages'] = pd.to_numeric(df['supported_languages'], errors='coerce').fillna(1)
    print("Used existing supported_languages column.")

if 'developers' in df.columns:
    df['developers_count'] = df['developers'].astype(str).str.split(',').str.len().fillna(1)
    print("Added developers_count (indie/team size proxy).")
else:
    df['developers_count'] = 1
    print("Default developers_count=1 (indie assumption).")

# Engineer platforms_count (sum win/mac/linux if present)
if all(col in df.columns for col in ['win', 'mac', 'linux']):
    df['platforms_count'] = df[['win', 'mac', 'linux']].sum(axis=1)
    print("Engineered platforms_count.")
else:
    df['platforms_count'] = 1  # Default Windows
    print("Default platforms_count=1.")

# Add tag_count if strings
if 'tags' in df.columns:
    df['tag_count'] = df['tags'].str.split(',').str.len().fillna(1)
    print("Added tag_count.")

# Release year from date
if 'release_date' in df.columns:
    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year.fillna(2025)
    df['release_year'] = np.minimum(df['release_year'], 2025)
    print("Added/capped release_year.")

# Franchise IP proxy (boost for known brands)
if 'publishers' in df.columns:
    df['big_ip'] = df['publishers'].str.contains('Nickelodeon|THQ|Paramount|EA|Ubisoft|Valve', case=False, na=False).astype(int)
    print("Added big_ip (franchise boost).")
else:
    df['big_ip'] = 0
    print("No publishers column—default big_ip=0.")

# Price tier with IP discount (reduces penalty for franchises)
df['price_tier'] = pd.cut(df['price'], bins=[0, 5, 15, 30, np.inf], labels=[0,1,2,3]).cat.codes  # 0=free/low (boost), 3=high (penalty)
df['price_ip_adjust'] = df['price_tier'] * (1 - df['big_ip'] * 0.5)  # Halve penalty if big_ip=1
print("Added price_ip_adjust (IP discounts high prices).")

# Impute achievements for new/low (median from highs ~25)
df['achievements'] = df['achievements'].fillna(25).clip(upper=100)  # Cap outliers
print("Imputed/capped achievements.")

# F2P-IP synergy (extra lift for free franchises)
df['f2p_ip_boost'] = ((df['price'] == 0) & (df['big_ip'] == 1)).astype(int)
print("Added f2p_ip_boost (F2P + IP extra).")

# Train-val-test split: 20% train, 20% val, 60% test (stratified)
n_samples = len(df)
test_size = round(0.6 * n_samples)
X_temp, X_test, y_temp, y_test = train_test_split(
    df.drop('positive', axis=1).select_dtypes(include=[np.number]).fillna(0),
    df['positive'], test_size=test_size, stratify=df['positive'], random_state=42
)
print(f"Train+Temp size: {len(X_temp)} ({len(X_temp)/n_samples:.1%}), Test size: {len(X_test)} ({len(X_test)/n_samples:.1%})")

val_size = round(0.5 * len(X_temp))  # 50% of temp = 20% val
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, train_size=val_size, stratify=y_temp, random_state=42
)
print(f"Train size: {len(X_train)} ({len(X_train)/n_samples:.1%}), Val size: {len(X_val)} ({len(X_val)/n_samples:.1%})")

# Handle imbalance with SMOTE (oversample 'high' in train only)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"Post-SMOTE Train shape: {X_train_res.shape} | Balance: {y_train_res.mean():.1%} high (balanced)")

# Scale continuous features (fit on resampled train)
scaler = StandardScaler()
continuous_cols = [
    'achievements', 'release_year', 'platforms_count', 'genre_count', 'tag_count', 'supported_languages', 'developers_count',
    'price_tier', 'price_ip_adjust'
]  # No owners_numeric (leak)
avail_cont = [c for c in continuous_cols if c in X_train_res.columns]
if avail_cont:
    X_train_res[avail_cont] = scaler.fit_transform(X_train_res[avail_cont])
    X_val[avail_cont] = scaler.transform(X_val[avail_cont])
    X_test[avail_cont] = scaler.transform(X_test[avail_cont])
    print(f"Scaled continuous: {avail_cont}")
else:
    print("No continuous columns to scale.")

feature_cols = X_train_res.columns.tolist()
print(f"Final features: {len(feature_cols)} | Sample: {feature_cols[:5]}...")

# Train XGBoost (fit on resampled)
params = {
    'n_estimators': [100, 200, 300], 
    'max_depth': [3, 6, 9], 
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}
xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
search = RandomizedSearchCV(xgb_model, params, cv=5, scoring='roc_auc', random_state=42, n_iter=20)
best_model = search.fit(X_train_res, y_train_res)
print(f"Best params: {search.best_params_}")

# Evaluate
val_proba = best_model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_proba)
print(f"Val AUC: {val_auc:.3f}")

test_proba = best_model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, test_proba)
print(f"Test AUC: {test_auc:.3f} (Realistic ~0.65 expected)")

# Feature importances plot (save for review)
importances = pd.Series(best_model.best_estimator_.feature_importances_, index=feature_cols).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=importances.values[:10], y=importances.index[:10])
plt.title('Top 10 Feature Importances')
plt.savefig('models/feature_importances.png')
plt.close()

# Save pipeline
os.makedirs('models', exist_ok=True)
pipeline = {
    'model': best_model.best_estimator_,
    'scaler': scaler,
    'features': feature_cols,
    'continuous_cols': continuous_cols
}
with open('models/full_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
print("Saved full_pipeline.pkl")