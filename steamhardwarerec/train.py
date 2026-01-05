
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

# === Check for required data ===
data_file = 'steam_games_final.csv'
if not os.path.exists(data_file):
    print(f"ERROR: {data_file} not found!")
    print("Please run the data_gathering.ipynb notebook first to generate it.")
    print("It requires:")
    print("  - Downloading the Steam dataset from Zenodo")
    print("  - Scraping Tom's Hardware benchmarks")
    print("  - Running feature engineering")
    print("\nAfter that, run this script again.")
    exit(1)

# Load enriched data
print(f"Loading {data_file}...")
df_steam = pd.read_csv(data_file)

# Filter to games with meaningful intensity
df_valid = df_steam[df_steam['intensity'] > 10].copy()
print(f"Using {len(df_valid):,} games for synthetic training")

# Generate synthetic data
num_samples = 100000
np.random.seed(42)

user_cpu = np.random.uniform(0, 100, num_samples)
user_gpu = np.random.uniform(0, 100, num_samples)
user_ram = np.random.choice([4, 8, 16, 32, 64, 128], num_samples)

game_samples = df_valid.sample(num_samples, replace=True).reset_index(drop=True)

# Features: deltas + raw user specs
X = np.column_stack([
    user_cpu - game_samples['cpu_score'],
    user_gpu - game_samples['gpu_score'],
    user_ram - game_samples['ram_gb_final'],
    user_cpu,
    user_gpu,
    user_ram
])

# Labels: can run (with small margin and 8% noise for realism)
y = np.all([
    user_cpu >= game_samples['cpu_score'] - 5,
    user_gpu >= game_samples['gpu_score'] - 5,
    user_ram >= game_samples['ram_gb_final']
], axis=0).astype(int)

noise = np.random.rand(num_samples) < 0.08
y = np.where(noise, 1 - y, y)

print(f"Synthetic data generated: {len(X):,} samples")
print(f"Label distribution: {np.bincount(y)}")

# Train/val/test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Retrain best model on full train + val
X_train_full = np.vstack([X_train, X_val])
y_train_full = np.concatenate([y_train, y_val])

print("\nRetraining final Random Forest on full training data...")
final_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=30,
    min_samples_split=2,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
final_model.fit(X_train_full, y_train_full)

# Final test evaluation
test_prob = final_model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, test_prob)
test_acc = accuracy_score(y_test, test_prob >= 0.5)

print(f"\nFINAL TEST RESULTS")
print(f"Test AUC: {test_auc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Feature importances
importances = final_model.feature_importances_
feat_names = ['delta_cpu', 'delta_gpu', 'delta_ram', 'user_cpu', 'user_gpu', 'user_ram']

plt.figure(figsize=(10, 6))
indices = np.argsort(importances)[::-1]
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [feat_names[i] for i in indices], rotation=45)
plt.title('Feature Importances - Final Model')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Feature importance plot saved as 'feature_importance.png'")
plt.show()

# Confirm model works
example = np.array([[60-80, 70-85, 16-16, 60, 70, 16]])  # Tough high-end game
pred = final_model.predict_proba(example)[0, 1]
print(f"\nExample prediction: Mid-range PC vs high-end game -> Probability can run: {pred:.1%}")

# Save model
model_path = 'can_run_model_final.pkl'
joblib.dump(final_model, model_path)
print(f"\nFinal model saved as '{model_path}'")
print("You can now run: streamlit run predict.py")
