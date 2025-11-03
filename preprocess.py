# preprocess.py
import pandas as pd
import numpy as np
import os

# === CONFIG ===
DATA_PATH = "data/food.csv"
FOOD_NUTRIENT_PATH = "data/food_nutrient.csv"
NUTRIENT_PATH = "data/nutrient.csv"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
INTERACTIONS_CSV = os.path.join(OUTPUT_DIR, "interactions_filtered.csv")

# === LOAD FOOD DATA ===
print(f"Loading food.csv from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Original rows: {len(df)}")

# === LOAD FOOD_NUTRIENT ===
print("Loading food_nutrient.csv...")
food_nutrients = pd.read_csv(FOOD_NUTRIENT_PATH, low_memory=False)
print(f"food_nutrient rows: {len(food_nutrients)}")

# === LOAD NUTRIENT DEFINITIONS ===
print("Loading nutrient.csv...")
nutrient_df = pd.read_csv(NUTRIENT_PATH, low_memory=False)

# === MERGE CALORIES (nutrient_id = 1008 → Energy KCAL) ===
print("\nMerging calories (nutrient_id = 1008 for Energy in KCAL)...")
df['fdc_id'] = df['fdc_id'].astype(str)
food_nutrients['fdc_id'] = food_nutrients['fdc_id'].astype(str)

energy_data = food_nutrients[food_nutrients['nutrient_id'] == 1008]
print(f"Energy (1008) rows: {len(energy_data)}")
energy_map = dict(zip(energy_data['fdc_id'], energy_data['amount']))
df['Energy (KCAL)'] = df['fdc_id'].map(energy_map).fillna(np.nan)
print(f"Added Energy (KCAL): {df['Energy (KCAL)'].notna().sum()} values")

# === ADD PROTEIN, FAT, CARBS ===
for nid, col in {1003: "Protein (G)", 1004: "Fat (G)", 1005: "Carbs (G)"}.items():
    data = food_nutrients[food_nutrients['nutrient_id'] == nid]
    if not data.empty:
        mapping = dict(zip(data['fdc_id'], data['amount']))
        df[col] = df['fdc_id'].map(mapping).fillna(np.nan)
        print(f"Added {col}: {df[col].notna().sum()} values")

# === ENSURE FOOD NAME ===
df['food_name'] = df['description'].astype(str).str.title()

# === SIMULATE INTERACTIONS ===
np.random.seed(42)
n_users = 500
n_interactions = 12_000
sampled = df.sample(n=n_interactions, replace=True).reset_index(drop=True)

interactions = pd.DataFrame({
    'user_id'   : np.random.randint(1, n_users + 1, size=n_interactions),
    'food_id'   : sampled['fdc_id'].values,
    'food_name' : sampled['food_name'].values,
    'rating'    : np.random.randint(1, 6, size=n_interactions),
})

# Add nutrients
nutrient_cols = ['Energy (KCAL)', 'Protein (G)', 'Fat (G)', 'Carbs (G)']
for col in nutrient_cols:
    if col in df.columns:
        interactions[col] = sampled[col].values

# Add category
interactions['category'] = sampled['food_category_id'].astype(str)

# === DEDUPLICATE ===
interactions = interactions.drop_duplicates(subset=['user_id', 'food_id'])
print(f"After deduplication: {len(interactions)}")

# === SAVE ===
cols_to_save = ['user_id', 'food_id', 'rating', 'food_name'] + nutrient_cols + ['category']
filtered_interactions = interactions[cols_to_save].copy()
filtered_interactions.to_csv(INTERACTIONS_CSV, index=False)
print(f"Saved {len(cols_to_save)} columns → {INTERACTIONS_CSV}")

# After merging calories
print(f"Added Energy (KCAL): {df['Energy (KCAL)'].notna().sum()} values")
