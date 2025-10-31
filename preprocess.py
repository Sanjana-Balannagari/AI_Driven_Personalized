# preprocess.py
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz
import os

# ================================
# CONFIGURATION
# ================================
DATA_PATH = "data/food.csv"  # <-- Updated path
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

INTERACTIONS_CSV = os.path.join(OUTPUT_DIR, "interactions_filtered.csv")
MATRIX_CSV       = os.path.join(OUTPUT_DIR, "user_item_matrix.csv")
SPARSE_NPZ       = os.path.join(OUTPUT_DIR, "user_item_sparse.npz")
USER_MAP_CSV     = os.path.join(OUTPUT_DIR, "user_id_map.csv")
ITEM_MAP_CSV     = os.path.join(OUTPUT_DIR, "item_id_map.csv")

MAX_CALORIES       = 600
CATEGORIES_TO_KEEP = ["Vegetables", "Fruits", "Grains", "Proteins"]

print(f"Loading dataset from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Original rows: {df.shape[0]}, columns: {list(df.columns)}")

# ------------------------------------------------------------------
# 1. Detect columns
# ------------------------------------------------------------------
def find_col(candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

food_id_col   = find_col(['fdc_id', 'NDB_number', 'id'])
food_name_col = find_col(['description', 'food_description', 'name', 'food_name'])
calories_col  = find_col(['Energy_kcal', 'calories', 'energy_kcal'])
category_col  = find_col(['food_category', 'category', 'food_group', 'food_category_id'])

if food_id_col is None:
    raise KeyError("No food ID column found.")

print(f"Detected → id:{food_id_col}, name:{food_name_col}, calories:{calories_col}, category:{category_col}")

# ------------------------------------------------------------------
# 2. Simulate interactions
# ------------------------------------------------------------------
np.random.seed(42)
n_users = 500
n_interactions = 12_000

sampled = df.sample(n=n_interactions, replace=True).reset_index(drop=True)

interactions = pd.DataFrame({
    'user_id'   : np.random.randint(1, n_users + 1, size=n_interactions),
    'food_id'   : sampled[food_id_col].values,
    'food_name' : sampled[food_name_col].values if food_name_col else sampled[food_id_col].values,
    'rating'    : np.random.randint(1, 6, size=n_interactions),
})

if calories_col:
    interactions['calories'] = sampled[calories_col].values
else:
    interactions['calories'] = np.nan

if category_col:
    interactions['category'] = sampled[category_col].values
else:
    interactions['category'] = 'Unknown'

print(f"Simulated {n_interactions} interactions.")

# ------------------------------------------------------------------
# 3. Filter
# ------------------------------------------------------------------
mask = pd.Series([True] * len(interactions))

if MAX_CALORIES is not None and calories_col:
    mask &= interactions['calories'] < MAX_CALORIES

if CATEGORIES_TO_KEEP and category_col:
    mask &= interactions['category'].isin(CATEGORIES_TO_KEEP)

filtered = interactions[mask].copy()
print(f"After filter: {len(filtered)} interactions.")

if len(filtered) == 0:
    print("Warning: No rows survived – using all data.")
    filtered = interactions.copy()

# ------------------------------------------------------------------
# 4. DEDUPLICATE + BUILD MATRIX
# ------------------------------------------------------------------
dedup = filtered.drop_duplicates(subset=['user_id', 'food_id'], keep='first')
print(f"After deduplication: {len(dedup)} unique interactions.")

user_to_idx = {u: i for i, u in enumerate(sorted(dedup['user_id'].unique()))}
item_to_idx = {i: j for j, i in enumerate(sorted(dedup['food_id'].unique()))}

dedup = dedup.copy()
dedup['user_idx'] = dedup['user_id'].map(user_to_idx)
dedup['item_idx'] = dedup['food_id'].map(item_to_idx)

matrix_df = dedup.pivot(index='user_idx', columns='item_idx', values='rating').fillna(0)
sparse = csr_matrix(matrix_df.values)

print(f"Matrix shape: {matrix_df.shape} (users x items)")

# ------------------------------------------------------------------
# 5. SAVE
# ------------------------------------------------------------------
# Save for Surprise
filtered_interactions = dedup[['user_id', 'food_id', 'rating', 'food_name', 'calories', 'category']].copy()
filtered_interactions.to_csv(INTERACTIONS_CSV, index=False)
print(f"Saved → {INTERACTIONS_CSV}")

matrix_df.to_csv(MATRIX_CSV)
save_npz(SPARSE_NPZ, sparse)
pd.Series(user_to_idx).to_csv(USER_MAP_CSV)
pd.Series(item_to_idx).to_csv(ITEM_MAP_CSV)

print("\nAll done! Files in", OUTPUT_DIR)