# train_cf.py
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.dump import dump
import os

OUTPUT_DIR = "output"
INTERACTIONS_CSV = os.path.join(OUTPUT_DIR, "interactions_filtered.csv")
MODEL_PATH = os.path.join(OUTPUT_DIR, "svd_model.pkl")
ITEM_MAP_PATH = os.path.join(OUTPUT_DIR, "item_id_map.csv")

# === LOAD INTERACTIONS ===
interactions = pd.read_csv(INTERACTIONS_CSV)
print(f"Loaded {len(interactions)} interactions")

# === BUILD DATASET FROM INTERACTIONS ===
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(interactions[['user_id', 'food_id', 'rating']], reader)
trainset = data.build_full_trainset()

# === TRAIN MODEL ===
model = SVD(random_state=42)
model.fit(trainset)
print("Model trained")

# === SAVE MODEL ===
dump(MODEL_PATH, algo=model)

# === SAVE ITEM MAP (MUST MATCH interactions['food_id']) ===
unique_food_ids = interactions['food_id'].unique()
item_map = {i: food_id for i, food_id in enumerate(unique_food_ids)}
item_map_df = pd.DataFrame.from_dict(item_map, orient='index', columns=['food_id'])
item_map_df.index.name = 'item_idx'
item_map_df.to_csv(ITEM_MAP_PATH)
print(f"Saved item map with {len(item_map)} items → {ITEM_MAP_PATH}")