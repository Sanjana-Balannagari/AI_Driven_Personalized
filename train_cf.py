# train_cf.py
import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from collections import defaultdict
import os

# ================================
# CONFIGURATION
# ================================
OUTPUT_DIR = "output"
INTERACTIONS_CSV = os.path.join(OUTPUT_DIR, "interactions_filtered.csv")  # We'll save this
USER_MAP_CSV = os.path.join(OUTPUT_DIR, "user_id_map.csv")
ITEM_MAP_CSV = os.path.join(OUTPUT_DIR, "item_id_map.csv")

# Model settings
TEST_SIZE = 0.2
RANDOM_STATE = 42
TOP_N = 10

print("Starting Collaborative Filtering with Surprise...\n")

# ================================
# 1. Load Preprocessed Data
# ================================
print("Loading preprocessed interaction data...")

# Load filtered interactions (we saved this during preprocessing)
if not os.path.exists(INTERACTIONS_CSV):
    raise FileNotFoundError(f"Run preprocess.py first! Missing: {INTERACTIONS_CSV}")

interactions = pd.read_csv(INTERACTIONS_CSV)
print(f"Loaded {len(interactions)} interactions.")
if len(interactions) == 0:
    raise ValueError("interactions_filtered.csv is empty – check your filter settings in preprocess.py")

# Load ID mappings
#user_map = pd.read_csv(USER_MAP_CSV, index_col=0, squeeze=True).to_dict()
#item_map = pd.read_csv(ITEM_MAP_CSV, index_col=0, squeeze=True).to_dict()
#item_map_rev = {v: k for k, v in item_map.items()}  # idx → original food_id

#print(f"Users: {len(user_map)}, Items: {len(item_map)}")

def read_map(csv_path):
    """Read a 1-column map CSV safely."""
    df = pd.read_csv(csv_path, index_col=0)
    return df.iloc[:, 0].to_dict()

user_map = read_map(USER_MAP_CSV)
item_map = read_map(ITEM_MAP_CSV)

item_map_rev = {v: k for k, v in item_map.items()}   # idx → original food_id

print(f"Users: {len(user_map)}, Items: {len(item_map)}")

# ================================
# 2. Prepare Surprise Dataset
# ================================
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(
    interactions[["user_id", "food_id", "rating"]],  # Must be: user, item, rating
    reader
)

trainset, testset = train_test_split(data, test_size=TEST_SIZE, random_state=RANDOM_STATE)
print(f"Trainset: {trainset.n_ratings} ratings, Testset: {len(testset)} ratings")

# ================================
# 3. Train SVD Model
# ================================
print("\nTraining SVD model...")
model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=RANDOM_STATE)
model.fit(trainset)

# ================================
# 4. Evaluate (RMSE)
# ================================
print("Evaluating on test set...")
predictions = model.test(testset)
rmse = accuracy.rmse(predictions, verbose=True)

# ================================
# 5. Generate Top-N Recommendations
# ================================
def get_top_n_recommendations(model, user_id, n=TOP_N, item_map_rev=item_map_rev):
    """Get top-N predicted items for a user."""
    # Get all item IDs
    all_item_ids = list(item_map_rev.keys())
    
    # Predict rating for all items
    preds = []
    for item_idx in all_item_ids:
        food_id = item_map_rev[item_idx]
        pred = model.predict(user_id, food_id)
        preds.append((food_id, pred.est))
    
    # Sort by estimated rating
    preds.sort(key=lambda x: x[1], reverse=True)
    return preds[:n]

# Example: Recommend for a random user
example_user = np.random.choice(list(user_map.keys()))
top_n = get_top_n_recommendations(model, example_user, n=TOP_N)

print(f"\nTop {TOP_N} Recommendations for User {example_user}:")
for food_id, est in top_n:
    food_name = interactions[interactions['food_id'] == food_id]['food_name'].iloc[0]
    print(f"   • {food_name} (ID: {food_id}) → Predicted Rating: {est:.2f}")

# ================================
# 6. Save Model (Optional)
# ================================
from surprise.dump import dump
MODEL_PATH = os.path.join(OUTPUT_DIR, "svd_model.pkl")
dump(MODEL_PATH, algo=model)
print(f"\nModel saved to: {MODEL_PATH}")