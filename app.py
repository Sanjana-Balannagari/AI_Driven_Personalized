# app.py
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pickle
import os
from surprise.dump import load
from openai_query import get_recommendations_with_query

app = Flask(__name__)

# ================================
# CONFIG
# ================================
OUTPUT_DIR = "output"
INTERACTIONS_PATH = os.path.join(OUTPUT_DIR, "interactions_filtered.csv")
MODEL_PATH = os.path.join(OUTPUT_DIR, "svd_model.pkl")
USER_MAP_PATH = os.path.join(OUTPUT_DIR, "user_id_map.csv")
ITEM_MAP_PATH = os.path.join(OUTPUT_DIR, "item_id_map.csv")

# Load data
print("Loading model and data...")
interactions = pd.read_csv(INTERACTIONS_PATH)
user_map = pd.read_csv(USER_MAP_PATH, index_col=0).iloc[:, 0].to_dict()
item_map = pd.read_csv(ITEM_MAP_PATH, index_col=0).iloc[:, 0].to_dict()
item_map_rev = {v: k for k, v in item_map.items()}

# Load trained SVD model
_, model = load(MODEL_PATH)

# Precompute food lookup
food_lookup = dict(zip(interactions['food_id'], interactions['food_name']))

# ================================
# ROUTES
# ================================
@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    selected_user = None

    if request.method == "POST":
        user_id = int(request.form["user_id"])
        query = request.form.get("query", "").strip()

        if query:
            # Use OpenAI + CF
            recs = get_recommendations_with_query(user_id, query, top_n=10)
            recommendations = [(fid, rating, name) for fid, rating, name in recs]
        else:
            # Fallback: pure CF
            preds = []
            for item_idx, food_id in item_map_rev.items():
                pred = model.predict(user_id, food_id)
                preds.append((food_id, pred.est, food_lookup.get(food_id, "Unknown")))
            preds.sort(key=lambda x: x[1], reverse=True)
            recommendations = preds[:10]

    return render_template(
        "index.html",
        recommendations=recommendations,
        selected_user=selected_user,
        users=sorted(user_map.keys())
    )

if __name__ == "__main__":
    print("Starting Flask app on http://127.0.0.1:5000")
    app.run(debug=True)