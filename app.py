# app.py
from flask import Flask, render_template, request
import pandas as pd
from surprise.dump import load
import os
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
app = Flask(__name__)

# === LOAD DATA ===
OUTPUT_DIR = "output"
interactions = pd.read_csv(f"{OUTPUT_DIR}/interactions_filtered.csv")
item_map = pd.read_csv(f"{OUTPUT_DIR}/item_id_map.csv", index_col=0)['food_id'].to_dict()
item_map_rev = {v: k for k, v in item_map.items()}
_, model = load(f"{OUTPUT_DIR}/svd_model.pkl")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def parse_query(query):
    if not query.strip():
        return {}
    prompt = f"Extract JSON: {{max_calories, keywords}} from: '{query}'"
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"OpenAI error: {e}")
        return {}

def get_recommendations(user_id, query="", top_n=10):
    results = []
    filters = parse_query(query)
    print(f"User: {user_id}, Query: '{query}'")
    print(f"Parsed filters: {filters}")

    preds = []
    for food_id in interactions['food_id'].unique():
        pred = model.predict(user_id, food_id)
        base_rating = pred.est

        # === BOOST KNOWN FOODS ===
        boost = 0.0
        if food_id in ["319874", "1234567"]:
            boost = 2.0

        final_score = base_rating + boost
        preds.append((food_id, final_score))

    preds.sort(key=lambda x: x[1], reverse=True)
    candidates = preds[:50]
    print(f"Top {len(candidates)} candidates loaded.")

    for food_id, final_score in candidates:
        row = interactions[interactions['food_id'] == food_id].iloc[0]
        name = row['food_name']
        cal = row.get("Energy (KCAL)")

        print(f"Checking: {name} | Cal: {cal}")

        # === CALORIE FILTER ===
        if filters.get("max_calories") is not None:
            if pd.notna(cal) and cal > filters["max_calories"]:
                continue

        # === KEYWORD FILTER (RELAXED + SYNONYMS) ===
        if filters.get("keywords"):
            name_lower = str(name).lower()
            matched = any(k.lower() in name_lower for k in filters["keywords"])
            if not matched:
                synonyms = {
                    "healthy": ["low cal", "light", "lean", "organic", "natural", "fresh", "whole", "choice"],
                    "lunch": ["bowl", "salad", "wrap", "sandwich", "meal", "hummus"]
                }
                for k in filters["keywords"]:
                    if any(syn in name_lower for syn in synonyms.get(k.lower(), [])):
                        matched = True
                        break
            if matched:
                continue

        results.append((food_id, name, round(final_score, 2), cal))
        if len(results) >= top_n:
            break

    print(f"Final recs: {len(results)}")
    return results

@app.route("/", methods=["GET", "POST"])
def index():
    recs = []
    user_id = 42
    query = ""

    if request.method == "POST":
        user_id = int(request.form["user_id"])
        query = request.form["query"]
        recs = get_recommendations(user_id, query)

    return render_template("index.html", recs=recs, user_id=user_id, query=query)

if __name__ == "__main__":
    app.run(debug=True)