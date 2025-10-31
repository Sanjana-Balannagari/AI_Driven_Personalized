# openai_query.py
import os
import json
import pandas as pd
from openai import OpenAI
from surprise.dump import load
from dotenv import load_dotenv

# ------------------------------------------------------------------
# Load environment variables
# ------------------------------------------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    api_key = input("Enter your OpenAI API key: ").strip()
    if not api_key:
        raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or create .env")

client = OpenAI(api_key=api_key)
print("OpenAI client ready.\n")

# ------------------------------------------------------------------
# Load CF data & model
# ------------------------------------------------------------------
OUTPUT_DIR = "output"
INTERACTIONS_PATH = os.path.join(OUTPUT_DIR, "interactions_filtered.csv")
MODEL_PATH        = os.path.join(OUTPUT_DIR, "svd_model.pkl")
ITEM_MAP_PATH     = os.path.join(OUTPUT_DIR, "item_id_map.csv")

interactions = pd.read_csv(INTERACTIONS_PATH)
item_map = pd.read_csv(ITEM_MAP_PATH, index_col=0).iloc[:, 0].to_dict()
item_map_rev = {v: k for k, v in item_map.items()}
food_lookup = dict(zip(interactions['food_id'], interactions['food_name']))

_, model = load(MODEL_PATH)

# ------------------------------------------------------------------
# 1. Parse query with OpenAI
# ------------------------------------------------------------------
def parse_query_with_openai(query: str) -> dict:
    prompt = f"""
    Extract structured filters from the user query and return ONLY a JSON object:
    - max_calories: number or null
    - meal_type: "breakfast"|"lunch"|"dinner"|"snack"|null
    - diet: "healthy"|"vegetarian"|"keto"|"low_carb"|null
    - keywords: list of food names (e.g. ["chicken","salmon"])

    Query: "{query}"
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

# ------------------------------------------------------------------
# 2. Get filtered recommendations
# ------------------------------------------------------------------
def get_recommendations_with_query(user_id: int, query: str, top_n: int = 10):
    # Raw SVD predictions (top 50)
    preds = []
    for item_idx, food_id in item_map_rev.items():
        pred = model.predict(user_id, food_id)
        preds.append((food_id, pred.est))
    preds.sort(key=lambda x: x[1], reverse=True)
    candidates = preds[:50]

    filters = parse_query_with_openai(query)
    print("Parsed filters:", filters)

    results = []
    for food_id, rating in candidates:
        row = interactions[interactions['food_id'] == food_id].iloc[0]
        name = food_lookup.get(food_id, "Unknown food")

        # Calorie filter — only apply if calorie data exists
        if filters.get("max_calories") is not None:
            cal = row.get("calories")
            if pd.notna(cal) and cal > filters["max_calories"]:
                continue
            # If cal is NaN, we *allow* it (since we don't know)

        if filters.get("keywords"):
            name_lower = name.lower()
            if not any(k.lower() in name_lower for k in filters["keywords"]):
                continue

        results.append((food_id, rating, name))
        if len(results) >= top_n:
            break

    return results

# ------------------------------------------------------------------
# TEST
# ------------------------------------------------------------------
if __name__ == "__main__":
    test_user = 42
    test_query = "healthy lunch under 600 calories with chicken"
    recs = get_recommendations_with_query(test_user, test_query, top_n=5)

    print(f"\nRecommendations for User {test_user}: '{test_query}'")
    for i, (fid, rating, name) in enumerate(recs, 1):
        print(f"{i}. {name} (ID: {fid}) → {rating:.2f}")