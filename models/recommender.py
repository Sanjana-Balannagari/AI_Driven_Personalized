# models/recommender.py
import pandas as pd
import os
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "meals.csv")
df = pd.read_csv(DATA_PATH)

_seen_ingredients = set()

def extract_ingredients(name):
    name = str(name).lower()
    common = ['chicken', 'salmon', 'tofu', 'hummus', 'quinoa', 'rice', 'pasta', 'egg', 'oat']
    return [ing for ing in common if ing in name]

def get_meal_plan(prefs, total_calories, k=5):
    global _seen_ingredients
    _seen_ingredients = set()

    targets = [
        ("Breakfast", int(total_calories * 0.3)),
        ("Lunch", int(total_calories * 0.4)),
        ("Dinner", int(total_calories * 0.3))
    ]

    meals = []
    filtered = df.copy()

    # === SAFE TAG HANDLING ===
    filtered['tags'] = filtered['tags'].fillna('').astype(str)

    # === 1. Filter by preferences ===
    prefs_lower = [p.lower().strip() for p in prefs]  # ← Define here

    if prefs:
        def has_exact_tags(tag_str):
            if not tag_str:
                return False
            tags = [t.strip().lower() for t in tag_str.split(',')]
            return all(p in tags for p in prefs_lower)

        exact = filtered[filtered['tags'].apply(has_exact_tags)]
        partial = filtered[filtered['tags'].str.contains('|'.join(prefs_lower), case=False, na=False)]
        filtered = pd.concat([exact, partial]).drop_duplicates()

    recommended_ids = []

    for meal_type, target_cal in targets:
        lower = target_cal * 0.9
        upper = target_cal * 1.1

        candidates = filtered[
            (filtered['meal_type'] == meal_type) &
            (filtered['Energy (KCAL)'] >= lower) &
            (filtered['Energy (KCAL)'] <= upper)
        ]

        # === SCORING — PASS prefs_lower ===
        def score_row(row, prefs=prefs_lower):  # ← Pass as default arg
            ingredients = extract_ingredients(row['food_name'])
            repeat_penalty = sum(1 for ing in ingredients if ing in _seen_ingredients)
            tag_bonus = sum(1 for p in prefs if p in row['tags'].lower()) * 10
            return tag_bonus - repeat_penalty * 5

        if not candidates.empty:
            candidates = candidates.copy()
            candidates['score'] = candidates.apply(score_row, axis=1)
            candidates = candidates.sort_values('score', ascending=False)

            selected = None
            for _, row in candidates.iterrows():
                ingredients = extract_ingredients(row['food_name'])
                if not any(ing in _seen_ingredients for ing in ingredients):
                    selected = row
                    _seen_ingredients.update(ingredients)
                    break
            if selected is None:
                selected = candidates.iloc[0]
                _seen_ingredients.update(extract_ingredients(selected['food_name']))
        else:
            fallback = filtered[
                (filtered['meal_type'] == meal_type) &
                (filtered['Energy (KCAL)'] <= target_cal * 1.3)
            ]
            selected = fallback.sample(1).iloc[0] if not fallback.empty else None

        if selected is not None:
            meal = {
                'id': str(selected['food_id']),
                'type': meal_type,
                'name': selected['food_name'],
                'calories': int(selected['Energy (KCAL)']),
                'tags': [t.strip() for t in str(selected['tags']).split(",") if t.strip()]
            }
            meals.append(meal)
            recommended_ids.append(meal['id'])
        else:
            meals.append({
                'id': 'N/A', 'type': meal_type, 'name': "No option", 'calories': 0, 'tags': []
            })

    return meals, recommended_ids

def precision_at_k(recommended, relevant, k=5):
    rec_k = recommended[:k]  
    hits = len(set(rec_k) & set(relevant))
    return hits / len(rec_k) if rec_k else 0.0