# meal_planner/models/recommender.py
import pandas as pd
import os

# CORRECT PATH: from models/ → up → data/
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "meals.csv")

# Debug: Print path
print(f"Loading meals from: {DATA_PATH}")

try:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} meals")
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    print("Check: Does data/meals.csv exist?")
    raise

def get_meal_plan(prefs, total_calories):
    breakfast_cal = int(total_calories * 0.3)
    lunch_cal = int(total_calories * 0.4)
    dinner_cal = int(total_calories * 0.3)

    meals = []
    filtered = df.copy()

    if prefs:
        pattern = '|'.join(prefs)
        filtered = filtered[filtered['tags'].str.contains(pattern, case=False, na=False)]

    targets = [
        ("Breakfast", breakfast_cal),
        ("Lunch", lunch_cal),
        ("Dinner", dinner_cal)
    ]

    for meal_type, target_cal in targets:
        candidates = filtered[
            (filtered['meal_type'] == meal_type) &
            (filtered['Energy (KCAL)'] <= target_cal + 150)
        ]
        if candidates.empty:
            candidates = filtered[filtered['Energy (KCAL)'] <= target_cal + 300]
        if not candidates.empty:
            meal = candidates.sample(1).iloc[0]
            meals.append({
                'id': str(meal.get('food_id', 'N/A')),
                'type': meal_type,
                'name': meal['food_name'],
                'calories': int(meal['Energy (KCAL)']),
                'tags': [t.strip() for t in str(meal['tags']).split(",") if t.strip()]
            })
        else:
            meals.append({
                'id': 'N/A',
                'type': meal_type,
                'name': "No option found",
                'calories': 0,
                'tags': []
            })

    return meals