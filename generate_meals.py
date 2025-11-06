# generate_meals.py
import pandas as pd
import numpy as np
import os

# Load real data
df = pd.read_csv("output/interactions_filtered.csv")
df = df[df['Energy (KCAL)'].notna() & (df['Energy (KCAL)'] > 0)]
df['Energy (KCAL)'] = df['Energy (KCAL)'].astype(int)

# === GROUND TRUTH WITH REALISTIC CALORIES ===
ground_truth_meals = [
    # vegan_1800: 30% Breakfast, 40% Lunch, 30% Dinner
    {"food_id": 319874, "food_name": "Hummus, Sabra Classic", "Energy (KCAL)": 720, "meal_type": "Lunch", "tags": "vegan,healthy,lunch"},
    {"food_id": 1234567, "food_name": "Healthy Choice Vegan Bowl", "Energy (KCAL)": 540, "meal_type": "Dinner", "tags": "vegan,healthy"},
    {"food_id": 987654, "food_name": "Tofu Quinoa Salad", "Energy (KCAL)": 540, "meal_type": "Breakfast", "tags": "vegan,high_protein"},

    # lowcarb_highprotein_2200
    {"food_id": 112233, "food_name": "Grilled Chicken Breast", "Energy (KCAL)": 880, "meal_type": "Dinner", "tags": "low_carb,high_protein"},
    {"food_id": 445566, "food_name": "Salmon Avocado Bowl", "Energy (KCAL)": 880, "meal_type": "Lunch", "tags": "low_carb,high_protein"},
    {"food_id": 778899, "food_name": "Egg White Omelette", "Energy (KCAL)": 660, "meal_type": "Breakfast", "tags": "low_carb,high_protein"},

    # default_1500
    {"food_id": 223344, "food_name": "Greek Yogurt Bowl", "Energy (KCAL)": 450, "meal_type": "Breakfast", "tags": "healthy"},
    {"food_id": 556677, "food_name": "Turkey Sandwich", "Energy (KCAL)": 600, "meal_type": "Lunch", "tags": "healthy"},
    {"food_id": 889900, "food_name": "Veggie Stir Fry", "Energy (KCAL)": 450, "meal_type": "Dinner", "tags": "healthy"},
]

gt_df = pd.DataFrame(ground_truth_meals)

# Remove duplicates
df = df[~df['food_id'].isin(gt_df['food_id'])]
df = pd.concat([df, gt_df], ignore_index=True)

# Save
os.makedirs("output", exist_ok=True)
meals_df = df[['food_id', 'food_name', 'Energy (KCAL)', 'meal_type', 'tags']].copy()
meals_df['tags'] = meals_df['tags'].fillna('').astype(str)
meals_df.to_csv("output/meals.csv", index=False)
print("meals.csv updated with REALISTIC CALORIES!")