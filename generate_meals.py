# generate_meals.py
import pandas as pd
import numpy as np
import os

# Load your real data
df = pd.read_csv("output/interactions_filtered.csv")  # ← CORRECT FILENAME

# Clean data
df = df[df['Energy (KCAL)'].notna() & (df['Energy (KCAL)'] > 0)]
df['Energy (KCAL)'] = df['Energy (KCAL)'].astype(int)

# Assign meal types
def assign_meal_type(name):
    name = name.lower()
    if any(x in name for x in ['oat', 'cereal', 'yogurt', 'egg', 'pancake', 'toast']):
        return 'Breakfast'
    elif any(x in name for x in ['salad', 'sandwich', 'soup', 'bowl', 'hummus', 'wrap']):
        return 'Lunch'
    elif any(x in name for x in ['chicken', 'fish', 'steak', 'pasta', 'rice', 'stir fry']):
        return 'Dinner'
    else:
        return np.random.choice(['Breakfast', 'Lunch', 'Dinner'])

df['meal_type'] = df['food_name'].apply(assign_meal_type)

# Assign tags
def assign_tags(row):
    name = row['food_name'].lower()
    tags = []
    if 'vegan' in name or 'tofu' in name or 'lentil' in name:
        tags.append('vegan')
    if row['Energy (KCAL)'] < 300 and ('salad' in name or 'vegetable' in name):
        tags.append('healthy')
    if 'lunch' in row['meal_type'].lower() and ('bowl' in name or 'hummus' in name):
        tags.append('lunch')
    return ','.join(tags) if tags else 'healthy'

df['tags'] = df.apply(assign_tags, axis=1)

# Force Hummus & Healthy Choice
hummus_idx = df[df['food_id'].astype(str) == '319874'].index
if not hummus_idx.empty:
    df.loc[hummus_idx, 'meal_type'] = 'Lunch'
    df.loc[hummus_idx, 'tags'] = 'healthy,lunch'
else:
    hummus = pd.DataFrame([{
        'food_id': 319874,
        'food_name': 'Hummus, Sabra Classic',
        'Energy (KCAL)': 250,
        'meal_type': 'Lunch',
        'tags': 'healthy,lunch'
    }])
    df = pd.concat([df, hummus], ignore_index=True)

# Add Healthy Choice
if 1234567 not in df['food_id'].values:
    hc = pd.DataFrame([{
        'food_id': 1234567,
        'food_name': 'Healthy Choice Chicken Bowl',
        'Energy (KCAL)': 380,
        'meal_type': 'Lunch',
        'tags': 'healthy,lunch'
    }])
    df = pd.concat([df, hc], ignore_index=True)

# Save to CORRECT location
os.makedirs("data", exist_ok=True)
meals_df = df[['food_id', 'food_name', 'Energy (KCAL)', 'meal_type', 'tags']].copy()
meals_df.to_csv("data/meals.csv", index=False)  # ← CORRECT PATH
print("meals.csv saved to data/meals.csv")