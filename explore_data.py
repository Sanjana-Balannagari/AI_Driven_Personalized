import pandas as pd

# Load the key CSV files (adjust paths if needed)
foods = pd.read_csv('data/food.csv')
nutrients = pd.read_csv('data/nutrient.csv')
data = pd.read_csv('data/foundation_food.csv')  # This is the main nutrient amount table

# Explore structure: Show first few rows
print("=== Foods Table (Food Descriptions) ===")
print(foods.head())
print("\nFoods shape:", foods.shape)
print("Foods columns:", foods.columns.tolist())

print("\n=== Nutrients Table (Nutrient List) ===")
print(nutrients.head())
print("\nNutrients shape:", nutrients.shape)
print("Nutrients columns:", nutrients.columns.tolist())

print("\n=== Data Table (Nutrient Amounts per Food) ===")
print(data.head())
print("\nData shape:", data.shape)
print("Data columns:", data.columns.tolist())

# Quick info on data types and missing values
print("\n=== Basic Info ===")
print(foods.info())
print(nutrients.info())
print(data.info())

