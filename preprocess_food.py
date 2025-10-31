import pandas as pd
import os

# Set folder path where USDA CSVs are extracted
folder_path = "data"

try:
    # Load data with low_memory disabled
    food = pd.read_csv(os.path.join(folder_path, "food.csv"), low_memory=False)
    food_nutrient = pd.read_csv(os.path.join(folder_path, "food_nutrient.csv"), low_memory=False)
    nutrients = pd.read_csv(os.path.join(folder_path, "nutrient.csv"), low_memory=False)
except FileNotFoundError as e:
    print(f"Error: One or more CSV files not found in {folder_path}. Please check the folder path and file names.")
    raise
except Exception as e:
    print(f"Error loading CSV files: {e}")
    raise

# Nutrient IDs we care about
nutrient_map = {
    1008: "calories",   # Energy (kcal)
    1003: "protein",    # Protein (g)
    1004: "fat",        # Total lipid (fat) (g)
    1005: "carbs"       # Carbohydrate, by difference (g)
}

# Filter food_nutrient for only the specified nutrient IDs
filtered_fn = food_nutrient[food_nutrient["nutrient_id"].isin(nutrient_map.keys())]

# Pivot to wide format: one row per fdc_id
pivot_df = filtered_fn.pivot_table(
    index="fdc_id",
    columns="nutrient_id",
    values="amount",
    aggfunc="first"
).reset_index()

# Rename columns using nutrient_map
pivot_df.rename(columns=nutrient_map, inplace=True)

# Merge with food names
merged = food[["fdc_id", "description"]].rename(columns={"description": "food_name"})
df = merged.merge(pivot_df, on="fdc_id", how="left")

# Drop rows missing key nutrient values
df = df.dropna(subset=["calories", "protein", "fat", "carbs"])

# Save cleaned dataset
output_file = "cleaned_dataset.csv"
try:
    df.to_csv(output_file, index=False)
    print(f"Cleaned dataset saved as '{output_file}'")
    print(f"Dataset shape: {df.shape}")
    print(df.head())
except Exception as e:
    print(f"Error saving cleaned dataset: {e}")
    raise