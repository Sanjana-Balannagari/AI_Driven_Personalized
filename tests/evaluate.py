# tests/evaluate.py
import pandas as pd
import os
from models.recommender import get_meal_plan, precision_at_k

# === GROUND TRUTH AS STRINGS ===
GROUND_TRUTH = {
    "vegan_1800":                ["319874", "1234567", "987654"],
    "lowcarb_highprotein_2200": ["112233", "445566", "778899"],
    "default_1500":              ["223344", "556677", "889900"]
}

def run_tests():
    test_cases = [
        (["vegan"],                     1800, "vegan_1800"),
        (["low_carb", "high_protein"], 2200, "lowcarb_highprotein_2200"),
        ([],                           1500, "default_1500")
    ]

    print("Running Evaluation (Sept 22–24)\n" + "="*50)
    results = []

    for prefs, calories, key in test_cases:
        plan, recommended_ids = get_meal_plan(prefs, calories, k=5)
        recommended_ids = [mid for mid in recommended_ids if mid != 'N/A']

        relevant = GROUND_TRUTH[key]
        p5 = precision_at_k(recommended_ids, relevant, k=5)

        results.append((prefs, calories, p5, recommended_ids))

        print(f"Query: prefs={prefs}, calories={calories}")
        print(f"   Recommended IDs : {recommended_ids}")
        print(f"   Relevant IDs    : {relevant}")
        print(f"   Precision@5     : {p5:.3f}\n")

    avg_p5 = sum(r[2] for r in results) / len(results)
    print(f"Average Precision@5: {avg_p5:.3f}")
    return results

if __name__ == "__main__":
    run_tests()