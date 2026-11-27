from pathlib import Path
import pandas as pd

# Path resolution to allow execution from project root or churndataset folder
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

INPUT_PATH = DATA_DIR / "curated_telco_churn.csv"
OUTPUT_X_PATH = DATA_DIR / "X_final.csv"
OUTPUT_Y_PATH = DATA_DIR / "y_final.csv"

print("=" * 65)
print("FINAL DATASET PREPARATION (FROM CURATED CHURN DATA)")
print("=" * 65)

# Load master curated dataset
df = pd.read_csv(INPUT_PATH)
print(f"Loaded Master Curated Dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")

# Columns to exclude from training features
exclude_from_training = ["Satisfaction Score", "Satisfaction_Risk_Tier"]
target_column = "Churn Label"

X = df.drop(columns=[target_column] + exclude_from_training)
y = df[[target_column]]

print(f"\nFinal Feature Matrix X Shape: {X.shape} (35 predictors)")
print(f"Final Target Vector y Shape:   {y.shape}")

# Verify exclusion
assert "Satisfaction Score" not in X.columns
assert "Satisfaction_Risk_Tier" not in X.columns

# Save synchronized final datasets
X.to_csv(OUTPUT_X_PATH, index=False)
y.to_csv(OUTPUT_Y_PATH, index=False)

print(f"\nSaved X_final.csv to: {OUTPUT_X_PATH}")
print(f"Saved y_final.csv to: {OUTPUT_Y_PATH}")
print("Final Dataset Preparation Completed Successfully!")
