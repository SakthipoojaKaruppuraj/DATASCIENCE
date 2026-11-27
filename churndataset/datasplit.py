from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Path resolution to allow running from any working directory
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

INPUT_PATH = DATA_DIR / "curated_telco_churn.csv"

# Output paths
X_TRAIN_PATH = DATA_DIR / "X_train.csv"
X_TEST_PATH = DATA_DIR / "X_test.csv"
Y_TRAIN_PATH = DATA_DIR / "y_train.csv"
Y_TEST_PATH = DATA_DIR / "y_test.csv"
X_FINAL_PATH = DATA_DIR / "X_final.csv"
Y_FINAL_PATH = DATA_DIR / "y_final.csv"

print("=" * 65)
print("STRATIFIED TRAIN-TEST SPLIT (LEAKAGE-FREE TRAINING MATRIX)")
print("=" * 65)

# Load master curated dataset (keeps all 38 columns intact)
df = pd.read_csv(INPUT_PATH)
print(f"Loaded Master Curated Dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")

# Columns to exclude strictly from training inputs to prevent post-outcome exit survey leakage
exclude_from_training = ["Satisfaction Score", "Satisfaction_Risk_Tier"]
target_col = "Churn Label"

print(f"\nExcluding post-outcome leakage features from training inputs: {exclude_from_training}")

# Separate training features and target
X = df.drop(columns=[target_col] + exclude_from_training)
y = df[target_col]

# Convert target labels into binary numeric values (No: 0, Yes: 1)
y_numeric = y.map({"No": 0, "Yes": 1})

# Also save full X_final and y_final for project-wide consistency
X.to_csv(X_FINAL_PATH, index=False)
y.to_frame(name=target_col).to_csv(Y_FINAL_PATH, index=False)

# Stratified Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_numeric,
    test_size=0.20,
    random_state=42,
    stratify=y_numeric
)

# Display dataset shapes
print("\nDataset Split Summary:")
print("-" * 45)
print(f"X_train Shape: {X_train.shape} (80% Train, 35 features)")
print(f"X_test Shape:  {X_test.shape} (20% Test, 35 features)")
print(f"X_final Shape: {X.shape} (Full Population, 35 features)")
print(f"y_train Shape: {y_train.shape}")
print(f"y_test Shape:  {y_test.shape}")

# Verify satisfaction features are completely absent from X
assert "Satisfaction Score" not in X.columns, "Satisfaction Score found in X!"
assert "Satisfaction_Risk_Tier" not in X.columns, "Satisfaction_Risk_Tier found in X!"
print("\n Zero leakage verified: Satisfaction Score and Risk Tier excluded from X.")

# Display target distribution & stratification verification
print("\nTarget Class Balance Verification:")
print("-" * 45)
train_dist = y_train.value_counts(normalize=True) * 100
test_dist = y_test.value_counts(normalize=True) * 100

dist_df = pd.DataFrame({
    "Train Count": y_train.value_counts(),
    "Train %": train_dist.round(2),
    "Test Count": y_test.value_counts(),
    "Test %": test_dist.round(2)
})
print(dist_df.to_string())

# Save split datasets
X_train.to_csv(X_TRAIN_PATH, index=False)
X_test.to_csv(X_TEST_PATH, index=False)

y_train.to_csv(Y_TRAIN_PATH, index=False)
y_test.to_csv(Y_TEST_PATH, index=False)

print(f"\nFiles successfully exported to: {DATA_DIR}")
print("Train-Test Split Completed Successfully!")
