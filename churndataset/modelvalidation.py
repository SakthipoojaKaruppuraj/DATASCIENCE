from pathlib import Path
import pandas as pd
import numpy as np

# Path resolution to allow execution from any working directory
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

print("=" * 65)
print("MODEL DEVELOPMENT - DATA INTEGRITY & SPLIT VALIDATION")
print("=" * 65)

# 1. Load full and split datasets
X_final = pd.read_csv(DATA_DIR / "X_final.csv")
y_final = pd.read_csv(DATA_DIR / "y_final.csv")
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").iloc[:, 0]
y_test = pd.read_csv(DATA_DIR / "y_test.csv").iloc[:, 0]

# 2. Dataset shapes validation
print("\n1. Dataset Dimensions & Partition Verification")
print("-" * 50)
print(f"X_final Shape: {X_final.shape} (Full Features)")
print(f"y_final Shape: {y_final.shape} (Full Target)")
print(f"X_train Shape: {X_train.shape} (80% Train Features)")
print(f"X_test Shape:  {X_test.shape} (20% Test Features)")
print(f"y_train Shape: {y_train.shape} (Train Target)")
print(f"y_test Shape:  {y_test.shape} (Test Target)")

assert X_train.shape[0] + X_test.shape[0] == X_final.shape[0], "Row count mismatch between full and split sets!"
assert X_train.shape[1] == 35, f"Expected 35 features, got {X_train.shape[1]}"
assert X_test.shape[1] == 35, f"Expected 35 features, got {X_test.shape[1]}"
assert X_final.shape[1] == 35, f"Expected 35 features in X_final, got {X_final.shape[1]}"
print(" Partition shapes verified successfully (35 features)!")

# 3. Explicit Target Leakage Check
print("\n2. Target Leakage Verification")
print("-" * 50)
leakage_candidates = ["Satisfaction Score", "Satisfaction_Risk_Tier", "Churn Label", "Customer Status", "Churn Score"]
for col in leakage_candidates:
    assert col not in X_train.columns, f"LEAKAGE DETECTED: {col} found in X_train!"
    assert col not in X_test.columns, f"LEAKAGE DETECTED: {col} found in X_test!"
    assert col not in X_final.columns, f"LEAKAGE DETECTED: {col} found in X_final!"
print(" Zero target leakage confirmed: Satisfaction Score and Risk Tier are strictly excluded from X.")

# 4. Missing values check
print("\n3. Missing Value Audit")
print("-" * 50)
train_nulls = X_train.isnull().sum().sum()
test_nulls = X_test.isnull().sum().sum()
y_train_nulls = y_train.isnull().sum()
y_test_nulls = y_test.isnull().sum()

print(f"Missing values in X_train: {train_nulls}")
print(f"Missing values in X_test:  {test_nulls}")
print(f"Missing values in y_train: {y_train_nulls}")
print(f"Missing values in y_test:  {y_test_nulls}")
assert train_nulls == 0 and test_nulls == 0, "Null values detected in feature matrices!"
print(" Zero missing values confirmed across all splits!")

# 5. Duplicate records check
print("\n4. Duplicate Records Audit")
print("-" * 50)
train_dups = X_train.duplicated().sum()
test_dups = X_test.duplicated().sum()
print(f"Duplicate rows in X_train: {train_dups}")
print(f"Duplicate rows in X_test:  {test_dups}")

# 6. Feature types inventory (Categorical vs Numerical)
print("\n5. Feature Types & Schema Inventory")
print("-" * 50)
categorical_columns = X_train.select_dtypes(include=["object", "str", "category"]).columns.tolist()
numerical_columns = X_train.select_dtypes(include=["number"]).columns.tolist()

print(f"Total Clean Features: {X_train.shape[1]}")
print(f"Categorical Features: {len(categorical_columns)}")
print(f"Numerical Features:   {len(numerical_columns)}")

print("\nCategorical Features (21):")
for i, col in enumerate(categorical_columns, 1):
    print(f"  {i:2d}. {col}")

print("\nNumerical Features (14):")
for i, col in enumerate(numerical_columns, 1):
    print(f"  {i:2d}. {col}")

assert len(categorical_columns) + len(numerical_columns) == 35, "Feature type classification incomplete!"
assert len(categorical_columns) == 21, f"Expected 21 categorical features, got {len(categorical_columns)}"
assert len(numerical_columns) == 14, f"Expected 14 numerical features, got {len(numerical_columns)}"
print("\n Feature schema exactly matches 35 clean pre-churn predictors!")

# 7. Target distribution & stratification validation
print("\n6. Target Stratification & Class Balance")
print("-" * 50)
train_churn_pct = (y_train == 1).mean() * 100
test_churn_pct = (y_test == 1).mean() * 100

print(f"Training Set Churn: {y_train.sum():,} / {len(y_train):,} ({train_churn_pct:.2f}%)")
print(f"Testing Set Churn:  {y_test.sum():,} / {len(y_test):,} ({test_churn_pct:.2f}%)")
assert abs(train_churn_pct - test_churn_pct) < 0.1, "Stratification drift detected between train and test sets!"
print(" Stratification perfectly preserved across train and test sets!")

# 8. Check infinite values in ratios
print("\n7. Numerical Health & Infinite Value Audit")
print("-" * 50)
inf_train = np.isinf(X_train[numerical_columns].values).sum()
inf_test = np.isinf(X_test[numerical_columns].values).sum()
print(f"Infinite values in X_train numericals: {inf_train}")
print(f"Infinite values in X_test numericals:  {inf_test}")
assert inf_train == 0 and inf_test == 0, "Infinite values found in engineered ratios!"
print(" Zero infinite values found!")

print("\n" + "=" * 65)
print("ALL DATA INTEGRITY & VALIDATION CHECKS PASSED SUCCESSFULLY!")
print("=" * 65)
