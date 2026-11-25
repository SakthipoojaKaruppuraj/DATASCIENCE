import pandas as pd

# Load final datasets
X = pd.read_csv("data/X_final.csv")
y = pd.read_csv("data/y_final.csv")

print("=" * 50)
print("MODEL DEVELOPMENT - DATA VALIDATION")
print("=" * 50)

# 1. Dataset shape
print("\n1. Dataset Shape")
print("X Shape:", X.shape)
print("y Shape:", y.shape)

# 2. Missing values
print("\n2. Missing Values in X")
print(X.isnull().sum()[X.isnull().sum() > 0])

print("\nMissing Values in y:", y.isnull().sum().sum())

# 3. Duplicate records
print("\n3. Duplicate Records")
print("Duplicates in X:", X.duplicated().sum())

# 4. Data types
print("\n4. Data Types")
print(X.dtypes)

# 5. Target distribution
print("\n5. Target Distribution")
print(y["Churn Label"].value_counts())

print("\nTarget Distribution in Percentage")
print(y["Churn Label"].value_counts(normalize=True) * 100)

# 6. Categorical columns
categorical_columns = X.select_dtypes(include=["object"]).columns

print("\n6. Categorical Columns")
print(list(categorical_columns))

# 7. Numerical columns
numerical_columns = X.select_dtypes(include=["number"]).columns

print("\n7. Numerical Columns")
print(list(numerical_columns))

print("\nValidation Completed!")