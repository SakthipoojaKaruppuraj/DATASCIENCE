import pandas as pd

# Load feature-engineered dataset
df = pd.read_csv("data/feature_engineered_telco_churn.csv")

print("Original Dataset Shape:", df.shape)

# Columns to exclude from model inputs
exclude_columns = [
    "City",
    "Zip Code",
    "Latitude",
    "Longitude",
    "Population",
    "CLTV",
    "Satisfaction Score"
]

# Separate target
target_column = "Churn Label"

X = df.drop(columns=[target_column] + exclude_columns)
y = df[target_column]

print("\nFeature Dataset Shape:", X.shape)
print("Target Dataset Shape:", y.shape)

print("\nRemaining Feature Columns:")
print(X.columns.tolist())

# Save final datasets
X.to_csv("data/X_final.csv", index=False)
y.to_csv("data/y_final.csv", index=False)

print("\nFinal datasets saved successfully!")