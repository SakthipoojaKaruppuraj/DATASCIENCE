
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Step 1: Load Dataset
df = pd.read_csv("dataset/StudentPerformanceFactors.csv")

print("Original Dataset Shape:", df.shape)

# Step 2: Check Missing Values
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Step 3: Handle Missing Values
for column in df.columns:

    if pd.api.types.is_numeric_dtype(df[column]):
        df[column] = df[column].fillna(df[column].median())

    else:
        df[column] = df[column].fillna(df[column].mode()[0])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Step 4: Remove Duplicates
df.drop_duplicates(inplace=True)

print("\nDataset Shape After Removing Duplicates:")
print(df.shape)

# Step 5: Separate Features and Target
X = df.drop("Exam_Score", axis=1)
y = df["Exam_Score"]

# Step 6: Encode Categorical Features
X = pd.get_dummies(X, drop_first=True, dtype=int)

print("\nEncoded Features:")
print(X.head())

# Step 7: Feature Scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(
    X_scaled,
    columns=X.columns
)

print("\nScaled Features:")
print(X_scaled.head())

# Step 8: Save Processed Dataset
processed_df = X_scaled.copy()

processed_df["Exam_Score"] = y.values

processed_df.to_csv(
    "dataset/processed_student_performance.csv",
    index=False
)

print("\nPreprocessing Completed Successfully!")
print("Final Dataset Shape:", processed_df.shape)
