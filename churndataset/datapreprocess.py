import pandas as pd
import numpy as np

df = pd.read_csv("/Users/apple/projects/Datascience/churndataset/data/telco.csv")

print(df.head())

print("dataset shape: " ,df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

for col in df.select_dtypes(include="object").columns:
    print(f"\n{col}:")
    print(df[col].unique())

print("\nMissing Value Analysis:")

missing = df.isnull().sum()

missing_percentage = (missing / len(df)) * 100

missing_df = pd.DataFrame({
    "Missing Count": missing,
    "Missing Percentage": missing_percentage
})

print(missing_df[missing_df["Missing Count"] > 0])

print("\nCustomer Status Distribution:")
print(df["Customer Status"].value_counts())

print("\nChurn Label Distribution:")
print(df["Churn Label"].value_counts())

print("\nCustomer Status vs Churn Category:")
print(pd.crosstab(
    df["Customer Status"],
    df["Churn Category"],
    dropna=False
))

print("\nCustomer Status vs Churn Reason:")
print(pd.crosstab(
    df["Customer Status"],
    df["Churn Reason"],
    dropna=False
))

