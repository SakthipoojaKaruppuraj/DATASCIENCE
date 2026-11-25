# featurecreation.py

import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

df = pd.read_csv("./data/cleaned_telco_churn.csv")

print("Dataset loaded successfully")
print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. Create Tenure_Group
# --------------------------------------------------

def create_tenure_group(tenure):

    if tenure <= 12:
        return "New"
    elif tenure <= 24:
        return "Early"
    elif tenure <= 48:
        return "Mid-Term"
    else:
        return "Long-Term"


df["Tenure_Group"] = df["Tenure in Months"].apply(create_tenure_group)


# --------------------------------------------------
# 3. Create Total_Additional_Services
# --------------------------------------------------

additional_service_columns = [
    "Online Security",
    "Online Backup",
    "Device Protection Plan",
    "Premium Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Streaming Music"
]

# Count only services marked as Yes
df["Total_Additional_Services"] = (
    df[additional_service_columns]
    .apply(lambda col: col.astype(str).str.strip().str.lower() == "yes")
    .sum(axis=1)
)


# --------------------------------------------------
# 4. Create Support_Service_Count
# --------------------------------------------------

support_service_columns = [
    "Premium Tech Support",
    "Online Security",
    "Online Backup"
]

df["Support_Service_Count"] = (
    df[support_service_columns]
    .apply(lambda col: col.astype(str).str.strip().str.lower() == "yes")
    .sum(axis=1)
)


# --------------------------------------------------
# 5. Create Extra_Charge_Ratio
# --------------------------------------------------

df["Extra_Charge_Ratio"] = np.where(
    df["Total Charges"] != 0,
    df["Total Extra Data Charges"] / df["Total Charges"],
    0
)


# --------------------------------------------------
# 6. Validate New Features
# --------------------------------------------------

new_features = [
    "Tenure_Group",
    "Total_Additional_Services",
    "Support_Service_Count",
    "Extra_Charge_Ratio"
]

print("\nNew Feature Validation")
print("=" * 50)

print(df[new_features].head())

print("\nMissing Values:")
print(df[new_features].isnull().sum())

print("\nFeature Statistics:")
print(df[new_features].describe(include="all"))


# --------------------------------------------------
# 7. Save Feature-Engineered Dataset
# --------------------------------------------------

df.to_csv("./data/feature_engineered_telco_churn.csv", index=False)

print("\nFeature creation completed successfully!")
print("Dataset saved as feature_engineered_telco_churn.csv")
print("Final dataset shape:", df.shape)