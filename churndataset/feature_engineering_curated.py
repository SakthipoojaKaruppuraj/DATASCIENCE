# feature_engineering_curated.py
# Production Feature Engineering and Noise Reduction Pipeline
# Strict non-redundant feature engineering without persona synthesis

import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------
# 1. PATH SETUP & DATA LOADING
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "cleaned_telco_churn.csv"
OUTPUT_PATH = BASE_DIR / "data" / "curated_telco_churn.csv"

print("=" * 70)
print("CURATED FEATURE ENGINEERING & NOISE REMOVAL PIPELINE")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
print(f"Loaded Cleaned Dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")

# --------------------------------------------------
# 2. FEATURE ENGINEERING: BUSINESS-ALIGNED LIFECYCLE GROUPS
# --------------------------------------------------

print("\n1. Engineering Tenure_Group (Capturing Non-Linear Lifecycle)...")

def create_tenure_group(tenure):
    if tenure <= 6:
        return "Onboarding"       # Months 1-6 (Extreme risk: 53.3% churn)
    elif tenure <= 12:
        return "First-Year"       # Months 7-12 (Stabilizing: 29.2% churn)
    elif tenure <= 24:
        return "Developing"       # Months 13-24 (21.7% churn)
    elif tenure <= 48:
        return "Established"      # Months 25-48 (14.2% churn)
    else:
        return "Loyal"            # Months 49+ (6.6% churn)

df["Tenure_Group"] = df["Tenure in Months"].apply(create_tenure_group)
print("   Distribution of Tenure_Group:")
print(df["Tenure_Group"].value_counts(normalize=True).round(3) * 100)

# --------------------------------------------------
# 3. FEATURE ENGINEERING: ECOSYSTEM SERVICES COUNT
# --------------------------------------------------

print("\n2. Engineering Total_Additional_Services & Support_Service_Count...")

all_additional_services = [
    "Online Security",
    "Online Backup",
    "Device Protection Plan",
    "Premium Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Streaming Music"
]

core_support_services = [
    "Premium Tech Support",
    "Online Security",
    "Online Backup"
]

# Total bundle depth (integer 0 to 7)
df["Total_Additional_Services"] = (
    df[[col for col in all_additional_services if col in df.columns]]
    .apply(lambda col: col.astype(str).str.strip().str.lower() == "yes")
    .sum(axis=1)
)

# Core support tools (integer 0 to 3)
df["Support_Service_Count"] = (
    df[[col for col in core_support_services if col in df.columns]]
    .apply(lambda col: col.astype(str).str.strip().str.lower() == "yes")
    .sum(axis=1)
)

print(f"   Average Additional Services per Customer: {df['Total_Additional_Services'].mean():.2f}")
print(f"   Average Core Support Services per Customer: {df['Support_Service_Count'].mean():.2f}")

# --------------------------------------------------
# 4. FEATURE ENGINEERING: FINANCIAL RATIOS & PRICING PACE
# --------------------------------------------------

print("\n3. Engineering Non-Redundant Financial & Value Ratios...")

# Ratio 1: Penalty overage ratio to total charges
df["Extra_Charge_Ratio"] = np.where(
    df["Total Charges"] != 0,
    df["Total Extra Data Charges"] / df["Total Charges"],
    0.0
).round(4)

# Ratio 2: Bill Hike Ratio (Current bill vs Historical Average monthly bill)
# Detects promo expirations / price surges (> 1.0 means recent bill increase)
avg_historical_charge = df["Total Charges"] / np.maximum(df["Tenure in Months"], 1)
df["Bill_Hike_Ratio"] = (df["Monthly Charge"] / avg_historical_charge).round(3)

# Ratio 3: Cost per GB Downloaded (Perceived bandwidth price efficiency)
df["Cost_Per_GB"] = (df["Monthly Charge"] / (df["Avg Monthly GB Download"] + 1)).round(2)

# Indicator: High ARPU customer bracket (>= $75/mo)
df["Is_High_ARPU"] = np.where(df["Monthly Charge"] >= 75.0, "Yes", "No")

print(f"   Extra Charge Ratio Mean: {df['Extra_Charge_Ratio'].mean():.4f}")
print(f"   Bill Hike Ratio Median: {df['Bill_Hike_Ratio'].median():.2f}")
print(f"   Cost per GB Median: ${df['Cost_Per_GB'].median():.2f}/GB")
print(f"   High-ARPU Accounts (>= $75/mo): {(df['Is_High_ARPU'] == 'Yes').sum():,} ({(df['Is_High_ARPU'] == 'Yes').mean()*100:.1f}%)")

# --------------------------------------------------
# 5. FEATURE ENGINEERING: COMPOUND RISK HOTSPOTS & BILLING
# --------------------------------------------------

print("\n4. Engineering Compound Interaction Hotspots & Sentiment Tier...")

# Hotspot 1: Month-to-Month + Fiber Optic (#1 Churn Epicenter: 58.8% Churn)
df["Is_Fiber_MonthToMonth"] = np.where(
    (df["Contract"] == "Month-to-Month") & (df["Internet Type"] == "Fiber Optic"),
    "Yes",
    "No"
)

# Hotspot 2: Fiber Optic with Zero Support Tools (58.9% Churn)
df["Is_Zero_Support_Fiber"] = np.where(
    (df["Internet Type"] == "Fiber Optic") & (df["Support_Service_Count"] == 0),
    "Yes",
    "No"
)

# Hotspot 3: Digital Billing Friction (Paperless + Bank Withdrawal: High Involuntary & Digital Exit)
df["High_Risk_Billing"] = np.where(
    (df["Paperless Billing"] == "Yes") & (df["Payment Method"] == "Bank Withdrawal"),
    "Yes",
    "No"
)

# Hotspot 4: Toxic Promotional Acquisition Channel (Offer E: 52.9% Churn)
df["Is_Toxic_Offer_E"] = np.where(df["Offer"] == "Offer E", "Yes", "No")

# Sentiment Tier: Non-linear operational risk classification
def categorize_satisfaction(score):
    if score <= 2:
        return "Critical"       # Score <= 2: 95%+ churn rate
    elif score == 3:
        return "At-Risk"        # Score 3: ~24% churn rate
    else:
        return "Safe"           # Score 4-5: < 0.5% churn rate

df["Satisfaction_Risk_Tier"] = df["Satisfaction Score"].apply(categorize_satisfaction)

print(f"   Month-to-Month Fiber Risk Accounts: {(df['Is_Fiber_MonthToMonth'] == 'Yes').sum():,} ({(df['Is_Fiber_MonthToMonth'] == 'Yes').mean()*100:.1f}%)")
print(f"   Zero-Support Fiber Accounts:        {(df['Is_Zero_Support_Fiber'] == 'Yes').sum():,} ({(df['Is_Zero_Support_Fiber'] == 'Yes').mean()*100:.1f}%)")
print(f"   High-Risk Billing Accounts:         {(df['High_Risk_Billing'] == 'Yes').sum():,} ({(df['High_Risk_Billing'] == 'Yes').mean()*100:.1f}%)")
print(f"   Offer E Accounts:                   {(df['Is_Toxic_Offer_E'] == 'Yes').sum():,} ({(df['Is_Toxic_Offer_E'] == 'Yes').mean()*100:.1f}%)")
print(f"   Critical Satisfaction Accounts:     {(df['Satisfaction_Risk_Tier'] == 'Critical').sum():,} ({(df['Satisfaction_Risk_Tier'] == 'Critical').mean()*100:.1f}%)")

# --------------------------------------------------
# 6. NOISE REMOVAL & REDUNDANCY PRUNING
# --------------------------------------------------

print("\n5. Removing Noise & Non-Predictive Features (16 Columns)...")

noise_columns = [
    # Demographics with zero predictive power (Cramer's V < 0.05, p > 0.05)
    "Gender",
    "Under 30",
    
    # Commoditized utilities with no retention signal
    "Phone Service",
    "Multiple Lines",
    
    # Redundant pass-through services (captured in Total_Additional_Services)
    "Streaming TV",
    "Streaming Movies",
    "Streaming Music",
    
    # Redundant financial subcomponents (collinear with Monthly Charge & Total Revenue)
    "Avg Monthly Long Distance Charges",
    "Total Long Distance Charges",
    "Total Refunds",
    
    # High-cardinality geographical / synthetic proxy metrics
    "City",
    "Zip Code",
    "Latitude",
    "Longitude",
    "Population",
    "CLTV"
]

existing_noise = [c for c in noise_columns if c in df.columns]
print(f"   Dropping {len(existing_noise)} noise/redundant columns:")
for col in existing_noise:
    print(f"   - {col}")

curated_df = df.drop(columns=existing_noise)

# --------------------------------------------------
# 7. FINAL VALIDATION & EXPORT
# --------------------------------------------------

print("\n" + "=" * 70)
print("6. FINAL DATASET VALIDATION")
print("=" * 70)

print(f"Original Cleaned Dataset Shape: {df.shape}")
print(f"Curated Dataset Shape:          {curated_df.shape}")
print(f"Total Features Retained:        {curated_df.shape[1] - 1} predictors + 1 target (Churn Label)")

# Null check
null_count = curated_df.isnull().sum().sum()
print(f"Remaining Missing Values:       {null_count}")

# Duplicate check
dup_count = curated_df.duplicated().sum()
print(f"Duplicate Rows:                 {dup_count}")

print("\nCurated Non-Redundant Columns Inventory (38 Columns):")
for i, col in enumerate(curated_df.columns, 1):
    print(f"  {i:2d}. {col}")

# Save
curated_df.to_csv(OUTPUT_PATH, index=False)
print(f"\n Curated Dataset Successfully Exported to: {OUTPUT_PATH}")
print("=" * 70)
