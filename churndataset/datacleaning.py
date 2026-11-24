
import pandas as pd

# ==========================================
# PHASE 3: DATA CLEANING
# ==========================================

print("\n========== PHASE 3: DATA CLEANING ==========")

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("./data/telco.csv")

print("\nOriginal Dataset Shape:")
print(df.shape)


# ==========================================
# 2. MISSING VALUE HANDLING
# ==========================================

print("\n========== MISSING VALUE HANDLING ==========")

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Context-based missing value handling

if "Offer" in df.columns:
    df["Offer"] = df["Offer"].fillna("No Offer")

if "Internet Type" in df.columns:
    df["Internet Type"] = df["Internet Type"].fillna("No Internet")

if "Churn Category" in df.columns:
    df["Churn Category"] = df["Churn Category"].fillna("Not Applicable")

if "Churn Reason" in df.columns:
    df["Churn Reason"] = df["Churn Reason"].fillna("Not Applicable")

print("\nMissing Values After Cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\nMissing Value Handling Completed.")


# ==========================================
# 3. DUPLICATE HANDLING
# ==========================================

print("\n========== DUPLICATE HANDLING ==========")

duplicate_count = df.duplicated().sum()

print("\nDuplicate Rows Before Removal:")
print(duplicate_count)

df = df.drop_duplicates()

print("\nDuplicate Rows After Removal:")
print(df.duplicated().sum())

print("\nDataset Shape After Duplicate Removal:")
print(df.shape)


# ==========================================
# 4. DATA TYPE INVESTIGATION
# ==========================================

print("\n========== DATA TYPE INVESTIGATION ==========")

print("\nData Types:")
print(df.dtypes)

print("\nData Type Investigation Completed.")


# ==========================================
# 5. CATEGORICAL DATA CONSISTENCY
# ==========================================

print("\n========== CATEGORICAL DATA CONSISTENCY ==========")

categorical_columns = df.select_dtypes(include="str").columns

# Whitespace investigation

print("\nWhitespace Analysis:")

for column in categorical_columns:

    whitespace_count = df[column].dropna().apply(
        lambda value: value != value.strip()
    ).sum()

    print(f"{column}: {whitespace_count} whitespace issues")

# Standardize whitespace

for column in categorical_columns:

    df[column] = df[column].str.strip()

print("\nWhitespace Standardization Completed.")

# Category distribution investigation

print("\nCategorical Distribution Analysis:")

for column in categorical_columns:

    print(f"\n{column}:")

    print(df[column].value_counts(dropna=False))

print("\nCategorical Consistency Validation Completed.")


# ==========================================
# 6. NUMERICAL DATA VALIDATION
# ==========================================

print("\n========== NUMERICAL VALIDATION ==========")

numerical_columns = df.select_dtypes(include="number").columns

print("\nNumerical Summary:")
print(df[numerical_columns].describe().round(2))

# Negative value investigation

print("\nNegative Value Investigation:")

for column in numerical_columns:

    negative_count = (df[column] < 0).sum()

    if negative_count > 0:

        print(f"{column}: {negative_count} negative values")

# Age validation

if "Age" in df.columns:

    print("\nAge Validation:")

    print("Minimum Age:", df["Age"].min())
    print("Maximum Age:", df["Age"].max())
    print("Age Below 18:", (df["Age"] < 18).sum())
    print("Age Above 100:", (df["Age"] > 100).sum())

# Tenure validation

if "Tenure in Months" in df.columns:

    print("\nTenure Validation:")

    print("Minimum Tenure:", df["Tenure in Months"].min())
    print("Maximum Tenure:", df["Tenure in Months"].max())
    print("Negative Tenure:", (df["Tenure in Months"] < 0).sum())
    print("Tenure Above 72:", (df["Tenure in Months"] > 72).sum())

# Satisfaction score validation

if "Satisfaction Score" in df.columns:

    print("\nSatisfaction Score Validation:")

    print("Minimum Score:", df["Satisfaction Score"].min())
    print("Maximum Score:", df["Satisfaction Score"].max())

    print(
        "Scores Outside 1-5:",
        (
            (df["Satisfaction Score"] < 1) |
            (df["Satisfaction Score"] > 5)
        ).sum()
    )

print("\nNumerical Validation Completed.")


# ==========================================
# 7. OUTLIER INVESTIGATION
# ==========================================

print("\n========== OUTLIER INVESTIGATION ==========")

outlier_columns = [
    "Age",
    "Number of Dependents",
    "Number of Referrals",
    "Tenure in Months",
    "Avg Monthly Long Distance Charges",
    "Avg Monthly GB Download",
    "Monthly Charge",
    "Total Charges",
    "Total Refunds",
    "Total Extra Data Charges",
    "Total Long Distance Charges",
    "Total Revenue",
    "Satisfaction Score",
    "CLTV"
]

outlier_results = []

for column in outlier_columns:

    if column not in df.columns:
        continue

    Q1 = df[column].quantile(0.25)

    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - (1.5 * IQR)

    upper_bound = Q3 + (1.5 * IQR)

    outlier_count = (
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ).sum()

    outlier_results.append({
        "Column": column,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Lower Bound": lower_bound,
        "Upper Bound": upper_bound,
        "Outlier Count": outlier_count
    })

outlier_df = pd.DataFrame(outlier_results)

print("\nOutlier Analysis Report:")

if not outlier_df.empty:

    print(outlier_df.round(2).to_string(index=False))

print("\nOutlier Investigation Completed.")

# Outliers are investigated, not automatically removed.


# ==========================================
# 8. IRRELEVANT AND CONSTANT COLUMN INVESTIGATION
# ==========================================

print("\n========== IRRELEVANT COLUMN INVESTIGATION ==========")

# Constant columns

constant_columns = []

for column in df.columns:

    unique_count = df[column].nunique()

    if unique_count == 1:

        constant_columns.append(column)

print("\nConstant Columns:")
print(constant_columns)

# Identifier investigation

identifier_columns = ["Customer ID"]

print("\nIdentifier Investigation:")

for column in identifier_columns:

    if column in df.columns:

        print(f"{column} - Unique Values: {df[column].nunique()}")
        print(f"{column} - Total Rows: {len(df)}")

# High cardinality investigation

print("\nHigh Cardinality Columns:")

for column in df.columns:

    unique_count = df[column].nunique()

    unique_percentage = (unique_count / len(df)) * 100

    if unique_percentage > 50:

        print(
            f"{column}: {unique_count} unique values "
            f"({unique_percentage:.2f}%)"
        )

print("\nIrrelevant Column Investigation Completed.")


# ==========================================
# 9. DATA LEAKAGE AND IRRELEVANT COLUMN REMOVAL
# ==========================================

print("\n========== COLUMN REMOVAL ==========")

columns_to_remove = [
    "Customer ID",
    "Customer Status",
    "Churn Category",
    "Churn Reason",
    "Churn Score",
    "Country",
    "State",
    "Quarter"
]

existing_columns = [
    column for column in columns_to_remove
    if column in df.columns
]

missing_columns = [
    column for column in columns_to_remove
    if column not in df.columns
]

print("\nColumns Identified for Removal:")
print(existing_columns)

print("\nColumns Not Found:")
print(missing_columns)

df = df.drop(columns=existing_columns)

print("\nColumns Removed Successfully.")

print("\nUpdated Dataset Shape:")
print(df.shape)

print("\nRemaining Columns:")
print(df.columns.tolist())


# ==========================================
# 10. FINAL DATA VALIDATION
# ==========================================

print("\n========== FINAL DATA VALIDATION ==========")

# Dataset shape

print("\nFinal Dataset Shape:")
print(df.shape)

# Missing values

print("\nRemaining Missing Values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Duplicate rows

print("\nRemaining Duplicate Rows:")
print(df.duplicated().sum())

# Data types

print("\nFinal Data Types:")
print(df.dtypes)

# Target validation

if "Churn Label" in df.columns:

    print("\nTarget Distribution:")

    print(df["Churn Label"].value_counts())

# Final column list

print("\nFinal Dataset Columns:")
print(df.columns.tolist())

print("\nFinal Data Validation Completed.")


# ==========================================
# 11. SAVE CLEANED DATASET
# ==========================================

print("\n========== SAVING CLEANED DATASET ==========")

output_path = "./data/cleaned_telco_churn.csv"

df.to_csv(output_path, index=False)

print(f"\nCleaned Dataset Saved Successfully: {output_path}")

# ==========================================
# STEP 9: FINAL CLEANING VALIDATION
# ==========================================

print("\n========== FINAL CLEANING VALIDATION ==========")

# 1. Dataset shape

print("\nFinal Dataset Shape:")
print(df.shape)


# 2. Missing values

print("\nMissing Values:")

print(df.isnull().sum().sum())


# 3. Duplicate rows

print("\nDuplicate Rows:")

print(df.duplicated().sum())


# 4. Target validation

print("\nTarget Distribution:")

print(df["Churn Label"].value_counts())


# 5. Final column list

print("\nFinal Columns:")

print(df.columns.tolist())


# 6. Data types

print("\nFinal Data Types:")

print(df.dtypes)


# 7. Save cleaned dataset

output_path = "./data/cleaned_telco_churn.csv"

df.to_csv(output_path, index=False)

print(f"\nCleaned Dataset Saved Successfully: {output_path}")


print("\n========== DATA CLEANING COMPLETED ==========")
