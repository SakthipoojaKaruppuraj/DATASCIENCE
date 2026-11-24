import pandas as pd

df = pd.read_csv("./data/telco.csv")


print("\n========== MISSING VALUE ANALYSIS ==========")

missing_analysis = pd.DataFrame({
    "Missing Count": df.isnull().sum(),
    "Missing Percentage": (df.isnull().sum() / len(df)) * 100
})

missing_analysis = missing_analysis[
    missing_analysis["Missing Count"] > 0
]

missing_analysis = missing_analysis.sort_values(
    by="Missing Percentage",
    ascending=False
)

print(missing_analysis)

print("\n========== 2. TARGET DISTRIBUTION ANALYSIS ==========")

print("\nTarget Counts:")

target_counts = df["Churn Label"].value_counts()

print(target_counts)

print("\nTarget Percentages:")

target_percentage = df["Churn Label"].value_counts(normalize=True) * 100

print(target_percentage.round(2))

print("\n========== 3. MISSING VALUE RELATIONSHIP ANALYSIS ==========")

print("\nCustomer Status Distribution:")

print(df["Customer Status"].value_counts(dropna=False))


print("\nCustomer Status vs Offer:")

print(pd.crosstab(
    df["Customer Status"],
    df["Offer"],
    dropna=False
))


print("\nCustomer Status vs Internet Type:")

print(pd.crosstab(
    df["Customer Status"],
    df["Internet Type"],
    dropna=False
))


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


print("\n========== 4. CONSTANT COLUMN ANALYSIS ==========")

unique_counts = df.nunique(dropna=False)

constant_columns = unique_counts[unique_counts == 1]

print("\nConstant Columns:")

print(constant_columns)


print("\nNear-Constant Columns (95% Threshold):")

near_constant_columns = []

for column in df.columns:

    value_percentage = df[column].value_counts(
        dropna=False,
        normalize=True
    )

    if len(value_percentage) > 0:

        dominant_percentage = value_percentage.iloc[0] * 100

        if dominant_percentage >= 95 and unique_counts[column] > 1:

            near_constant_columns.append({
                "Column": column,
                "Dominant Value Percentage": round(dominant_percentage, 2)
            })

near_constant_df = pd.DataFrame(near_constant_columns)

if not near_constant_df.empty:
    print(near_constant_df)
else:
    print("No near-constant columns detected.")



print("\n========== 5. POTENTIAL DATA LEAKAGE ANALYSIS ==========")

leakage_candidates = [
    "Customer ID",
    "Customer Status",
    "Churn Category",
    "Churn Reason",
    "Churn Score"
]

for column in leakage_candidates:

    print(f"\n--- {column} ---")

    print(df[column].value_counts(dropna=False).head(15))


print("\nCustomer Status vs Churn Label:")

print(pd.crosstab(
    df["Customer Status"],
    df["Churn Label"],
    dropna=False
))

print("\n========== 6. DATA CONSISTENCY ANALYSIS ==========")


numerical_columns = df.select_dtypes(include="number").columns

print("\nNumerical Range Analysis:")

for column in numerical_columns:

    print(f"\n--- {column} ---")

    print("Minimum:", df[column].min())

    print("Maximum:", df[column].max())



print("\n========== AGE VALIDATION ==========")

print("Customers below age 18:")

print(df[df["Age"] < 18].shape[0])


print("\nCustomers above age 100:")

print(df[df["Age"] > 100].shape[0])



print("\n========== NEGATIVE VALUE CHECK ==========")

for column in numerical_columns:

    negative_count = (df[column] < 0).sum()

    if negative_count > 0:

        print(f"{column}: {negative_count} negative values")

if not any((df[column] < 0).sum() > 0 for column in numerical_columns):

    print("No negative numerical values detected.")



print("\n========== CATEGORICAL VALUE ANALYSIS ==========")

categorical_columns = df.select_dtypes(
    include=["object", "str", "category"]
).columns

for column in categorical_columns:

    unique_values = df[column].dropna().unique()

    if len(unique_values) <= 10:

        print(f"\n{column}:")

        print(unique_values)



print("\n========== STATUS & TARGET VALIDATION ==========")

print(pd.crosstab(
    df["Customer Status"],
    df["Churn Label"],
    dropna=False
))


print("\n========== 7. DATA QUALITY SUMMARY ==========")

print("\nDataset Shape:")

print(df.shape)


print("\nTotal Missing Values:")

print(df.isnull().sum().sum())


print("\nDuplicate Rows:")

print(df.duplicated().sum())


print("\nConstant Columns:")

print(list(constant_columns.index))


print("\nTarget Distribution:")

print(df["Churn Label"].value_counts())


print("\n========== DATA ANALYSIS COMPLETED ==========")

