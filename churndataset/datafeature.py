
import pandas as pd

# ==========================================
# PHASE 4: FEATURE ENGINEERING
# STEP 1: FEATURE INVESTIGATION
# ==========================================

df = pd.read_csv("./data/cleaned_telco_churn.csv")

print("\n========== FEATURE INVESTIGATION ==========")

# 1. Dataset overview

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# 2. Separate target and input features

target_column = "Churn Label"

X = df.drop(columns=[target_column])
y = df[target_column]

print("\nTarget Column:")
print(target_column)

print("\nTarget Distribution:")
print(y.value_counts())


# 3. Identify numerical features

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("\nNumerical Features:")
print(numerical_features)

print("\nNumerical Feature Count:")
print(len(numerical_features))


# 4. Identify categorical features

categorical_features = X.select_dtypes(
    include=["str"]
).columns.tolist()

print("\nCategorical Features:")
print(categorical_features)

print("\nCategorical Feature Count:")
print(len(categorical_features))


# 5. Investigate categorical feature cardinality

print("\n========== CATEGORICAL CARDINALITY ==========")

for column in categorical_features:

    unique_count = X[column].nunique()

    print(f"\nColumn: {column}")
    print(f"Unique Values: {unique_count}")
    print(X[column].value_counts().head(10))


# 6. Investigate numerical feature statistics

print("\n========== NUMERICAL FEATURE STATISTICS ==========")

print(X[numerical_features].describe().round(2))


# 7. Binary categorical feature investigation

print("\n========== BINARY FEATURE INVESTIGATION ==========")

for column in categorical_features:

    unique_values = X[column].nunique()

    if unique_values == 2:

        print(f"\n{column}")
        print(X[column].value_counts())


# ==========================================
# STEP 2: FEATURE RELEVANCE INVESTIGATION
# ==========================================

print("\n========== FEATURE RELEVANCE INVESTIGATION ==========")

# 1. Geographic feature investigation

geographic_features = [
    "City",
    "Zip Code",
    "Latitude",
    "Longitude",
    "Population"
]

print("\nGeographic Features:")

for column in geographic_features:

    if column in X.columns:

        print(f"{column}: {X[column].nunique()} unique values")


# 2. Financial feature investigation

financial_features = [
    "Monthly Charge",
    "Total Charges",
    "Total Revenue",
    "Total Refunds",
    "Total Extra Data Charges",
    "Total Long Distance Charges",
    "CLTV"
]

print("\nFinancial Features:")

for column in financial_features:

    if column in X.columns:

        print(f"\n{column}")

        print(X[column].describe().round(2))


# 3. Correlation investigation

print("\n========== NUMERICAL CORRELATION INVESTIGATION ==========")

correlation_df = df[numerical_features].corr()

print("\nCorrelation Matrix:")

print(correlation_df.round(2).to_string())


# 4. Strong correlation pairs

print("\n========== STRONG CORRELATION PAIRS ==========")

for i in range(len(numerical_features)):

    for j in range(i + 1, len(numerical_features)):

        column1 = numerical_features[i]
        column2 = numerical_features[j]

        correlation = correlation_df.loc[column1, column2]

        if abs(correlation) >= 0.80:

            print(
                f"{column1} <-> {column2}: "
                f"{correlation:.2f}"
            )


# 5. Target relationship investigation

print("\n========== TARGET RELATIONSHIP INVESTIGATION ==========")

df["Churn Binary"] = df["Churn Label"].map({
    "Yes": 1,
    "No": 0
})

for column in numerical_features:

    churn_correlation = df[column].corr(df["Churn Binary"])

    print(
        f"{column}: "
        f"{churn_correlation:.3f}"
    )


# Remove temporary target column

df.drop(columns=["Churn Binary"], inplace=True)


# ==========================================
# STEP 3: FEATURE SELECTION
# ==========================================

print("\n========== FEATURE SELECTION ==========")

# Define target

target_column = "Churn Label"

# Define columns to exclude

columns_to_drop = [
    "City",
    "Zip Code",
    "Latitude",
    "Longitude",
    "Population",
    "CLTV"
]

# Verify exclusions

columns_to_drop = [
    column for column in columns_to_drop
    if column in df.columns
]

print("\nColumns Excluded:")

print(columns_to_drop)

# Create selected dataset

selected_df = df.drop(columns=columns_to_drop)

# Separate features and target

X_selected = selected_df.drop(columns=[target_column])

y_selected = selected_df[target_column]

# Final feature investigation

print("\nSelected Dataset Shape:")

print(selected_df.shape)

print("\nSelected Feature Count:")

print(X_selected.shape[1])

print("\nSelected Features:")

print(X_selected.columns.tolist())

print("\nTarget Distribution:")

print(y_selected.value_counts())

print("\nFeature Selection Completed.")

# ==========================================
# STEP 4: CATEGORICAL FEATURE ENCODING
# ==========================================

print("\n========== CATEGORICAL FEATURE ENCODING ==========")

from sklearn.preprocessing import OneHotEncoder

# Separate categorical and numerical features
categorical_features_selected = X_selected.select_dtypes(
    include="str"
).columns.tolist()

numerical_features_selected = X_selected.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("\nCategorical Features:")
print(categorical_features_selected)

print("\nNumerical Features:")
print(numerical_features_selected)


# 1. Binary categorical feature investigation

binary_features = []

for column in categorical_features_selected:

    unique_values = X_selected[column].nunique()

    if unique_values == 2:

        binary_features.append(column)

print("\nBinary Features:")
print(binary_features)


# 2. Multi-category feature investigation

multi_category_features = []

for column in categorical_features_selected:

    unique_values = X_selected[column].nunique()

    if unique_values > 2:

        multi_category_features.append(column)

print("\nMulti-Category Features:")
print(multi_category_features)


# 3. Encoding strategy report

print("\n========== ENCODING STRATEGY ==========")

print("\nBinary Features:")
print("These can be encoded using binary mapping or One-Hot Encoding.")

print("\nMulti-Category Features:")
print("These will use One-Hot Encoding.")

print("\nNumerical Features:")
print("These will remain numerical and will be evaluated for scaling later.")


# 4. Investigate expected One-Hot Encoded columns

print("\n========== ONE-HOT ENCODING INVESTIGATION ==========")

for column in multi_category_features:

    print(f"\n{column}")

    print("Categories:")

    print(X_selected[column].unique().tolist())


print("\nCategorical Encoding Investigation Completed.")

# ==========================================
# STEP 5: NUMERICAL SCALING INVESTIGATION
# ==========================================

print("\n========== NUMERICAL SCALING INVESTIGATION ==========")

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Identify numerical features

numerical_features_selected = X_selected.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("\nNumerical Features:")
print(numerical_features_selected)


# 1. Feature range investigation

print("\n========== FEATURE RANGE ==========")

for column in numerical_features_selected:

    minimum = X_selected[column].min()

    maximum = X_selected[column].max()

    print(
        f"{column} | "
        f"Min: {minimum} | "
        f"Max: {maximum}"
    )


# 2. Scaling strategy investigation

print("\n========== SCALING STRATEGIES ==========")

print("\nStandardScaler:")
print("Transforms features to approximately mean 0 and standard deviation 1.")

print("\nMinMaxScaler:")
print("Transforms features to a specified range, usually 0 to 1.")


# 3. Investigate scaling impact

scaling_comparison = X_selected[numerical_features_selected].describe().T

scaling_comparison["Range"] = (
    scaling_comparison["max"] - scaling_comparison["min"]
)

print("\nNumerical Scaling Comparison:")

print(
    scaling_comparison[
        ["mean", "std", "min", "max", "Range"]
    ].round(2).to_string()
)


print("\nNumerical Scaling Investigation Completed.")

# ==============================
# STEP 5: TRAIN-TEST SPLIT
# ==============================

from sklearn.model_selection import train_test_split

target_column = "Churn Label"

X = selected_df.drop(columns=[target_column])
y = selected_df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== TRAIN-TEST SPLIT ==========")

print("Training Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)

print("\nTraining Target Distribution:")
print(y_train.value_counts())

print("\nTesting Target Distribution:")
print(y_test.value_counts())

# ==========================================
# STEP 6 - ENCODING AND SCALING
# ==========================================

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

print("\n========== ENCODING AND SCALING ==========")

# Identify numerical and categorical columns
numerical_columns = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

categorical_columns = X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()

print("\nNumerical Columns:", numerical_columns)
print("\nCategorical Columns:", categorical_columns)

# Create preprocessing transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_columns),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ]
)

# Fit on training data and transform
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data
X_test_processed = preprocessor.transform(X_test)

print("\nTraining Data After Processing:", X_train_processed.shape)
print("Testing Data After Processing:", X_test_processed.shape)

print("\nEncoding and Scaling Completed!")

# ==========================================
# STEP 7 - REUSABLE PREPROCESSING PIPELINE
# ==========================================

from sklearn.pipeline import Pipeline

print("\n========== REUSABLE PIPELINE ==========")

# Create reusable preprocessing pipeline
preprocessing_pipeline = Pipeline([
    ("preprocessing", preprocessor)
])

# Fit pipeline using training data
preprocessing_pipeline.fit(X_train, y_train)

# Transform training and testing data
X_train_pipeline = preprocessing_pipeline.transform(X_train)
X_test_pipeline = preprocessing_pipeline.transform(X_test)

# Validation
print("\nPipeline Training Shape:", X_train_pipeline.shape)
print("Pipeline Testing Shape:", X_test_pipeline.shape)

print("\nPipeline Steps:")
print(preprocessing_pipeline.named_steps)

print("\nPipeline Validation:")
print("Training data processed:", X_train_pipeline.shape[1] > 0)
print("Testing data processed:", X_test_pipeline.shape[1] > 0)
print("Feature count consistent:", X_train_pipeline.shape[1] == X_test_pipeline.shape[1])

print("\nReusable Pipeline Completed!")

# ==========================================
# STEP 8 - FINAL FEATURE ENGINEERING VALIDATION
# ==========================================

import numpy as np

print("\n========== FINAL FEATURE ENGINEERING VALIDATION ==========")

# Extract transformed feature names
feature_names = preprocessing_pipeline.named_steps["preprocessing"].get_feature_names_out()

# Validate feature names
print("\nTotal Transformed Features:", len(feature_names))
print("First 15 Transformed Features:")
print(feature_names[:15])

# Validate training and testing dimensions
print("\nTraining Shape:", X_train_pipeline.shape)
print("Testing Shape:", X_test_pipeline.shape)

# Check missing values in transformed data
train_array = X_train_pipeline.toarray() if hasattr(X_train_pipeline, "toarray") else X_train_pipeline
test_array = X_test_pipeline.toarray() if hasattr(X_test_pipeline, "toarray") else X_test_pipeline

print("\nMissing Values in Training Data:", np.isnan(train_array).sum())
print("Missing Values in Testing Data:", np.isnan(test_array).sum())

# Check infinity values
print("Infinity Values in Training Data:", np.isinf(train_array).sum())
print("Infinity Values in Testing Data:", np.isinf(test_array).sum())

# Feature count validation
print("\nFeature Count Match:", len(feature_names) == X_train_pipeline.shape[1])

print("\n========== FEATURE ENGINEERING VALIDATION COMPLETED ==========")