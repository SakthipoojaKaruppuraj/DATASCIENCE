
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import math

# --------------------------------------------------
# 1. PATH SETUP
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cleaned_telco_churn.csv"

OUTPUT_DIR = BASE_DIR / "statistical_eda" / "plots" / "bivariate_target"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Remove previous output files
for file in OUTPUT_DIR.iterdir():
    if file.is_file():
        file.unlink()

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

TARGET = "Churn Label"

print("Dataset Loaded:", df.shape)

# --------------------------------------------------
# 3. SELECT FEATURES
# --------------------------------------------------

excluded_cols = [
    "Customer ID",
    "City",
    "Zip Code",
    "Latitude",
    "Longitude",
    "Population"
]

# Numerical features
numerical_cols = df.select_dtypes(
    include=["number"]
).columns.tolist()

numerical_cols = [
    col for col in numerical_cols
    if col not in excluded_cols
]

# Categorical features
categorical_cols = df.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

categorical_cols = [
    col for col in categorical_cols
    if col not in excluded_cols and col != TARGET
]

print("\nNumerical Features:")
print(numerical_cols)

print("\nCategorical Features:")
print(categorical_cols)

# --------------------------------------------------
# 4. NUMERICAL FEATURES VS TARGET
# --------------------------------------------------

numerical_target_summary = df.groupby(
    TARGET
)[numerical_cols].agg(
    ["mean", "median", "std"]
)

numerical_target_summary.to_csv(
    OUTPUT_DIR / "numerical_target_summary.csv"
)

print("\nNumerical Feature Summary by Churn:")
print(numerical_target_summary)

# --------------------------------------------------
# 5. NUMERICAL BOXPLOTS BY TARGET
# --------------------------------------------------

n_cols = 3
n_rows = math.ceil(len(numerical_cols) / n_cols)

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(18, 5 * n_rows)
)

axes = np.array(axes).flatten()

for i, col in enumerate(numerical_cols):

    sns.boxplot(
        data=df,
        x=TARGET,
        y=col,
        ax=axes[i]
    )

    axes[i].set_title(f"{col} vs {TARGET}")
    axes[i].set_xlabel("Churn Label")
    axes[i].set_ylabel(col)

# Remove unused axes
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "numerical_features_vs_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 6. CATEGORICAL FEATURES VS TARGET
# --------------------------------------------------

categorical_target_results = []

for col in categorical_cols:

    temp = df[[col, TARGET]].dropna()

    # Percentage of churn/non-churn within each category
    percentage_table = pd.crosstab(
        temp[col],
        temp[TARGET],
        normalize="index"
    ) * 100

    percentage_table = percentage_table.reset_index()

    percentage_table["feature"] = col

    categorical_target_results.append(
        percentage_table
    )

categorical_target_df = pd.concat(
    categorical_target_results,
    ignore_index=True
)

categorical_target_df.to_csv(
    OUTPUT_DIR / "categorical_target_percentage_summary.csv",
    index=False
)

print("\nCategorical Target Percentage Summary:")
print(categorical_target_df.head(20))

# --------------------------------------------------
# 7. CATEGORICAL TARGET VISUALIZATION
# --------------------------------------------------

n_cols = 3
n_rows = math.ceil(len(categorical_cols) / n_cols)

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(18, 5 * n_rows)
)

axes = np.array(axes).flatten()

for i, col in enumerate(categorical_cols):

    temp = df[[col, TARGET]].dropna()

    percentage_table = pd.crosstab(
        temp[col],
        temp[TARGET],
        normalize="index"
    ) * 100

    percentage_table.plot(
        kind="bar",
        stacked=True,
        ax=axes[i]
    )

    axes[i].set_title(f"{col} vs {TARGET}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Customer Percentage (%)")

    axes[i].tick_params(
        axis="x",
        rotation=45
    )

    axes[i].legend(
        title=TARGET,
        loc="best"
    )

# Remove unused axes
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "categorical_features_vs_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 8. FINAL OUTPUT
# --------------------------------------------------

print("\nBivariate Target Analysis Completed!")

print("\nGenerated Files:")

for file in OUTPUT_DIR.iterdir():
    print(file.name)

print(f"\nOutput Folder: {OUTPUT_DIR}")
