
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# --------------------------------------------------
# 1. PATH SETUP
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cleaned_telco_churn.csv"

OUTPUT_DIR = BASE_DIR / "statistical_eda" / "plots" / "multivariate_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Clean previous output files
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
# 3. SELECT NUMERICAL FEATURES
# --------------------------------------------------

selected_numerical_features = [
    "Age",
    "Tenure in Months",
    "Monthly Charge",
    "Total Revenue",
    "Satisfaction Score",
    "CLTV"
]

numerical_cols = [
    col for col in selected_numerical_features
    if col in df.columns
]

print("\nSelected Numerical Features:")
print(numerical_cols)

# --------------------------------------------------
# 4. MULTIVARIATE CORRELATION ANALYSIS
# --------------------------------------------------

correlation_data = df[numerical_cols].copy()

correlation_matrix = correlation_data.corr(
    method="pearson"
)

correlation_matrix.to_csv(
    OUTPUT_DIR / "multivariate_correlation_matrix.csv"
)

print("\nMultivariate Correlation Matrix:")
print(correlation_matrix)

# --------------------------------------------------
# 5. CORRELATION HEATMAP
# --------------------------------------------------

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5
)

plt.title("Multivariate Numerical Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "multivariate_correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 6. MULTIVARIATE PAIRPLOT
# --------------------------------------------------

pairplot_features = [
    col for col in [
        "Tenure in Months",
        "Monthly Charge",
        "Total Revenue",
        "Age",
        "Satisfaction Score"
    ]
    if col in df.columns
]

pairplot_data = df[
    pairplot_features + [TARGET]
].dropna().copy()

# Sample to reduce computational load
if len(pairplot_data) > 1500:
    pairplot_data = pairplot_data.sample(
        n=1500,
        random_state=42
    )

sns.pairplot(
    pairplot_data,
    hue=TARGET,
    diag_kind="hist",
    corner=True,
    plot_kws={"alpha": 0.5}
)

plt.suptitle(
    "Multivariate Pairplot by Churn",
    y=1.02
)

plt.savefig(
    OUTPUT_DIR / "multivariate_pairplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 7. MULTIVARIATE GROUP ANALYSIS
# --------------------------------------------------

group_features = [
    "Contract",
    "Internet Type"
]

group_features = [
    col for col in group_features
    if col in df.columns
]

if len(group_features) == 2:

    grouped_data = df.groupby(
        group_features,
        observed=False
    ).agg(
        total_customers=(TARGET, "count"),
        churned_customers=(
            TARGET,
            lambda x: (x == "Yes").sum()
        )
    ).reset_index()

    grouped_data["churn_rate"] = (
        grouped_data["churned_customers"]
        / grouped_data["total_customers"]
        * 100
    )

    grouped_data = grouped_data.sort_values(
        "churn_rate",
        ascending=False
    )

    grouped_data.to_csv(
        OUTPUT_DIR / "contract_internet_churn_summary.csv",
        index=False
    )

    print("\nContract + Internet Type Churn Summary:")
    print(grouped_data.to_string(index=False))

else:
    print(
        "\nContract or Internet Type column unavailable."
    )

# --------------------------------------------------
# 8. MULTIVARIATE CHURN RATE HEATMAP
# --------------------------------------------------

if len(group_features) == 2:

    pivot_data = df.pivot_table(
        index="Contract",
        columns="Internet Type",
        values=TARGET,
        aggfunc=lambda x: (x == "Yes").mean() * 100,
        observed=False
    )

    pivot_data.to_csv(
        OUTPUT_DIR / "contract_internet_churn_pivot.csv"
    )

    plt.figure(figsize=(10, 7))

    sns.heatmap(
        pivot_data,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        linewidths=0.5,
        cbar_kws={
            "label": "Churn Rate (%)"
        }
    )

    plt.title(
        "Churn Rate: Contract vs Internet Type"
    )

    plt.xlabel("Internet Type")
    plt.ylabel("Contract")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "contract_internet_churn_heatmap.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# --------------------------------------------------
# 9. NUMERICAL FEATURES BY MULTIPLE GROUPS
# --------------------------------------------------

numeric_group_features = [
    "Contract",
    "Internet Type",
    "Churn Label"
]

numeric_group_features = [
    col for col in numeric_group_features
    if col in df.columns
]

if len(numeric_group_features) == 3:

    selected_numeric = [
        col for col in [
            "Tenure in Months",
            "Monthly Charge",
            "Total Revenue",
            "Satisfaction Score"
        ]
        if col in df.columns
    ]

    grouped_numeric = df.groupby(
        numeric_group_features,
        observed=False
    )[selected_numeric].agg(
        ["mean", "median"]
    )

    grouped_numeric.to_csv(
        OUTPUT_DIR / "multivariate_grouped_numerical_summary.csv"
    )

    print("\nMultivariate Grouped Numerical Summary:")
    print(grouped_numeric)

# --------------------------------------------------
# 10. FINAL OUTPUT
# --------------------------------------------------

print("\nMultivariate Analysis Completed!")

print("\nGenerated Files:")

for file in OUTPUT_DIR.iterdir():
    print(file.name)

print(f"\nOutput Folder: {OUTPUT_DIR}")
