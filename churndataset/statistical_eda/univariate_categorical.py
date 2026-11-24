
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import math

# --------------------------------------------------
# 1. PATH CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "cleaned_telco_churn.csv"

PLOT_DIR = Path(__file__).resolve().parent / "plots" / "univariate_categorical"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. CLEAN PREVIOUS GENERATED OUTPUTS
# --------------------------------------------------

for file in PLOT_DIR.iterdir():
    if file.is_file():
        file.unlink()


# --------------------------------------------------
# 3. LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("UNIVARIATE CATEGORICAL ANALYSIS")
print("=" * 70)

print(f"\nDataset Shape: {df.shape}")


# --------------------------------------------------
# 4. IDENTIFY CATEGORICAL COLUMNS
# --------------------------------------------------

categorical_columns = df.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

# Exclude high-cardinality / geographical columns
# provisionally excluded during earlier feature investigation.

excluded_columns = [
    "City",
    "Zip Code"
]

categorical_columns = [
    col for col in categorical_columns
    if col not in excluded_columns
]

print("\nCategorical Columns Selected:")

for col in categorical_columns:
    print(f"- {col}")

print(f"\nTotal Categorical Columns: {len(categorical_columns)}")


# --------------------------------------------------
# 5. FREQUENCY AND PERCENTAGE ANALYSIS
# --------------------------------------------------

print("\n" + "=" * 70)
print("FREQUENCY AND PERCENTAGE ANALYSIS")
print("=" * 70)

frequency_results = []

for col in categorical_columns:

    value_counts = df[col].value_counts(dropna=False)

    percentages = df[col].value_counts(
        normalize=True,
        dropna=False
    ) * 100

    print(f"\nColumn: {col}")
    print("-" * 50)

    for category, count in value_counts.items():

        percentage = percentages[category]

        print(
            f"{category}: {count} "
            f"({percentage:.2f}%)"
        )

        frequency_results.append({
            "Column": col,
            "Category": category,
            "Frequency": count,
            "Percentage": percentage
        })


frequency_df = pd.DataFrame(frequency_results)

frequency_df.to_csv(
    PLOT_DIR / "categorical_frequency_summary.csv",
    index=False
)


# --------------------------------------------------
# 6. UNIQUE CATEGORY AND RARE CATEGORY ANALYSIS
# --------------------------------------------------

print("\n" + "=" * 70)
print("CATEGORY DIVERSITY AND RARE CATEGORY ANALYSIS")
print("=" * 70)

category_results = []

RARE_CATEGORY_THRESHOLD = 1.0

for col in categorical_columns:

    value_counts = df[col].value_counts(dropna=False)

    percentages = df[col].value_counts(
        normalize=True,
        dropna=False
    ) * 100

    unique_count = df[col].nunique(dropna=False)

    rare_categories = percentages[
        percentages < RARE_CATEGORY_THRESHOLD
    ]

    print(f"\nColumn: {col}")
    print(f"Unique Categories: {unique_count}")
    print(f"Rare Categories (<1%): {len(rare_categories)}")

    if len(rare_categories) > 0:

        print("Rare Category Details:")

        for category, percentage in rare_categories.items():

            print(
                f"  {category}: "
                f"{percentage:.2f}%"
            )

    category_results.append({
        "Column": col,
        "Unique Categories": unique_count,
        "Rare Category Count": len(rare_categories),
        "Rare Category Threshold (%)": RARE_CATEGORY_THRESHOLD
    })


category_df = pd.DataFrame(category_results)

category_df.to_csv(
    PLOT_DIR / "categorical_diversity_summary.csv",
    index=False
)


# --------------------------------------------------
# 7. COMBINED COUNT PLOTS
# --------------------------------------------------

print("\nGenerating Combined Count Plots...")

n_cols = 2
n_rows = math.ceil(len(categorical_columns) / n_cols)

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(18, 6 * n_rows)
)

axes = np.array(axes).flatten()

for i, col in enumerate(categorical_columns):

    category_order = df[col].value_counts(
        dropna=False
    ).index

    sns.countplot(
        data=df,
        x=col,
        order=category_order,
        ax=axes[i]
    )

    axes[i].set_title(f"Category Distribution: {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Frequency")

    axes[i].tick_params(
        axis="x",
        rotation=45
    )

# Remove unused subplot axes

for j in range(len(categorical_columns), len(axes)):
    fig.delaxes(axes[j])

fig.suptitle(
    "Univariate Categorical Analysis - Count Plots",
    fontsize=18,
    fontweight="bold"
)

fig.tight_layout()

fig.savefig(
    PLOT_DIR / "categorical_countplots.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# --------------------------------------------------
# 8. FINAL VALIDATION
# --------------------------------------------------

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

generated_files = list(PLOT_DIR.iterdir())

for file in generated_files:
    print(f"- {file.name}")

print(f"\nTotal Output Files: {len(generated_files)}")

print(
    f"Count Plot Exists: "
    f"{(PLOT_DIR / 'categorical_countplots.png').exists()}"
)

print(
    f"Frequency Summary Exists: "
    f"{(PLOT_DIR / 'categorical_frequency_summary.csv').exists()}"
)

print(
    f"Diversity Summary Exists: "
    f"{(PLOT_DIR / 'categorical_diversity_summary.csv').exists()}"
)

print("\nUnivariate Categorical Analysis Completed!")
