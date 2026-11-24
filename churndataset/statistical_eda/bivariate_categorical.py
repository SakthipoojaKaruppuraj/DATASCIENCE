
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from scipy.stats import chi2_contingency
import math

# --------------------------------------------------
# 1. PATH SETUP
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cleaned_telco_churn.csv"

OUTPUT_DIR = BASE_DIR / "statistical_eda" / "plots" / "bivariate_categorical"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Remove previous outputs
for file in OUTPUT_DIR.iterdir():
    if file.is_file():
        file.unlink()

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

TARGET = "Churn Label"

# --------------------------------------------------
# 3. SELECT CATEGORICAL COLUMNS
# --------------------------------------------------

categorical_cols = df.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

excluded_cols = ["City", "Zip Code"]

categorical_cols = [
    col for col in categorical_cols
    if col not in excluded_cols and col != TARGET
]

print("\nCategorical Columns:")
print(categorical_cols)

# --------------------------------------------------
# 4. CHURN RATE ANALYSIS
# --------------------------------------------------

churn_rate_results = []

for col in categorical_cols:

    temp = df[[col, TARGET]].copy()

    temp = temp.dropna(subset=[TARGET])

    # Calculate category-wise churn rate
    summary = temp.groupby(col, observed=False)[TARGET].agg(
        total_customers="count",
        churned_customers=lambda x: (x == "Yes").sum()
    ).reset_index()

    summary["churn_rate"] = (
        summary["churned_customers"]
        / summary["total_customers"]
        * 100
    )

    summary["feature"] = col

    churn_rate_results.append(summary)

# Combine all feature summaries
churn_rate_df = pd.concat(
    churn_rate_results,
    ignore_index=True
)

churn_rate_df.to_csv(
    OUTPUT_DIR / "categorical_churn_rates.csv",
    index=False
)

print("\nCategory-wise Churn Rate Summary:")
print(churn_rate_df.head(20))

# --------------------------------------------------
# 5. CHI-SQUARE TEST AND CRAMER'S V
# --------------------------------------------------

association_results = []

for col in categorical_cols:

    contingency_table = pd.crosstab(
        df[col],
        df[TARGET]
    )

    # Chi-square test
    chi2, p_value, dof, expected = chi2_contingency(
        contingency_table
    )

    # Cramer's V
    n = contingency_table.sum().sum()

    rows, cols = contingency_table.shape

    denominator = n * min(rows - 1, cols - 1)

    cramer_v = (
        np.sqrt(chi2 / denominator)
        if denominator > 0
        else np.nan
    )

    association_results.append({
        "feature": col,
        "chi_square": chi2,
        "p_value": p_value,
        "degrees_of_freedom": dof,
        "cramers_v": cramer_v,
        "significant": p_value < 0.05
    })

association_df = pd.DataFrame(association_results)

association_df = association_df.sort_values(
    by="cramers_v",
    ascending=False
)

association_df.to_csv(
    OUTPUT_DIR / "chi_square_cramers_v_summary.csv",
    index=False
)

print("\nChi-Square and Cramer's V Results:")
print(association_df.to_string(index=False))

# --------------------------------------------------
# 6. INTERPRET ASSOCIATION STRENGTH
# --------------------------------------------------

def interpret_cramers_v(value):

    if pd.isna(value):
        return "Not calculated"

    if value < 0.10:
        return "Very weak"

    elif value < 0.20:
        return "Weak"

    elif value < 0.40:
        return "Moderate"

    else:
        return "Strong"


association_df["association_strength"] = (
    association_df["cramers_v"]
    .apply(interpret_cramers_v)
)

association_df.to_csv(
    OUTPUT_DIR / "chi_square_cramers_v_summary.csv",
    index=False
)

# --------------------------------------------------
# 7. CHURN RATE VISUALIZATION
# --------------------------------------------------

plot_cols = categorical_cols

n_cols = 3
n_rows = math.ceil(len(plot_cols) / n_cols)

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(18, 5 * n_rows)
)

axes = np.array(axes).flatten()

for i, col in enumerate(plot_cols):

    plot_data = churn_rate_df[
        churn_rate_df["feature"] == col
    ].copy()

    plot_data = plot_data.sort_values(
        "churn_rate",
        ascending=False
    )

    # Limit plotting if categories are too many
    plot_data = plot_data.head(15)

    sns.barplot(
        data=plot_data,
        x=col,
        y="churn_rate",
        ax=axes[i]
    )

    axes[i].set_title(f"Churn Rate by {col}")
    axes[i].set_ylabel("Churn Rate (%)")
    axes[i].set_xlabel(col)

    axes[i].tick_params(
        axis="x",
        rotation=45
    )

    axes[i].set_ylim(0, 100)

# Remove unused subplot axes
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "categorical_churn_rate_plots.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 8. CHI-SQUARE ASSOCIATION VISUALIZATION
# --------------------------------------------------

association_plot = association_df.sort_values(
    "cramers_v",
    ascending=True
)

plt.figure(figsize=(12, 8))

sns.barplot(
    data=association_plot,
    x="cramers_v",
    y="feature"
)

plt.title("Categorical Features - Association with Churn")
plt.xlabel("Cramer's V")
plt.ylabel("Categorical Features")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "cramers_v_association_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 9. FINAL OUTPUT
# --------------------------------------------------

print("\nBivariate Categorical Analysis Completed!")

print("\nOutput Files:")
for file in OUTPUT_DIR.iterdir():
    print(file.name)

print(f"\nOutput Folder: {OUTPUT_DIR}")
