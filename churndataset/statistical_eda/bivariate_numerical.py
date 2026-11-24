
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

PLOT_DIR = (
    Path(__file__).resolve().parent
    / "plots"
    / "bivariate_numerical"
)

PLOT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. CLEAN PREVIOUS OUTPUTS
# --------------------------------------------------

for file in PLOT_DIR.iterdir():
    if file.is_file():
        file.unlink()


# --------------------------------------------------
# 3. LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("BI-VARIATE NUMERICAL ANALYSIS")
print("=" * 70)

print(f"\nDataset Shape: {df.shape}")


# --------------------------------------------------
# 4. IDENTIFY NUMERICAL COLUMNS
# --------------------------------------------------

numerical_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

excluded_columns = [
    "Zip Code",
    "Latitude",
    "Longitude",
    "Population"
]

numerical_columns = [
    col for col in numerical_columns
    if col not in excluded_columns
]

print("\nNumerical Columns Selected:")

for col in numerical_columns:
    print(f"- {col}")

print(f"\nTotal Numerical Columns: {len(numerical_columns)}")


# --------------------------------------------------
# 5. PEARSON CORRELATION ANALYSIS
# --------------------------------------------------

print("\n" + "=" * 70)
print("PEARSON CORRELATION ANALYSIS")
print("=" * 70)

pearson_corr = df[numerical_columns].corr(
    method="pearson"
)

print("\nPearson Correlation Matrix:")
print(pearson_corr.round(2))

pearson_corr.to_csv(
    PLOT_DIR / "pearson_correlation_matrix.csv"
)


# --------------------------------------------------
# 6. SPEARMAN CORRELATION ANALYSIS
# --------------------------------------------------

print("\n" + "=" * 70)
print("SPEARMAN CORRELATION ANALYSIS")
print("=" * 70)

spearman_corr = df[numerical_columns].corr(
    method="spearman"
)

print("\nSpearman Correlation Matrix:")
print(spearman_corr.round(2))

spearman_corr.to_csv(
    PLOT_DIR / "spearman_correlation_matrix.csv"
)


# --------------------------------------------------
# 7. IDENTIFY STRONG CORRELATION PAIRS
# --------------------------------------------------

print("\n" + "=" * 70)
print("CORRELATION PAIR INVESTIGATION")
print("=" * 70)

upper_triangle = pearson_corr.where(
    np.triu(
        np.ones(pearson_corr.shape),
        k=1
    ).astype(bool)
)

correlation_pairs = (
    upper_triangle.stack()
    .reset_index()
)

correlation_pairs.columns = [
    "Feature 1",
    "Feature 2",
    "Correlation"
]

correlation_pairs["Absolute Correlation"] = (
    correlation_pairs["Correlation"].abs()
)

correlation_pairs = correlation_pairs.sort_values(
    by="Absolute Correlation",
    ascending=False
)

print("\nStrongest Correlation Pairs:")
print(correlation_pairs.head(20).round(3))

correlation_pairs.to_csv(
    PLOT_DIR / "correlation_pairs_summary.csv",
    index=False
)


# --------------------------------------------------
# 8. PEARSON CORRELATION HEATMAP
# --------------------------------------------------

print("\nGenerating Pearson Correlation Heatmap...")

plt.figure(
    figsize=(14, 10)
)

sns.heatmap(
    pearson_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5
)

plt.title(
    "Pearson Correlation Heatmap",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "pearson_correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 9. SPEARMAN CORRELATION HEATMAP
# --------------------------------------------------

print("\nGenerating Spearman Correlation Heatmap...")

plt.figure(
    figsize=(14, 10)
)

sns.heatmap(
    spearman_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5
)

plt.title(
    "Spearman Correlation Heatmap",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "spearman_correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 10. SELECT SCATTER PLOT PAIRS
# --------------------------------------------------

print("\n" + "=" * 70)
print("SCATTER PLOT ANALYSIS")
print("=" * 70)

candidate_pairs = [
    ("Tenure in Months", "Total Revenue"),
    ("Monthly Charge", "Total Revenue"),
    ("Age", "Monthly Charge"),
    ("Tenure in Months", "Monthly Charge"),
    ("Age", "Total Revenue"),
    ("CLTV", "Total Revenue"),
    ("Age", "Tenure in Months"),
    ("Satisfaction Score", "Monthly Charge"),
    ("Satisfaction Score", "Total Revenue")
]

scatter_pairs = [
    (x, y)
    for x, y in candidate_pairs
    if x in numerical_columns and y in numerical_columns
]

print("\nSelected Scatter Plot Pairs:")

for x, y in scatter_pairs:
    print(f"- {x} vs {y}")


# --------------------------------------------------
# 11. COMBINED SCATTER PLOTS
# --------------------------------------------------

if scatter_pairs:

    print("\nGenerating Combined Scatter Plots...")

    n_cols = 3
    n_rows = math.ceil(len(scatter_pairs) / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(18, 5 * n_rows)
    )

    axes = np.array(axes).flatten()

    for i, (x, y) in enumerate(scatter_pairs):

        sns.scatterplot(
            data=df,
            x=x,
            y=y,
            alpha=0.4,
            ax=axes[i]
        )

        axes[i].set_title(
            f"{x} vs {y}"
        )

        axes[i].set_xlabel(x)
        axes[i].set_ylabel(y)

    # Remove unused axes

    for j in range(len(scatter_pairs), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(
        "Bivariate Numerical Analysis - Scatter Plots",
        fontsize=18,
        fontweight="bold"
    )

    fig.tight_layout()

    fig.savefig(
        PLOT_DIR / "combined_scatterplots.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# --------------------------------------------------
# 12. FINAL VALIDATION
# --------------------------------------------------

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

generated_files = list(PLOT_DIR.iterdir())

for file in generated_files:
    print(f"- {file.name}")

print(f"\nTotal Output Files: {len(generated_files)}")

print(
    f"Pearson Heatmap Exists: "
    f"{(PLOT_DIR / 'pearson_correlation_heatmap.png').exists()}"
)

print(
    f"Spearman Heatmap Exists: "
    f"{(PLOT_DIR / 'spearman_correlation_heatmap.png').exists()}"
)

print(
    f"Scatter Plot Exists: "
    f"{(PLOT_DIR / 'combined_scatterplots.png').exists()}"
)

print(
    f"Pearson CSV Exists: "
    f"{(PLOT_DIR / 'pearson_correlation_matrix.csv').exists()}"
)

print(
    f"Spearman CSV Exists: "
    f"{(PLOT_DIR / 'spearman_correlation_matrix.csv').exists()}"
)

print(
    f"Correlation Pair Summary Exists: "
    f"{(PLOT_DIR / 'correlation_pairs_summary.csv').exists()}"
)

print("\nBivariate Numerical Analysis Completed!")
