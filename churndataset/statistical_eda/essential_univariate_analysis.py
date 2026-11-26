import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# 1. PATH SETUP
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cleaned_telco_churn.csv"
OUTPUT_DIR = BASE_DIR / "statistical_eda" / "plots" / "essential_univariate"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Clean previous outputs
for f in OUTPUT_DIR.iterdir():
    if f.is_file():
        f.unlink()

# Styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)
print("=" * 70)
print("ESSENTIAL UNIVARIATE ANALYSIS (HIGH-SIGNAL BUSINESS DRIVERS)")
print("=" * 70)
print(f"Dataset Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

# --------------------------------------------------
# 3. DEFINE ESSENTIAL DRIVING FACTORS
# --------------------------------------------------

essential_numerical = ["Tenure in Months", "Monthly Charge"]
essential_categorical = [
    "Contract",
    "Internet Type",
    "Online Security",
    "Premium Tech Support",
    "Payment Method"
]

print("\nSelected Essential Numerical Factors:", essential_numerical)
print("Selected Essential Categorical Factors:", essential_categorical)

# --------------------------------------------------
# 4. NUMERICAL EVALUATION (STATS & OUTLIERS)
# --------------------------------------------------

print("\n" + "=" * 70)
print("1. NUMERICAL FACTOR EVALUATION")
print("=" * 70)

num_stats = []
for col in essential_numerical:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
    stat_dict = {
        "Factor": col,
        "Type": "Numerical",
        "Mean": round(df[col].mean(), 2),
        "Median": round(df[col].median(), 2),
        "Std": round(df[col].std(), 2),
        "Min": round(df[col].min(), 2),
        "Q1 (25%)": round(Q1, 2),
        "Q3 (75%)": round(Q3, 2),
        "Max": round(df[col].max(), 2),
        "IQR": round(IQR, 2),
        "Skewness": round(df[col].skew(), 2),
        "Outlier_Count": len(outliers)
    }
    num_stats.append(stat_dict)
    print(f"\n{col}:")
    print(f"  • Median: {stat_dict['Median']} | Mean: {stat_dict['Mean']} | Std: {stat_dict['Std']}")
    print(f"  • 50% Spread (Q1 to Q3): {stat_dict['Q1 (25%)']} to {stat_dict['Q3 (75%)']} (IQR: {stat_dict['IQR']})")
    print(f"  • Skewness: {stat_dict['Skewness']} | Outliers: {stat_dict['Outlier_Count']}")

num_stats_df = pd.DataFrame(num_stats)
num_stats_df.to_csv(OUTPUT_DIR / "essential_numerical_statistics.csv", index=False)

# --------------------------------------------------
# 5. CATEGORICAL EVALUATION (FREQUENCIES & PERCENTAGES)
# --------------------------------------------------

print("\n" + "=" * 70)
print("2. CATEGORICAL FACTOR EVALUATION")
print("=" * 70)

cat_stats = []
for col in essential_categorical:
    counts = df[col].value_counts(dropna=False)
    pcts = df[col].value_counts(normalize=True, dropna=False) * 100
    print(f"\n{col}:")
    for cat, cnt in counts.items():
        pct = pcts[cat]
        print(f"  • {cat}: {cnt:,} ({pct:.2f}%)")
        cat_stats.append({
            "Factor": col,
            "Type": "Categorical",
            "Category": cat,
            "Frequency": cnt,
            "Percentage": round(pct, 2)
        })

cat_stats_df = pd.DataFrame(cat_stats)
cat_stats_df.to_csv(OUTPUT_DIR / "essential_categorical_statistics.csv", index=False)

# --------------------------------------------------
# 6. GENERATE UNIFIED EXECUTIVE DASHBOARD (6 PANELS)
# --------------------------------------------------

print("\n" + "=" * 70)
print("3. GENERATING UNIFIED EXECUTIVE DASHBOARD (300 DPI)")
print("=" * 70)

fig, axes = plt.subplots(3, 2, figsize=(16, 16))

# Panel 1: Tenure in Months (Histogram + KDE)
ax1 = axes[0, 0]
sns.histplot(df["Tenure in Months"], bins=30, kde=True, color="#2b5c8f", ax=ax1, edgecolor="black", alpha=0.7)
ax1.axvline(df["Tenure in Months"].median(), color="red", linestyle="--", linewidth=2, label=f"Median: {df['Tenure in Months'].median():.0f} mos")
ax1.axvspan(1, 6, color="#f8d7da", alpha=0.5, label="High-Risk Onboarding (Months 1–6)")
ax1.set_title("1. Tenure Distribution: Customer Lifecycle Maturity", fontsize=12, weight="bold")
ax1.set_xlabel("Tenure in Months")
ax1.set_ylabel("Customer Count")
ax1.legend(loc="upper right")

# Panel 2: Monthly Charge (Histogram + KDE)
ax2 = axes[0, 1]
sns.histplot(df["Monthly Charge"], bins=30, kde=True, color="#d9534f", ax=ax2, edgecolor="black", alpha=0.7)
ax2.axvline(df["Monthly Charge"].median(), color="black", linestyle="--", linewidth=2, label=f"Median: ${df['Monthly Charge'].median():.2f}/mo")
ax2.set_title("2. Monthly Charge Distribution: ARPU Concentration", fontsize=12, weight="bold")
ax2.set_xlabel("Monthly Charge ($)")
ax2.set_ylabel("Customer Count")
ax2.annotate("Budget Peak (~$20)", xy=(20, 600), xytext=(25, 750),
             arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6), fontsize=9, weight="bold")
ax2.annotate("Premium Plateau ($70–$95)", xy=(80, 500), xytext=(65, 650),
             arrowprops=dict(facecolor="#d9534f", shrink=0.05, width=1, headwidth=6), fontsize=9, weight="bold")
ax2.legend(loc="upper right")

# Panel 3: Contract Breakdown
ax3 = axes[1, 0]
contract_order = ["Month-to-Month", "One Year", "Two Year"]
contract_counts = df["Contract"].value_counts()[contract_order]
colors_contract = ["#d9534f", "#f39c12", "#2ecc71"]
bars_c = ax3.bar(contract_counts.index, contract_counts.values, color=colors_contract, edgecolor="black", width=0.5, alpha=0.9)
ax3.set_title("3. Contract Structure: Stability vs. Volatility", fontsize=12, weight="bold")
ax3.set_ylabel("Customer Count")
ax3.set_ylim(0, 4200)
for bar in bars_c:
    h = bar.get_height()
    pct = (h / len(df)) * 100
    ax3.annotate(f"{h:,}\n({pct:.1f}%)", xy=(bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10, weight="bold")

# Panel 4: Internet Type Breakdown
ax4 = axes[1, 1]
internet_counts = df["Internet Type"].value_counts()
colors_int = ["#3498db", "#9b59b6", "#95a5a6", "#e67e22"]
bars_i = ax4.bar(internet_counts.index, internet_counts.values, color=colors_int[:len(internet_counts)], edgecolor="black", width=0.5, alpha=0.9)
ax4.set_title("4. Core Internet Engine: Fiber vs. Legacy DSL", fontsize=12, weight="bold")
ax4.set_ylabel("Customer Count")
ax4.set_ylim(0, 3600)
for bar in bars_i:
    h = bar.get_height()
    pct = (h / len(df)) * 100
    ax4.annotate(f"{h:,}\n({pct:.1f}%)", xy=(bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10, weight="bold")

# Panel 5: Ecosystem Protection Services (Security & Support Whitespace)
ax5 = axes[2, 0]
services = ["Online Security", "Premium Tech Support"]
sec_counts = [(df[col] == "Yes").sum() for col in services]
no_sec_counts = [(df[col] == "No").sum() for col in services]

x = np.arange(len(services))
w = 0.35
b_yes = ax5.bar(x - w/2, sec_counts, width=w, label="Subscribed (Yes)", color="#2ecc71", edgecolor="black", alpha=0.9)
b_no = ax5.bar(x + w/2, no_sec_counts, width=w, label="Unpenetrated (No)", color="#e74c3c", edgecolor="black", alpha=0.9)
ax5.set_title("5. Value-Added Services: The Untapped Retention Whitespace", fontsize=12, weight="bold")
ax5.set_xticks(x)
ax5.set_xticklabels(services, fontsize=10, weight="bold")
ax5.set_ylabel("Customer Count")
ax5.set_ylim(0, 6000)
ax5.legend(loc="upper left")

for bar in b_yes:
    h = bar.get_height()
    pct = (h / len(df)) * 100
    ax5.annotate(f"{h:,}\n({pct:.1f}%)", xy=(bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, weight="bold")
for bar in b_no:
    h = bar.get_height()
    pct = (h / len(df)) * 100
    ax5.annotate(f"{h:,}\n({pct:.1f}%)", xy=(bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, weight="bold")

# Panel 6: Payment Method Friction
ax6 = axes[2, 1]
pay_counts = df["Payment Method"].value_counts()
colors_pay = ["#e67e22", "#27ae60", "#7f8c8d"]
bars_p = ax6.bar(pay_counts.index, pay_counts.values, color=colors_pay, edgecolor="black", width=0.5, alpha=0.9)
ax6.set_title("6. Payment Method: High Bank Transfer Friction", fontsize=12, weight="bold")
ax6.set_ylabel("Customer Count")
ax6.set_ylim(0, 4600)
for bar in bars_p:
    h = bar.get_height()
    pct = (h / len(df)) * 100
    ax6.annotate(f"{h:,}\n({pct:.1f}%)", xy=(bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10, weight="bold")

plt.suptitle("Executive Dashboard: Essential Univariate Drivers for Business Growth", fontsize=16, weight="bold", y=0.995)
plt.tight_layout()

dashboard_path = OUTPUT_DIR / "essential_univariate_dashboard.png"
plt.savefig(dashboard_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\n Dashboard Saved Successfully: {dashboard_path}")
print("=" * 70)
