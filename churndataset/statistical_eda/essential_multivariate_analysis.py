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
OUTPUT_DIR = BASE_DIR / "statistical_eda" / "plots" / "essential_multivariate"
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
# 2. LOAD DATA & PREPARE LIFECYCLE FEATURE
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)
TARGET = "Churn Label"

def get_tenure_group(tenure):
    if tenure <= 6:
        return "1. Onboarding (1-6m)"
    elif tenure <= 12:
        return "2. First-Year (7-12m)"
    elif tenure <= 24:
        return "3. Developing (13-24m)"
    elif tenure <= 48:
        return "4. Established (25-48m)"
    else:
        return "5. Loyal (49+m)"

df["Tenure_Cohort"] = df["Tenure in Months"].apply(get_tenure_group)

# Support service count (Security + Tech Support + Backup)
support_cols = ["Online Security", "Premium Tech Support", "Online Backup"]
df["Support_Count"] = (df[support_cols] == "Yes").sum(axis=1)

print("=" * 70)
print("ESSENTIAL MULTIVARIATE ANALYSIS (INTERACTIONS & COMPOUND RISK)")
print("=" * 70)

# --------------------------------------------------
# 3. INTERACTION 1: CONTRACT × INTERNET TYPE
# --------------------------------------------------

pivot_contract_internet = df.pivot_table(
    index="Contract",
    columns="Internet Type",
    values=TARGET,
    aggfunc=lambda s: (s == "Yes").mean() * 100
).reindex(["Month-to-Month", "One Year", "Two Year"])[["Fiber Optic", "Cable", "DSL", "No Internet"]]

pivot_contract_internet.to_csv(OUTPUT_DIR / "multivariate_contract_internet_churn.csv")

print("\n1. Contract × Internet Type Churn Rate (%):")
print(pivot_contract_internet.round(1))

# --------------------------------------------------
# 4. INTERACTION 2: TENURE COHORT × CONTRACT
# --------------------------------------------------

pivot_tenure_contract = df.pivot_table(
    index="Tenure_Cohort",
    columns="Contract",
    values=TARGET,
    aggfunc=lambda s: (s == "Yes").mean() * 100
)[["Month-to-Month", "One Year", "Two Year"]]

pivot_tenure_contract.to_csv(OUTPUT_DIR / "multivariate_tenure_contract_churn.csv")

print("\n2. Tenure Cohort × Contract Churn Rate (%):")
print(pivot_tenure_contract.round(1))

# --------------------------------------------------
# 5. INTERACTION 3: FIBER OPTIC × SUPPORT BUNDLE COUNT
# --------------------------------------------------

fiber_df = df[df["Internet Type"] == "Fiber Optic"]
fiber_support_churn = fiber_df.groupby("Support_Count")[TARGET].agg(
    Total="count",
    Churn_Rate=lambda s: (s == "Yes").mean() * 100
).reset_index()

fiber_support_churn.to_csv(OUTPUT_DIR / "multivariate_fiber_support_retention.csv", index=False)

print("\n3. Fiber Optic Churn Rate by Number of Support Tools Subscribed:")
print(fiber_support_churn.to_string(index=False))

# --------------------------------------------------
# 6. GENERATE UNIFIED MULTIVARIATE DASHBOARD (4 PANELS)
# --------------------------------------------------

print("\nGenerating Unified Multivariate Dashboard (300 DPI)...")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Panel 1: Contract × Internet Type Heatmap
ax1 = axes[0, 0]
sns.heatmap(pivot_contract_internet, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={"label": "Churn Rate (%)"},
            linewidths=0.5, ax=ax1, annot_kws={"weight": "bold", "size": 11})
ax1.set_title("1. Compound Risk Matrix: Contract × Internet Type\n(Month-to-Month + Fiber = 58.8% Churn Trap)", fontsize=12, weight="bold")
ax1.set_ylabel("Contract Type")
ax1.set_xlabel("Internet Service Type")

# Panel 2: Tenure Cohort × Contract Heatmap
ax2 = axes[0, 1]
sns.heatmap(pivot_tenure_contract, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={"label": "Churn Rate (%)"},
            linewidths=0.5, ax=ax2, annot_kws={"weight": "bold", "size": 11})
ax2.set_title("2. Lifecycle × Contract Matrix: How Contracts Protect Onboarding\n(Contracts drop 0-6m churn from 62.7% to <10%)", fontsize=12, weight="bold")
ax2.set_ylabel("Customer Lifecycle Cohort")
ax2.set_xlabel("Contract Commitment")

# Panel 3: De-Risking Fiber Optic with Support Tools
ax3 = axes[1, 0]
bars_f = ax3.bar(fiber_support_churn["Support_Count"], fiber_support_churn["Churn_Rate"],
                 color=["#d9534f", "#e67e22", "#f39c12", "#27ae60"], edgecolor="black", width=0.5, alpha=0.9)
ax3.set_title("3. De-Risking Fiber Optic: Impact of Ecosystem Support Tools\n(Adding Security & Tech Support cuts Fiber churn from 52% down to 18%)", fontsize=12, weight="bold")
ax3.set_xlabel("Number of Support Add-ons (Online Security, Tech Support, Backup)")
ax3.set_ylabel("Fiber Customer Churn Rate (%)")
ax3.set_ylim(0, 65)
for bar in bars_f:
    h = bar.get_height()
    ax3.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", weight="bold", fontsize=10)

# Panel 4: 3-Way Scatter (Tenure vs Monthly Charge conditioned by Churn)
ax4 = axes[1, 1]
sample_df = df.sample(n=min(2000, len(df)), random_state=42)
sns.scatterplot(data=sample_df, x="Tenure in Months", y="Monthly Charge", hue=TARGET,
                palette={"No": "#2b5c8f", "Yes": "#d9534f"}, alpha=0.6, ax=ax4, s=30)
ax4.set_title("4. Joint Separation: Tenure vs Monthly Charge by Churn\n(Churners cluster in High-Bill [>$75] + Low-Tenure [<12m] space)", fontsize=12, weight="bold")
ax4.set_xlabel("Tenure in Months")
ax4.set_ylabel("Monthly Charge ($)")
ax4.axvspan(1, 12, color="#f8d7da", alpha=0.3, label="Critical Year 1 Zone")
ax4.axhline(75, color="gray", linestyle="--", alpha=0.7, label="High-ARPU ($75+) Line")
ax4.legend(loc="upper right")

plt.suptitle("Executive Dashboard: Essential Multivariate Interactions & Compound Risk", fontsize=16, weight="bold", y=0.995)
plt.tight_layout()

multivariate_dashboard_path = OUTPUT_DIR / "essential_multivariate_dashboard.png"
plt.savefig(multivariate_dashboard_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\n Multivariate Dashboard Saved Successfully: {multivariate_dashboard_path}")
print("=" * 70)
