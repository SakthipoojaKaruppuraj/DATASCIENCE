import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# --------------------------------------------------
# 1. PATH SETUP
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cleaned_telco_churn.csv"
OUTPUT_DIR = BASE_DIR / "statistical_eda" / "plots" / "essential_bivariate"
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
TARGET = "Churn Label"

print("=" * 70)
print("ESSENTIAL BIVARIATE ANALYSIS (RELATIONSHIPS WITH TARGET)")
print("=" * 70)
print(f"Dataset Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

# --------------------------------------------------
# 3. STATISTICAL EVALUATION: CRAMER'S V & CHURN RATES
# --------------------------------------------------

essential_categorical = [
    "Contract",
    "Internet Type",
    "Offer",
    "Payment Method",
    "Online Security",
    "Premium Tech Support",
    "Paperless Billing"
]

bivariate_results = []
cramers_v_results = []

for col in essential_categorical:
    # 1. Chi-Square & Cramer's V
    contingency = pd.crosstab(df[col], df[TARGET])
    chi2, p_val, dof, _ = chi2_contingency(contingency)
    n = contingency.sum().sum()
    rows, cols = contingency.shape
    cramers_v = np.sqrt(chi2 / (n * min(rows - 1, cols - 1)))
    
    cramers_v_results.append({
        "Feature": col,
        "Chi_Square": round(chi2, 2),
        "P_Value": p_val,
        "Cramers_V": round(cramers_v, 3),
        "Association": "Strong" if cramers_v >= 0.4 else "Moderate" if cramers_v >= 0.2 else "Weak"
    })
    
    # 2. Churn Rate per Category
    summary = df.groupby(col, observed=False)[TARGET].agg(
        Total="count",
        Churned=lambda s: (s == "Yes").sum(),
        Churn_Rate=lambda s: (s == "Yes").mean() * 100
    ).reset_index()
    summary["Feature"] = col
    bivariate_results.append(summary)

cramers_df = pd.DataFrame(cramers_v_results).sort_values(by="Cramers_V", ascending=False)
cramers_df.to_csv(OUTPUT_DIR / "essential_cramers_v.csv", index=False)

print("\nEssential Categorical Association Ranking (Cramer's V):")
print(cramers_df.to_string(index=False))

# --------------------------------------------------
# 4. NUMERICAL EVALUATION BY TARGET
# --------------------------------------------------

essential_numerical = ["Tenure in Months", "Monthly Charge", "Satisfaction Score"]

num_target_summary = df.groupby(TARGET)[essential_numerical].agg(["mean", "median", "std"]).round(2)
num_target_summary.to_csv(OUTPUT_DIR / "essential_numerical_target_summary.csv")

print("\nNumerical Differences: Retained vs Churned:")
print(num_target_summary)

# --------------------------------------------------
# 5. GENERATE UNIFIED 6-PANEL BIVARIATE DASHBOARD
# --------------------------------------------------

print("\nGenerating Unified Bivariate Dashboard (300 DPI)...")

fig, axes = plt.subplots(3, 2, figsize=(16, 16))

# Panel 1: Contract vs Churn (100% Stacked Bar)
ax1 = axes[0, 0]
contract_order = ["Month-to-Month", "One Year", "Two Year"]
c_table = pd.crosstab(df["Contract"], df[TARGET], normalize="index").reindex(contract_order) * 100
c_table.plot(kind="bar", stacked=True, ax=ax1, color=["#2b5c8f", "#d9534f"], edgecolor="black", alpha=0.85)
ax1.set_title("1. Contract vs Churn Rate (Cramer's V: 0.453 - Strongest)", fontsize=12, weight="bold")
ax1.set_ylabel("Customer Share (%)")
ax1.set_xlabel("Contract Type")
ax1.tick_params(axis="x", rotation=0)
for i, (idx, row) in enumerate(c_table.iterrows()):
    churn_pct = row["Yes"]
    ax1.annotate(f"Churn: {churn_pct:.1f}%", xy=(i, 100 - churn_pct / 2),
                 ha="center", va="center", color="white", weight="bold", fontsize=10)
ax1.legend(["Retained (No)", "Churned (Yes)"], loc="lower left")

# Panel 2: Internet Type vs Churn (100% Stacked Bar)
ax2 = axes[0, 1]
i_order = ["Fiber Optic", "Cable", "DSL", "No Internet"]
i_table = pd.crosstab(df["Internet Type"], df[TARGET], normalize="index").reindex(i_order) * 100
i_table.plot(kind="bar", stacked=True, ax=ax2, color=["#2b5c8f", "#d9534f"], edgecolor="black", alpha=0.85)
ax2.set_title("2. Internet Type vs Churn Rate (Fiber Optic Exposure)", fontsize=12, weight="bold")
ax2.set_ylabel("Customer Share (%)")
ax2.set_xlabel("Internet Type")
ax2.tick_params(axis="x", rotation=0)
for i, (idx, row) in enumerate(i_table.iterrows()):
    churn_pct = row["Yes"]
    ax2.annotate(f"{churn_pct:.1f}%", xy=(i, 100 - churn_pct / 2),
                 ha="center", va="center", color="white", weight="bold", fontsize=10)
ax2.legend(["Retained (No)", "Churned (Yes)"], loc="lower left")

# Panel 3: Tenure in Months vs Churn (Boxplot + Jitter Overlay)
ax3 = axes[1, 0]
sns.boxplot(x=TARGET, y="Tenure in Months", data=df, ax=ax3, palette={"No": "#2b5c8f", "Yes": "#d9534f"}, width=0.45, boxprops=dict(alpha=0.85))
ax3.set_title("3. Tenure Distribution by Churn Status (The Onboarding Cliff)", fontsize=12, weight="bold")
ax3.set_xlabel("Customer Churn Status")
ax3.set_ylabel("Tenure in Months")
m_no = df[df[TARGET] == "No"]["Tenure in Months"].median()
m_yes = df[df[TARGET] == "Yes"]["Tenure in Months"].median()
ax3.scatter([0, 1], [m_no, m_yes], color="gold", s=100, zorder=5, edgecolor="black")
ax3.annotate(f"Median: {m_no:.0f} mos", xy=(0, m_no), xytext=(15, 10), textcoords="offset points", weight="bold", fontsize=10)
ax3.annotate(f"Median: {m_yes:.0f} mos\n(-28 mos)", xy=(1, m_yes), xytext=(15, 10), textcoords="offset points", weight="bold", color="#d9534f", fontsize=10)

# Panel 4: Monthly Charge vs Churn (Boxplot)
ax4 = axes[1, 1]
sns.boxplot(x=TARGET, y="Monthly Charge", data=df, ax=ax4, palette={"No": "#2b5c8f", "Yes": "#d9534f"}, width=0.45, boxprops=dict(alpha=0.85))
ax4.set_title("4. Monthly Charge Distribution: High-ARPU Vulnerability", fontsize=12, weight="bold")
ax4.set_xlabel("Customer Churn Status")
ax4.set_ylabel("Monthly Charge ($)")
mc_no = df[df[TARGET] == "No"]["Monthly Charge"].mean()
mc_yes = df[df[TARGET] == "Yes"]["Monthly Charge"].mean()
ax4.scatter([0, 1], [mc_no, mc_yes], color="gold", s=100, zorder=5, edgecolor="black")
ax4.annotate(f"Avg: ${mc_no:.2f}", xy=(0, mc_no), xytext=(15, 10), textcoords="offset points", weight="bold", fontsize=10)
ax4.annotate(f"Avg: ${mc_yes:.2f}\n(+$13.17)", xy=(1, mc_yes), xytext=(15, 10), textcoords="offset points", weight="bold", color="#d9534f", fontsize=10)

# Panel 5: Ecosystem Retention Lift (Security & Tech Support)
ax5 = axes[2, 0]
services = ["Online Security", "Premium Tech Support"]
lift_data = []
for s in services:
    r_yes = (df[df[s] == "Yes"][TARGET] == "Yes").mean() * 100
    r_no = (df[df[s] == "No"][TARGET] == "Yes").mean() * 100
    lift_data.append({"Service": s, "With Service": r_yes, "Without Service": r_no})
lift_df = pd.DataFrame(lift_data)

x = np.arange(len(services))
w = 0.35
b1 = ax5.bar(x - w/2, lift_df["With Service"], width=w, label="With Service (Protected)", color="#2ecc71", edgecolor="black", alpha=0.9)
b2 = ax5.bar(x + w/2, lift_df["Without Service"], width=w, label="Without Service (Vulnerable)", color="#e74c3c", edgecolor="black", alpha=0.9)
ax5.set_title("5. Value-Added Service Retention Lift (+23% Moat)", fontsize=12, weight="bold")
ax5.set_xticks(x)
ax5.set_xticklabels(services, fontsize=10, weight="bold")
ax5.set_ylabel("Churn Rate (%)")
ax5.set_ylim(0, 50)
ax5.legend(loc="upper right")
for bar in b1:
    ax5.annotate(f"{bar.get_height():.1f}%", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 3), textcoords="offset points", ha="center", weight="bold", fontsize=10)
for bar in b2:
    ax5.annotate(f"{bar.get_height():.1f}%", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 3), textcoords="offset points", ha="center", weight="bold", fontsize=10)

# Panel 6: Satisfaction Score vs Churn Rate
ax6 = axes[2, 1]
sat_rates = df.groupby("Satisfaction Score")[TARGET].apply(lambda s: (s == "Yes").mean() * 100)
colors_sat = ["#c0392b", "#d9534f", "#f39c12", "#27ae60", "#2ecc71"]
bars_sat = ax6.bar(sat_rates.index, sat_rates.values, color=colors_sat, edgecolor="black", width=0.55, alpha=0.9)
ax6.set_title("6. Satisfaction Score: The Immediate Warning Signal", fontsize=12, weight="bold")
ax6.set_xlabel("Satisfaction Score (1 = Lowest, 5 = Highest)")
ax6.set_ylabel("Churn Rate (%)")
ax6.set_ylim(0, 115)
for bar in bars_sat:
    h = bar.get_height()
    ax6.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", weight="bold", fontsize=10)

plt.suptitle("Executive Dashboard: Essential Bivariate Relationships with Customer Churn", fontsize=16, weight="bold", y=0.995)
plt.tight_layout()

bivariate_dashboard_path = OUTPUT_DIR / "essential_bivariate_dashboard.png"
plt.savefig(bivariate_dashboard_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\n Bivariate Dashboard Saved Successfully: {bivariate_dashboard_path}")
print("=" * 70)
