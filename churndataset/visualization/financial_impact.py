import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "telco.csv")
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Styling configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11

def run_financial_analysis():
    print("Running Financial Impact Analysis...")
    df = pd.read_csv(DATA_PATH)

    # 1. Total and Monthly Revenue Lost vs Retained
    total_rev = df.groupby("Churn Label")["Total Revenue"].sum()
    monthly_rev = df.groupby("Churn Label")["Monthly Charge"].sum()
    avg_monthly = df.groupby("Churn Label")["Monthly Charge"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar 1: Monthly Recurring Revenue at Risk
    colors = ["#2b5c8f", "#d9534f"]
    bars1 = axes[0].bar(monthly_rev.index, monthly_rev.values / 1e3, color=colors, width=0.55, edgecolor="black", alpha=0.9)
    axes[0].set_title("Monthly Recurring Revenue (MRR) at Risk", fontsize=14, weight="bold", pad=12)
    axes[0].set_ylabel("Monthly Revenue ($ in Thousands)", fontsize=12)
    axes[0].set_xlabel("Customer Churn Status", fontsize=12)
    for bar in bars1:
        height = bar.get_height()
        axes[0].annotate(f"${height:,.1f}K\n({height/monthly_rev.sum()*100:.1f}%)",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 5), textcoords="offset points",
                         ha="center", va="bottom", fontsize=11, weight="bold")

    # Bar 2: Total Historical Revenue Comparison
    bars2 = axes[1].bar(total_rev.index, total_rev.values / 1e6, color=colors, width=0.55, edgecolor="black", alpha=0.9)
    axes[1].set_title("Cumulative Lifetime Revenue Lost vs Retained", fontsize=14, weight="bold", pad=12)
    axes[1].set_ylabel("Total Revenue ($ in Millions)", fontsize=12)
    axes[1].set_xlabel("Customer Churn Status", fontsize=12)
    for bar in bars2:
        height = bar.get_height()
        axes[1].annotate(f"${height:,.2f}M\n({height/total_rev.sum()*100:.1f}%)",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 5), textcoords="offset points",
                         ha="center", va="bottom", fontsize=11, weight="bold")

    plt.suptitle("Executive Financial Summary: Revenue Exposure from Customer Churn", fontsize=16, weight="bold", y=1.02)
    plt.tight_layout()
    plot_path1 = os.path.join(PLOTS_DIR, "financial_1_revenue_loss.png")
    plt.savefig(plot_path1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Saved: {plot_path1}")

    # 2. Monthly Charges Distribution (KDE / Boxplot)
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x="Churn Label", y="Monthly Charge", data=df, palette={"No": "#2b5c8f", "Yes": "#d9534f"}, width=0.45, boxprops=dict(alpha=0.85))
    sns.stripplot(x="Churn Label", y="Monthly Charge", data=df, color="black", alpha=0.08, jitter=0.2, size=3)
    
    # Annotate means
    mean_no = df[df["Churn Label"] == "No"]["Monthly Charge"].mean()
    mean_yes = df[df["Churn Label"] == "Yes"]["Monthly Charge"].mean()
    
    plt.scatter([0, 1], [mean_no, mean_yes], color="gold", s=120, zorder=5, edgecolor="black", label="Average Monthly Charge")
    plt.annotate(f"Avg: ${mean_no:.2f}/mo", xy=(0, mean_no), xytext=(15, 15), textcoords="offset points",
                 fontsize=11, weight="bold", arrowprops=dict(arrowstyle="->", color="black"))
    plt.annotate(f"Avg: ${mean_yes:.2f}/mo\n(+${mean_yes-mean_no:.2f}/mo)", xy=(1, mean_yes), xytext=(15, 15), textcoords="offset points",
                 fontsize=11, weight="bold", color="#d9534f", arrowprops=dict(arrowstyle="->", color="#d9534f"))

    plt.title("Monthly Charges Comparison: Churned vs Retained Customers\n(Higher-paying customers churn at significantly higher rates)", fontsize=14, weight="bold", pad=12)
    plt.xlabel("Customer Churn Status", fontsize=12)
    plt.ylabel("Monthly Charge ($)", fontsize=12)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plot_path2 = os.path.join(PLOTS_DIR, "financial_2_monthly_charges_distribution.png")
    plt.savefig(plot_path2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Saved: {plot_path2}")

    # 3. Churn Rate by Spending Tiers (ARPU Bins)
    bins = [0, 35, 70, 95, 150]
    labels = ["Budget (<$35/mo)", "Moderate ($35-$70/mo)", "High ($70-$95/mo)", "Premium ($95+/mo)"]
    df["Spending Tier"] = pd.cut(df["Monthly Charge"], bins=bins, labels=labels)

    tier_stats = df.groupby("Spending Tier", observed=False)["Churn Label"].agg(
        Total="count",
        Churned=lambda s: (s == "Yes").sum(),
        Churn_Rate=lambda s: (s == "Yes").mean() * 100
    ).reset_index()

    plt.figure(figsize=(10, 6))
    bars = plt.bar(tier_stats["Spending Tier"], tier_stats["Churn_Rate"], color="#d9534f", width=0.55, edgecolor="black", alpha=0.85)
    plt.title("Churn Rate by Customer Spending Tier (ARPU)\n(Premium & High spenders churn over 3x more often than Budget customers)", fontsize=14, weight="bold", pad=12)
    plt.ylabel("Churn Rate (%)", fontsize=12)
    plt.xlabel("Monthly Spending Tier", fontsize=12)
    plt.ylim(0, 50)
    plt.axhline(df["Churn Label"].value_counts(normalize=True)["Yes"] * 100, color="gray", linestyle="--", label=f"Overall Avg Churn Rate ({df['Churn Label'].value_counts(normalize=True)['Yes']*100:.1f}%)")

    for i, bar in enumerate(bars):
        height = bar.get_height()
        total_cust = tier_stats.loc[i, "Total"]
        churned_cust = tier_stats.loc[i, "Churned"]
        plt.annotate(f"{height:.1f}%\n({churned_cust:,}/{total_cust:,})",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 5), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, weight="bold")

    plt.legend(loc="upper left")
    plt.tight_layout()
    plot_path3 = os.path.join(PLOTS_DIR, "financial_3_spending_tier_churn.png")
    plt.savefig(plot_path3, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Saved: {plot_path3}")

if __name__ == "__main__":
    run_financial_analysis()
