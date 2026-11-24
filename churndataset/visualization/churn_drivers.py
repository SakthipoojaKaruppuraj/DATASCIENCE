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

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11

def run_churn_drivers_analysis():
    print("Running Churn Drivers & Root Cause Analysis...")
    df = pd.read_csv(DATA_PATH)
    churned = df[df["Churn Label"] == "Yes"].copy()

    # 1. Pareto Chart: Churn Categories (80/20 Rule)
    cat_counts = churned["Churn Category"].value_counts().reset_index()
    cat_counts.columns = ["Category", "Count"]
    cat_counts["Percentage"] = (cat_counts["Count"] / cat_counts["Count"].sum()) * 100
    cat_counts["Cumulative_Percentage"] = cat_counts["Percentage"].cumsum()

    fig, ax1 = plt.subplots(figsize=(11, 6))

    color_palette = ["#d9534f", "#e67e22", "#f39c12", "#3498db", "#95a5a6"]
    bars = ax1.bar(cat_counts["Category"], cat_counts["Count"], color=color_palette[:len(cat_counts)], width=0.55, edgecolor="black", alpha=0.9)
    ax1.set_ylabel("Number of Churned Customers", fontsize=12, weight="bold")
    ax1.set_xlabel("Churn Category", fontsize=12, weight="bold")
    ax1.set_title("Pareto Chart: Primary Drivers of Customer Churn\n(Competitor Offers & Dissatisfaction drive over 65% of all churn)", fontsize=14, weight="bold", pad=15)

    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:,}\n({h/len(churned)*100:.1f}%)",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, weight="bold")

    ax2 = ax1.twinx()
    ax2.plot(cat_counts["Category"], cat_counts["Cumulative_Percentage"], color="#2c3e50", marker="D", linewidth=2.5, markersize=8, label="Cumulative %")
    ax2.set_ylabel("Cumulative Percentage (%)", fontsize=12, weight="bold")
    ax2.set_ylim(0, 110)
    ax2.axhline(80, color="red", linestyle="--", alpha=0.7, label="80% Cutoff Line")
    ax2.legend(loc="center right")

    plt.tight_layout()
    plot_path1 = os.path.join(PLOTS_DIR, "drivers_1_category_pareto.png")
    plt.savefig(plot_path1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Saved: {plot_path1}")

    # 2. Top 10 Specific Reasons Why Customers Churn
    top_reasons = churned.groupby("Churn Reason").agg(
        Count=("Customer ID", "count"),
        Monthly_Revenue_Lost=("Monthly Charge", "sum")
    ).sort_values(by="Count", ascending=True).tail(10)

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(top_reasons.index, top_reasons["Count"], color="#d9534f", height=0.6, edgecolor="black", alpha=0.85)
    ax.set_title("Top 10 Specific Churn Reasons (Ranked by Customer Volume)", fontsize=14, weight="bold", pad=12)
    ax.set_xlabel("Number of Churned Customers", fontsize=12, weight="bold")
    ax.set_ylabel("Reported Churn Reason", fontsize=12, weight="bold")

    for i, bar in enumerate(bars):
        w = bar.get_width()
        rev_lost = top_reasons["Monthly_Revenue_Lost"].iloc[i]
        ax.annotate(f" {w:,} customers  |  ${rev_lost:,.0f}/mo lost",
                    xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=10, weight="bold")

    ax.set_xlim(0, top_reasons["Count"].max() * 1.35)
    plt.tight_layout()
    plot_path2 = os.path.join(PLOTS_DIR, "drivers_2_top_reasons.png")
    plt.savefig(plot_path2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Saved: {plot_path2}")

    # 3. Competitor Threat Deep-Dive
    comp_reasons = churned[churned["Churn Category"] == "Competitor"]["Churn Reason"].value_counts()
    
    plt.figure(figsize=(10, 5))
    bars = plt.barh(comp_reasons.index[::-1], comp_reasons.values[::-1], color="#e67e22", height=0.55, edgecolor="black", alpha=0.9)
    plt.title("Competitor Threat Analysis: Why Competitors Are Winning Our Customers", fontsize=14, weight="bold", pad=12)
    plt.xlabel("Customer Count", fontsize=12, weight="bold")
    for bar in bars:
        w = bar.get_width()
        plt.annotate(f" {w:,} ({w/comp_reasons.sum()*100:.1f}%)",
                     xy=(w, bar.get_y() + bar.get_height() / 2),
                     xytext=(5, 0), textcoords="offset points",
                     ha="left", va="center", fontsize=10, weight="bold")
    plt.xlim(0, comp_reasons.max() * 1.25)
    plt.tight_layout()
    plot_path3 = os.path.join(PLOTS_DIR, "drivers_3_competitor_breakdown.png")
    plt.savefig(plot_path3, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Saved: {plot_path3}")

if __name__ == "__main__":
    run_churn_drivers_analysis()
