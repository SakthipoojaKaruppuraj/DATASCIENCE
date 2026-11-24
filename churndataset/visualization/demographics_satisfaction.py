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

def run_demographics_satisfaction_analysis():
    print("Running Demographics & Customer Sentiment Analysis...")
    df = pd.read_csv(DATA_PATH)

    # 1. Satisfaction Score vs Churn Rate
    sat_stats = df.groupby("Satisfaction Score")["Churn Label"].agg(
        Total="count",
        Churned=lambda s: (s == "Yes").sum(),
        Churn_Rate=lambda s: (s == "Yes").mean() * 100
    ).reset_index()

    plt.figure(figsize=(10, 6))
    colors = ["#c0392b", "#d9534f", "#f39c12", "#27ae60", "#2ecc71"]
    bars = plt.bar(sat_stats["Satisfaction Score"], sat_stats["Churn_Rate"], color=colors, width=0.5, edgecolor="black", alpha=0.9)
    plt.title("Early Warning Signal: Churn Rate by Customer Satisfaction Score\n(Satisfaction Scores of 1 and 2 result in ~90%+ churn rate)", fontsize=14, weight="bold", pad=12)
    plt.ylabel("Churn Rate (%)", fontsize=12, weight="bold")
    plt.xlabel("Satisfaction Score (1 = Extremely Dissatisfied, 5 = Highly Satisfied)", fontsize=12, weight="bold")
    plt.ylim(0, 110)

    for i, bar in enumerate(bars):
        h = bar.get_height()
        tot = sat_stats.loc[i, "Total"]
        churned_cnt = sat_stats.loc[i, "Churned"]
        plt.annotate(f"{h:.1f}%\n({churned_cnt:,}/{tot:,})",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 5), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, weight="bold")

    plt.tight_layout()
    plot_path1 = os.path.join(PLOTS_DIR, "sentiment_1_satisfaction_score.png")
    plt.savefig(plot_path1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Saved: {plot_path1}")

    # 2. Referral Power (Referred a Friend vs Not)
    ref_stats = df.groupby("Referred a Friend")["Churn Label"].agg(
        Total="count",
        Churned=lambda s: (s == "Yes").sum(),
        Churn_Rate=lambda s: (s == "Yes").mean() * 100
    ).reset_index()

    plt.figure(figsize=(8, 5.5))
    bars = plt.bar(ref_stats["Referred a Friend"], ref_stats["Churn_Rate"], color=["#d9534f", "#2ecc71"], width=0.45, edgecolor="black", alpha=0.9)
    plt.title("The Referral Effect: Churn Rate by Referral Activity\n(Customers who refer others churn less than half as often)", fontsize=14, weight="bold", pad=12)
    plt.ylabel("Churn Rate (%)", fontsize=12, weight="bold")
    plt.xlabel("Has Referred a Friend", fontsize=12, weight="bold")
    plt.ylim(0, 45)

    for i, bar in enumerate(bars):
        h = bar.get_height()
        tot = ref_stats.loc[i, "Total"]
        churned_cnt = ref_stats.loc[i, "Churned"]
        plt.annotate(f"{h:.1f}%\n({churned_cnt:,}/{tot:,})",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 5), textcoords="offset points",
                     ha="center", va="bottom", fontsize=11, weight="bold")

    plt.tight_layout()
    plot_path2 = os.path.join(PLOTS_DIR, "sentiment_2_referrals_impact.png")
    plt.savefig(plot_path2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Saved: {plot_path2}")

    # 3. Demographics: Senior Citizens, Married, Dependents
    demo_segments = [
        {"Feature": "Senior Citizen", "Group": "Senior (Yes)", "Churn Rate": (df[df["Senior Citizen"] == "Yes"]["Churn Label"] == "Yes").mean() * 100},
        {"Feature": "Senior Citizen", "Group": "Non-Senior (No)", "Churn Rate": (df[df["Senior Citizen"] == "No"]["Churn Label"] == "Yes").mean() * 100},
        {"Feature": "Married", "Group": "Married (Yes)", "Churn Rate": (df[df["Married"] == "Yes"]["Churn Label"] == "Yes").mean() * 100},
        {"Feature": "Married", "Group": "Single (No)", "Churn Rate": (df[df["Married"] == "No"]["Churn Label"] == "Yes").mean() * 100},
        {"Feature": "Dependents", "Group": "Has Dependents", "Churn Rate": (df[df["Dependents"] == "Yes"]["Churn Label"] == "Yes").mean() * 100},
        {"Feature": "Dependents", "Group": "No Dependents", "Churn Rate": (df[df["Dependents"] == "No"]["Churn Label"] == "Yes").mean() * 100},
    ]
    demo_df = pd.DataFrame(demo_segments)

    plt.figure(figsize=(11, 5.5))
    bars = plt.barh(demo_df["Group"][::-1], demo_df["Churn Rate"][::-1], color="#34495e", height=0.55, edgecolor="black", alpha=0.85)
    plt.title("Demographic Profile Comparison: Churn Rates Across Customer Cohorts", fontsize=14, weight="bold", pad=12)
    plt.xlabel("Churn Rate (%)", fontsize=12, weight="bold")
    plt.xlim(0, 48)

    for bar in bars:
        w = bar.get_width()
        plt.annotate(f" {w:.1f}%", xy=(w, bar.get_y() + bar.get_height() / 2), xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=10, weight="bold")

    plt.tight_layout()
    plot_path3 = os.path.join(PLOTS_DIR, "sentiment_3_demographics.png")
    plt.savefig(plot_path3, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Saved: {plot_path3}")

if __name__ == "__main__":
    run_demographics_satisfaction_analysis()
