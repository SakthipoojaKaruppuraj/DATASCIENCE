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

def run_tenure_lifecycle_analysis():
    print("Running Customer Lifecycle & Tenure Analysis...")
    df = pd.read_csv(DATA_PATH)

    # 1. Month-by-Month Hazard / Churn Rate Curve
    tenure_stats = df.groupby("Tenure in Months")["Churn Label"].agg(
        Total="count",
        Churned=lambda s: (s == "Yes").sum(),
        Churn_Rate=lambda s: (s == "Yes").mean() * 100
    ).reset_index()

    # Apply rolling average for smoothed hazard curve
    tenure_stats["Smooth_Churn_Rate"] = tenure_stats["Churn_Rate"].rolling(window=3, min_periods=1).mean()

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Highlight Danger Zone (Months 1-6)
    ax1.axvspan(1, 6, color="#f8d7da", alpha=0.6, label="Critical Danger Zone (Months 1-6)")
    ax1.plot(tenure_stats["Tenure in Months"], tenure_stats["Smooth_Churn_Rate"], color="#d9534f", linewidth=3, label="Smoothed Churn Rate (%)")
    ax1.scatter(tenure_stats["Tenure in Months"], tenure_stats["Churn_Rate"], color="#d9534f", alpha=0.4, s=25)

    ax1.set_title("Customer Lifecycle Hazard Curve: The Early Onboarding Cliff\n(Highest customer loss happens within the first 6 months)", fontsize=14, weight="bold", pad=12)
    ax1.set_xlabel("Customer Tenure (in Months)", fontsize=12, weight="bold")
    ax1.set_ylabel("Churn Rate (%)", fontsize=12, weight="bold")
    ax1.set_xlim(1, 72)
    ax1.set_ylim(0, 70)

    # Annotations
    ax1.annotate("Onboarding Cliff:\nMonth 1 Churn ~60%", xy=(1, 60), xytext=(8, 55),
                 arrowprops=dict(facecolor="#d9534f", shrink=0.05, width=1.5, headwidth=8),
                 fontsize=11, weight="bold", color="#d9534f")
    ax1.annotate("Loyalty Plateau:\nChurn stabilizes below 10%", xy=(36, 12), xytext=(42, 25),
                 arrowprops=dict(facecolor="#2b5c8f", shrink=0.05, width=1.5, headwidth=8),
                 fontsize=11, weight="bold", color="#2b5c8f")

    ax1.legend(loc="upper right")
    plt.tight_layout()
    plot_path1 = os.path.join(PLOTS_DIR, "lifecycle_1_hazard_curve.png")
    plt.savefig(plot_path1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Saved: {plot_path1}")

    # 2. Churn Rate by Tenure Cohorts
    bins = [0, 6, 12, 24, 48, 72]
    cohort_labels = ["0–6 Months\n(New / Onboarding)", "7–12 Months\n(1st Year)", "1–2 Years\n(Developing)", "2–4 Years\n(Established)", "4–6 Years\n(Loyal / Mature)"]
    df["Tenure Cohort"] = pd.cut(df["Tenure in Months"], bins=bins, labels=cohort_labels)

    cohort_stats = df.groupby("Tenure Cohort", observed=False)["Churn Label"].agg(
        Total="count",
        Churned=lambda s: (s == "Yes").sum(),
        Churn_Rate=lambda s: (s == "Yes").mean() * 100
    ).reset_index()

    plt.figure(figsize=(11, 6))
    colors = ["#d9534f", "#e67e22", "#f39c12", "#3498db", "#2ecc71"]
    bars = plt.bar(cohort_stats["Tenure Cohort"], cohort_stats["Churn_Rate"], color=colors, width=0.55, edgecolor="black", alpha=0.9)
    plt.title("Churn Rate by Tenure Cohort\n(Retention jumps dramatically after customer's first year)", fontsize=14, weight="bold", pad=12)
    plt.ylabel("Churn Rate (%)", fontsize=12, weight="bold")
    plt.xlabel("Tenure Cohort", fontsize=12, weight="bold")
    plt.ylim(0, 65)

    for i, bar in enumerate(bars):
        h = bar.get_height()
        tot = cohort_stats.loc[i, "Total"]
        churned_cnt = cohort_stats.loc[i, "Churned"]
        plt.annotate(f"{h:.1f}%\n({churned_cnt:,}/{tot:,})",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 5), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, weight="bold")

    plt.tight_layout()
    plot_path2 = os.path.join(PLOTS_DIR, "lifecycle_2_cohort_churn.png")
    plt.savefig(plot_path2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Saved: {plot_path2}")

    # 3. Density Distribution of Tenure (KDE)
    plt.figure(figsize=(10, 5.5))
    sns.kdeplot(df[df["Churn Label"] == "Yes"]["Tenure in Months"], fill=True, color="#d9534f", alpha=0.45, label="Churned Customers", linewidth=2)
    sns.kdeplot(df[df["Churn Label"] == "No"]["Tenure in Months"], fill=True, color="#2b5c8f", alpha=0.45, label="Retained Customers", linewidth=2)
    plt.title("Customer Lifespan Density Distribution (Tenure in Months)", fontsize=14, weight="bold", pad=12)
    plt.xlabel("Tenure (Months)", fontsize=12, weight="bold")
    plt.ylabel("Density", fontsize=12, weight="bold")
    plt.xlim(0, 72)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plot_path3 = os.path.join(PLOTS_DIR, "lifecycle_3_tenure_distribution.png")
    plt.savefig(plot_path3, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Saved: {plot_path3}")

if __name__ == "__main__":
    run_tenure_lifecycle_analysis()
