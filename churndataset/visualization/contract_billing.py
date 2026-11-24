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

def run_contract_billing_analysis():
    print("Running Contract & Billing Analysis...")
    df = pd.read_csv(DATA_PATH)

    # 1. Churn Rate by Contract Type
    contract_stats = df.groupby("Contract")["Churn Label"].agg(
        Total="count",
        Churned=lambda s: (s == "Yes").sum(),
        Churn_Rate=lambda s: (s == "Yes").mean() * 100
    ).reindex(["Month-to-Month", "One Year", "Two Year"]).reset_index()

    plt.figure(figsize=(9, 6))
    colors = ["#d9534f", "#f39c12", "#2ecc71"]
    bars = plt.bar(contract_stats["Contract"], contract_stats["Churn_Rate"], color=colors, width=0.5, edgecolor="black", alpha=0.9)
    plt.title("Churn Rate by Contract Type\n(Month-to-Month contracts carry 15x the churn risk of Two-Year contracts)", fontsize=14, weight="bold", pad=12)
    plt.ylabel("Churn Rate (%)", fontsize=12, weight="bold")
    plt.xlabel("Contract Type", fontsize=12, weight="bold")
    plt.ylim(0, 55)

    for i, bar in enumerate(bars):
        h = bar.get_height()
        tot = contract_stats.loc[i, "Total"]
        churned_cnt = contract_stats.loc[i, "Churned"]
        plt.annotate(f"{h:.1f}%\n({churned_cnt:,} / {tot:,})",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 5), textcoords="offset points",
                     ha="center", va="bottom", fontsize=11, weight="bold")

    plt.tight_layout()
    plot_path1 = os.path.join(PLOTS_DIR, "billing_1_contract_churn.png")
    plt.savefig(plot_path1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Saved: {plot_path1}")

    # 2. Payment Method & Paperless Billing
    pay_stats = df.groupby("Payment Method")["Churn Label"].agg(
        Total="count",
        Churned=lambda s: (s == "Yes").sum(),
        Churn_Rate=lambda s: (s == "Yes").mean() * 100
    ).sort_values(by="Churn_Rate", ascending=False).reset_index()

    plt.figure(figsize=(10, 5.5))
    bars = plt.bar(pay_stats["Payment Method"], pay_stats["Churn_Rate"], color="#3498db", width=0.5, edgecolor="black", alpha=0.85)
    plt.title("Churn Rate by Payment Method\n(Bank Withdrawal & Electronic Check users churn significantly faster)", fontsize=14, weight="bold", pad=12)
    plt.ylabel("Churn Rate (%)", fontsize=12, weight="bold")
    plt.xlabel("Payment Method", fontsize=12, weight="bold")
    plt.ylim(0, 45)

    for i, bar in enumerate(bars):
        h = bar.get_height()
        tot = pay_stats.loc[i, "Total"]
        churned_cnt = pay_stats.loc[i, "Churned"]
        plt.annotate(f"{h:.1f}%\n({churned_cnt:,}/{tot:,})",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 5), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, weight="bold")

    plt.tight_layout()
    plot_path2 = os.path.join(PLOTS_DIR, "billing_2_payment_method_churn.png")
    plt.savefig(plot_path2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Saved: {plot_path2}")

    # 3. Promotional Offer Analysis (Offer A to E vs No Offer)
    df["Offer_Clean"] = df["Offer"].fillna("None")
    offer_order = ["Offer A", "Offer B", "Offer C", "Offer D", "Offer E", "None"]
    offer_stats = df.groupby("Offer_Clean")["Churn Label"].agg(
        Total="count",
        Churned=lambda s: (s == "Yes").sum(),
        Churn_Rate=lambda s: (s == "Yes").mean() * 100
    ).reindex(offer_order).reset_index()

    plt.figure(figsize=(10, 5.5))
    offer_colors = ["#2ecc71", "#27ae60", "#f39c12", "#e67e22", "#d9534f", "#95a5a6"]
    bars = plt.bar(offer_stats["Offer_Clean"], offer_stats["Churn_Rate"], color=offer_colors, width=0.55, edgecolor="black", alpha=0.9)
    plt.title("Marketing Offer Retention Effectiveness\n(Offer E customers churn at 50%+, while Offer A retains over 90%)", fontsize=14, weight="bold", pad=12)
    plt.ylabel("Churn Rate (%)", fontsize=12, weight="bold")
    plt.xlabel("Promotional Offer Program", fontsize=12, weight="bold")
    plt.ylim(0, 65)

    for i, bar in enumerate(bars):
        h = bar.get_height()
        tot = offer_stats.loc[i, "Total"]
        churned_cnt = offer_stats.loc[i, "Churned"]
        plt.annotate(f"{h:.1f}%\n({churned_cnt:,}/{tot:,})",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 5), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, weight="bold")

    plt.tight_layout()
    plot_path3 = os.path.join(PLOTS_DIR, "billing_3_offers_churn.png")
    plt.savefig(plot_path3, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Saved: {plot_path3}")

if __name__ == "__main__":
    run_contract_billing_analysis()
