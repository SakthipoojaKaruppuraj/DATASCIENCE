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

def run_product_services_analysis():
    print("Running Product & Services Stickiness Analysis...")
    df = pd.read_csv(DATA_PATH)

    # 1. Value Added Services (VAS) Retention Lift
    vas_features = [
        "Online Security",
        "Premium Tech Support",
        "Online Backup",
        "Device Protection Plan",
        "Streaming TV",
        "Streaming Movies"
    ]

    vas_comparison = []
    for col in vas_features:
        rate_yes = (df[df[col] == "Yes"]["Churn Label"] == "Yes").mean() * 100
        rate_no = (df[df[col] == "No"]["Churn Label"] == "Yes").mean() * 100
        vas_comparison.append({
            "Service": col,
            "Has Service (Churn %)": rate_yes,
            "No Service (Churn %)": rate_no,
            "Retention Lift (%)": rate_no - rate_yes
        })

    vas_df = pd.DataFrame(vas_comparison).sort_values(by="Retention Lift (%)", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(vas_df))
    width = 0.35

    bars1 = ax.bar(x - width/2, vas_df["Has Service (Churn %)"], width, label="Has Service", color="#2ecc71", edgecolor="black", alpha=0.9)
    bars2 = ax.bar(x + width/2, vas_df["No Service (Churn %)"], width, label="No Service", color="#d9534f", edgecolor="black", alpha=0.9)

    ax.set_title("Product Stickiness: Churn Rate With vs. Without Value-Added Services\n(Online Security and Tech Support cut churn rate by more than half)", fontsize=14, weight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(vas_df["Service"], rotation=15, ha="right", fontsize=11, weight="bold")
    ax.set_ylabel("Churn Rate (%)", fontsize=12, weight="bold")
    ax.set_ylim(0, 50)
    ax.legend(loc="upper right")

    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, weight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, weight="bold")

    plt.tight_layout()
    plot_path1 = os.path.join(PLOTS_DIR, "services_1_vas_retention_lift.png")
    plt.savefig(plot_path1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Saved: {plot_path1}")

    # 2. Internet Type Churn Rate & Monthly Charges
    df["Internet_Type_Clean"] = df["Internet Type"].fillna("No Internet")
    int_stats = df.groupby("Internet_Type_Clean").agg(
        Total=("Customer ID", "count"),
        Churn_Rate=("Churn Label", lambda s: (s == "Yes").mean() * 100),
        Avg_Monthly_Charge=("Monthly Charge", "mean")
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    colors = ["#3498db", "#d9534f", "#2ecc71"]
    bars = ax1.bar(int_stats["Internet_Type_Clean"], int_stats["Churn_Rate"], color=colors, width=0.45, edgecolor="black", alpha=0.85)
    ax1.set_title("Internet Service Analysis: Churn Rate vs Average Monthly Charge\n(Fiber Optic has the highest charges and the highest churn risk)", fontsize=14, weight="bold", pad=12)
    ax1.set_ylabel("Churn Rate (%)", fontsize=12, weight="bold", color="#d9534f")
    ax1.set_xlabel("Internet Type", fontsize=12, weight="bold")
    ax1.set_ylim(0, 55)

    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=11, weight="bold")

    ax2 = ax1.twinx()
    ax2.plot(int_stats["Internet_Type_Clean"], int_stats["Avg_Monthly_Charge"], color="black", marker="s", markersize=9, linewidth=2.5, label="Avg Monthly Charge ($)")
    ax2.set_ylabel("Avg Monthly Charge ($)", fontsize=12, weight="bold", color="black")
    ax2.set_ylim(0, 110)
    for i, txt in enumerate(int_stats["Avg_Monthly_Charge"]):
        ax2.annotate(f"${txt:.1f}/mo", (int_stats["Internet_Type_Clean"][i], txt), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=10, weight="bold")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plot_path2 = os.path.join(PLOTS_DIR, "services_2_internet_type.png")
    plt.savefig(plot_path2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Saved: {plot_path2}")

    # 3. Bundle Depth: Number of Subscribed Add-ons vs Churn Rate
    addon_cols = ["Online Security", "Online Backup", "Device Protection Plan", "Premium Tech Support", "Streaming TV", "Streaming Movies"]
    df["Addon_Count"] = (df[addon_cols] == "Yes").sum(axis=1)

    bundle_stats = df.groupby("Addon_Count")["Churn Label"].agg(
        Total="count",
        Churn_Rate=lambda s: (s == "Yes").mean() * 100
    ).reset_index()

    plt.figure(figsize=(10, 5.5))
    bars = plt.bar(bundle_stats["Addon_Count"], bundle_stats["Churn_Rate"], color="#8e44ad", width=0.55, edgecolor="black", alpha=0.85)
    plt.title("Service Bundle Depth: Churn Rate vs Number of Add-On Services\n(The more services a customer adopts, the higher the switching cost & retention)", fontsize=14, weight="bold", pad=12)
    plt.xlabel("Number of Add-On Services Subscribed (0 to 6)", fontsize=12, weight="bold")
    plt.ylabel("Churn Rate (%)", fontsize=12, weight="bold")
    plt.ylim(0, 50)

    for i, bar in enumerate(bars):
        h = bar.get_height()
        tot = bundle_stats.loc[i, "Total"]
        plt.annotate(f"{h:.1f}%\n(n={tot:,})", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=10, weight="bold")

    plt.tight_layout()
    plot_path3 = os.path.join(PLOTS_DIR, "services_3_bundle_depth.png")
    plt.savefig(plot_path3, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Saved: {plot_path3}")

if __name__ == "__main__":
    run_product_services_analysis()
