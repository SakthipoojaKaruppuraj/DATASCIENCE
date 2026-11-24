import os
import sys

# Set config dir to avoid font cache warnings
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from financial_impact import run_financial_analysis
from churn_drivers import run_churn_drivers_analysis
from tenure_lifecycle import run_tenure_lifecycle_analysis
from contract_billing import run_contract_billing_analysis
from product_services import run_product_services_analysis
from demographics_satisfaction import run_demographics_satisfaction_analysis

def main():
    print("=" * 60)
    print("STARTING COMPLETE BUSINESS EDA VISUALIZATION PIPELINE")
    print("=" * 60)

    run_financial_analysis()
    print("-" * 60)
    run_churn_drivers_analysis()
    print("-" * 60)
    run_tenure_lifecycle_analysis()
    print("-" * 60)
    run_contract_billing_analysis()
    print("-" * 60)
    run_product_services_analysis()
    print("-" * 60)
    run_demographics_satisfaction_analysis()
    print("-" * 60)

    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    files = sorted([f for f in os.listdir(plots_dir) if f.endswith(".png")])
    print(f"\nALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print(f"Total plots saved: {len(files)} in {plots_dir}")
    print("=" * 60)
    for i, file_name in enumerate(files, 1):
        print(f"{i:2d}. {file_name}")
    print("=" * 60)

if __name__ == "__main__":
    main()
