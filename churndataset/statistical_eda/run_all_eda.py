
import subprocess
import sys

from pathlib import Path

# --------------------------------------------------
# 1. PATH SETUP
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

analysis_files = [
    "univariate_numerical.py",
    "univariate_categorical.py",
    "bivariate_numerical.py",
    "bivariate_categorical.py",
    "bivariate_target.py",
    "multivariate_analysis.py"
]

# --------------------------------------------------
# 2. RUN ALL ANALYSIS FILES
# --------------------------------------------------

print("=" * 60)
print("STATISTICAL EDA PIPELINE")
print("=" * 60)

for file_name in analysis_files:

    file_path = BASE_DIR / file_name

    if not file_path.exists():
        print(f"\nFile not found: {file_name}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"Running: {file_name}")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, str(file_path)],
        cwd=BASE_DIR.parent
    )

    if result.returncode != 0:
        print(f"\nERROR: {file_name} failed.")
        sys.exit(result.returncode)

    print(f"\nCompleted: {file_name}")

# --------------------------------------------------
# 3. FINAL STATUS
# --------------------------------------------------

print("\n" + "=" * 60)
print("ALL STATISTICAL EDA ANALYSES COMPLETED SUCCESSFULLY!")
print("=" * 60)
