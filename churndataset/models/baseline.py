from pathlib import Path
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Path resolution to allow execution from any working directory
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data" if (BASE_DIR.parent / "data").exists() else BASE_DIR / "data"

# Load train and test datasets
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

y_train = pd.read_csv(DATA_DIR / "y_train.csv").iloc[:, 0]
y_test = pd.read_csv(DATA_DIR / "y_test.csv").iloc[:, 0]

# Identify categorical and numerical columns
categorical_columns = X_train.select_dtypes(
    include=["str", "object", "category"]
).columns.tolist()

numerical_columns = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

# Preprocessing: One-Hot encode categoricals, Scale numericals
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ("numerical", StandardScaler(), numerical_columns)
    ]
)

# Build Logistic Regression pipeline
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print("=" * 55)
print("LOGISTIC REGRESSION - BASELINE MODEL EVALUATION")
print("=" * 55)

print(f"\nTraining Samples: {len(X_train):,}")
print(f"Testing Samples:  {len(X_test):,}")
print(f"Features:         {X_train.shape[1]} (22 Categorical, 15 Numerical)")

print("\n--- Key Metrics ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
print(f"TN: {cm[0, 0]:<5} | FP: {cm[0, 1]}")
print(f"FN: {cm[1, 0]:<5} | TP: {cm[1, 1]}")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, digits=4))

# Also train balanced variant to evaluate recall sensitivity
print("=" * 55)
print("LOGISTIC REGRESSION (CLASS-WEIGHT BALANCED)")
print("=" * 55)

balanced_model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
])

balanced_model.fit(X_train, y_train)
y_pred_bal = balanced_model.predict(X_test)
y_prob_bal = balanced_model.predict_proba(X_test)[:, 1]

print(f"Accuracy:  {accuracy_score(y_test, y_pred_bal):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_bal):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_bal):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_bal):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob_bal):.4f}")

cm_bal = confusion_matrix(y_test, y_pred_bal)
print("\n--- Confusion Matrix (Balanced) ---")
print(f"TN: {cm_bal[0, 0]:<5} | FP: {cm_bal[0, 1]}")
print(f"FN: {cm_bal[1, 0]:<5} | TP: {cm_bal[1, 1]}")

print("\nBaseline Model Training Completed!")
