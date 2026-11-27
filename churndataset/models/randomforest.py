from pathlib import Path
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Path resolution to allow execution from any directory
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data" if (BASE_DIR.parent / "data").exists() else BASE_DIR / "data"

# ==========================================
# 1. LOAD TRAIN AND TEST DATA
# ==========================================

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

y_train = pd.read_csv(DATA_DIR / "y_train.csv").iloc[:, 0]
y_test = pd.read_csv(DATA_DIR / "y_test.csv").iloc[:, 0]

# ==========================================
# 2. IDENTIFY CATEGORICAL FEATURES
# ==========================================

categorical_columns = X_train.select_dtypes(
    include=["str", "object", "category"]
).columns.tolist()

# ==========================================
# 3. PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ],
    remainder="passthrough"
)

# ==========================================
# 4. CREATE RANDOM FOREST MODEL
# ==========================================

model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

# ==========================================
# 5. TRAIN MODEL
# ==========================================

model.fit(X_train, y_train)

# ==========================================
# 6. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ==========================================
# 7. EVALUATE MODEL
# ==========================================

print("=" * 50)
print("RANDOM FOREST - MODEL EVALUATION")
print("=" * 50)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nRandom Forest Training Completed!")
