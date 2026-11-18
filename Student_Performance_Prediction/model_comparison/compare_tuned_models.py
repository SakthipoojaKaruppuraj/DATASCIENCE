
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Step 1: Load Dataset
df = pd.read_csv("dataset/StudentPerformanceFactors.csv")
df = df.drop_duplicates()

# Step 2: Create Target
threshold = df["Exam_Score"].median()

df["Performance"] = (
    df["Exam_Score"] >= threshold
).astype(int)

X = df.drop(["Exam_Score", "Performance"], axis=1)
y = df["Performance"]

# Step 3: Identify Columns
numerical_columns = X.select_dtypes(include=np.number).columns
categorical_columns = X.select_dtypes(exclude=np.number).columns

# Step 4: Preprocessing
preprocessor = ColumnTransformer([
    ("numerical", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numerical_columns),

    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_columns)
])

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 6: Define Original Models
original_models = {
    "KNN": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),

    "Decision Tree": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            criterion="gini",
            max_depth=5,
            random_state=42
        ))
    ]),

    "Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ))
    ])
}

# Step 7: Evaluate Function
def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        )
    }

# Step 8: Compare Models
results = []

for name, model in original_models.items():

    print(f"\nTraining Original {name}...")

    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    results.append({
        "Algorithm": name,
        "Version": "Original",
        **metrics
    })

    tuned_model = joblib.load(
        f"saved_models/{name.lower().replace(' ', '_')}_tuned_model.pkl"
    )

    print(f"Evaluating Tuned {name}...")

    tuned_metrics = evaluate_model(
        tuned_model, X_test, y_test
    )

    results.append({
        "Algorithm": name,
        "Version": "Tuned",
        **tuned_metrics
    })

# Step 9: Results Table
results_df = pd.DataFrame(results)

print("\n========== ORIGINAL VS TUNED ==========")
print(results_df.to_string(index=False))

# Step 10: Save Results
results_df.to_csv(
    "model_comparison/original_vs_tuned.csv",
    index=False
)

# Step 11: F1 Score Comparison
pivot_df = results_df.pivot(
    index="Algorithm",
    columns="Version",
    values="F1 Score"
)

pivot_df.plot(kind="bar", figsize=(9, 5))

plt.title("Original vs Tuned Model - F1 Score")
plt.ylabel("F1 Score")
plt.xlabel("Algorithm")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

print("\nComparison Completed Successfully!")
