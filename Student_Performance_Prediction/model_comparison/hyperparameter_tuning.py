
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
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

print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

# Step 2: Create Classification Target
threshold = df["Exam_Score"].median()

df["Performance"] = (
    df["Exam_Score"] >= threshold
).astype(int)

print("\nClassification Threshold:", threshold)

# Step 3: Separate Features and Target
X = df.drop(["Exam_Score", "Performance"], axis=1)
y = df["Performance"]

# Step 4: Identify Column Types
numerical_columns = X.select_dtypes(include=np.number).columns
categorical_columns = X.select_dtypes(exclude=np.number).columns

# Step 5: Preprocessing
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# Step 7: Define Models and Parameters
models = {

    "KNN": {
        "classifier": KNeighborsClassifier(),
        "parameters": {
            "classifier__n_neighbors": [3, 5, 7, 9],
            "classifier__weights": ["uniform", "distance"],
            "classifier__metric": ["euclidean", "manhattan"]
        }
    },

    "Decision Tree": {
        "classifier": DecisionTreeClassifier(random_state=42),
        "parameters": {
            "classifier__max_depth": [3, 5, 7, 10, None],
            "classifier__criterion": ["gini", "entropy"],
            "classifier__min_samples_split": [2, 5, 10]
        }
    },

    "Random Forest": {
        "classifier": RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        ),
        "parameters": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [5, 10, None],
            "classifier__min_samples_split": [2, 5],
            "classifier__max_features": ["sqrt", "log2"]
        }
    }
}

# Step 8: Hyperparameter Tuning
results = []
best_models = {}

for name, configuration in models.items():

    print(f"\n========== TUNING {name.upper()} ==========")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", configuration["classifier"])
    ])

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=configuration["parameters"],
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    best_models[name] = best_model

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print("\nBest Cross-Validation F1 Score:")
    print(round(grid_search.best_score_, 4))

    # Step 9: Test Best Model
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\nTest Results:")
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

    results.append({
        "Algorithm": name,
        "Best Parameters": str(grid_search.best_params_),
        "CV F1 Score": grid_search.best_score_,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

# Step 10: Comparison Table
results_df = pd.DataFrame(results)

print("\n========== TUNING COMPARISON ==========")
print(results_df.to_string(index=False))

# Step 11: Save Results
results_df.to_csv(
    "model_comparison/tuning_results.csv",
    index=False
)

# Step 12: Save Best Models
import joblib

for name, model in best_models.items():

    filename = name.lower().replace(" ", "_")

    joblib.dump(
        model,
        f"saved_models/{filename}_tuned_model.pkl"
    )

print("\nAll Tuned Models Saved Successfully!")
print("Hyperparameter Tuning Completed!")
