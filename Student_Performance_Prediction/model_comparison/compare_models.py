
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    silhouette_score
)

# Step 1: Load Dataset
df = pd.read_csv("dataset/StudentPerformanceFactors.csv")

df = df.drop_duplicates()

print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

# Step 2: Define Features and Targets
X = df.drop("Exam_Score", axis=1)
y_regression = df["Exam_Score"]

threshold = y_regression.median()

y_classification = (
    y_regression >= threshold
).astype(int)

print("\nClassification Threshold:", threshold)

# Step 3: Identify Feature Types
numerical_columns = X.select_dtypes(
    include=np.number
).columns

categorical_columns = X.select_dtypes(
    exclude=np.number
).columns

# Step 4: Create Preprocessing Function
def create_preprocessor():

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

# Step 5: Train-Test Split
X_train, X_test, y_train_reg, y_test_reg = train_test_split(
    X,
    y_regression,
    test_size=0.2,
    random_state=42
)

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X,
    y_classification,
    test_size=0.2,
    random_state=42,
    stratify=y_classification
)

print("\nTraining and Testing Data Prepared!")

# Step 6: Linear Regression
print("\n========== LINEAR REGRESSION ==========")

linear_model = Pipeline([
    ("preprocessor", create_preprocessor()),
    ("model", LinearRegression())
])

linear_model.fit(X_train, y_train_reg)

linear_pred = linear_model.predict(X_test)

linear_mae = mean_absolute_error(y_test_reg, linear_pred)
linear_mse = mean_squared_error(y_test_reg, linear_pred)
linear_rmse = np.sqrt(linear_mse)
linear_r2 = r2_score(y_test_reg, linear_pred)

print("MAE:", round(linear_mae, 4))
print("MSE:", round(linear_mse, 4))
print("RMSE:", round(linear_rmse, 4))
print("R2 Score:", round(linear_r2, 4))

# Step 7: Classification Models
classification_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),

    "KNN": KNeighborsClassifier(n_neighbors=5),

    "Decision Tree": DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
}

classification_results = []

for name, algorithm in classification_models.items():

    print(f"\n========== {name.upper()} ==========")

    model = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", algorithm)
    ])

    model.fit(X_train_cls, y_train_cls)

    predictions = model.predict(X_test_cls)

    accuracy = accuracy_score(y_test_cls, predictions)
    precision = precision_score(
        y_test_cls, predictions, zero_division=0
    )
    recall = recall_score(
        y_test_cls, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test_cls, predictions, zero_division=0
    )

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

    classification_results.append({
        "Algorithm": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

# Step 8: K-Means Clustering
print("\n========== K-MEANS CLUSTERING ==========")

cluster_features = df.select_dtypes(
    include=np.number
).drop(columns=["Exam_Score"], errors="ignore")

imputer = SimpleImputer(strategy="median")
cluster_imputed = imputer.fit_transform(cluster_features)

scaler = StandardScaler()
cluster_scaled = scaler.fit_transform(cluster_imputed)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(cluster_scaled)

inertia = kmeans.inertia_

silhouette = silhouette_score(
    cluster_scaled,
    clusters
)

print("Inertia:", round(inertia, 4))
print("Silhouette Score:", round(silhouette, 4))

# Step 9: Classification Comparison Table
classification_df = pd.DataFrame(classification_results)

classification_df = classification_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n========== CLASSIFICATION COMPARISON ==========")
print(classification_df.to_string(index=False))

# Step 10: Save Comparison Results
classification_df.to_csv(
    "model_comparison/classification_comparison.csv",
    index=False
)

regression_df = pd.DataFrame([{
    "Algorithm": "Linear Regression",
    "MAE": linear_mae,
    "MSE": linear_mse,
    "RMSE": linear_rmse,
    "R2 Score": linear_r2
}])

regression_df.to_csv(
    "model_comparison/regression_comparison.csv",
    index=False
)

clustering_df = pd.DataFrame([{
    "Algorithm": "K-Means",
    "Clusters": 3,
    "Inertia": inertia,
    "Silhouette Score": silhouette
}])

clustering_df.to_csv(
    "model_comparison/clustering_comparison.csv",
    index=False
)

# Step 11: Classification Performance Visualization
plt.figure(figsize=(10, 6))

plt.bar(
    classification_df["Algorithm"],
    classification_df["Accuracy"]
)

plt.xlabel("Algorithms")
plt.ylabel("Accuracy")
plt.title("Classification Model Comparison")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# Step 12: Classification F1 Score Visualization
plt.figure(figsize=(10, 6))

plt.bar(
    classification_df["Algorithm"],
    classification_df["F1 Score"]
)

plt.xlabel("Algorithms")
plt.ylabel("F1 Score")
plt.title("Classification F1 Score Comparison")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# Step 13: Regression Visualization
plt.figure(figsize=(6, 5))

plt.scatter(y_test_reg, linear_pred, alpha=0.5)

plt.xlabel("Actual Exam Scores")
plt.ylabel("Predicted Exam Scores")
plt.title("Linear Regression Performance")

plt.tight_layout()
plt.show()

# Step 14: Final Results
print("\n===================================")
print("ALL MODEL COMPARISONS COMPLETED!")
print("===================================")

print("\nRegression Results:")
print(regression_df.to_string(index=False))

print("\nClassification Results:")
print(classification_df.to_string(index=False))

print("\nClustering Results:")
print(clustering_df.to_string(index=False))

print("\nComparison CSV Files Saved Successfully!")
