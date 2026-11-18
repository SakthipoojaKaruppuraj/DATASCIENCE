
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

import joblib

# Step 1: Load Dataset
df = pd.read_csv("dataset/StudentPerformanceFactors.csv")

print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

# Step 2: Remove Duplicates
df = df.drop_duplicates()

# Step 3: Create Target Variable
threshold = df["Exam_Score"].median()

df["Performance"] = (
    df["Exam_Score"] >= threshold
).astype(int)

print("\nClassification Threshold:", threshold)
print("\nClass Distribution:")
print(df["Performance"].value_counts())

# Step 4: Separate Features and Target
X = df.drop(["Exam_Score", "Performance"], axis=1)
y = df["Performance"]

# Step 5: Identify Column Types
numerical_columns = X.select_dtypes(include=np.number).columns
categorical_columns = X.select_dtypes(exclude=np.number).columns

# Step 6: Preprocessing
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

# Step 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Step 8: Create Decision Tree Model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        random_state=42
    ))
])

# Step 9: Train Model
model.fit(X_train, y_train)

print("\nDecision Tree Model Trained Successfully!")

# Step 10: Predictions
y_pred = model.predict(X_test)

print("\nActual Values:")
print(y_test.head(10).values)

print("\nPredicted Values:")
print(y_pred[:10])

# Step 11: Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\nDecision Tree Evaluation Results:")
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Step 12: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Below Median", "Median or Above"]
)

disp.plot(cmap="Blues")
plt.title("Decision Tree - Confusion Matrix")
plt.show()

# Step 13: Visualize Decision Tree
tree_model = model.named_steps["classifier"]
feature_names = model.named_steps["preprocessor"].get_feature_names_out()

plt.figure(figsize=(20, 10))

plot_tree(
    tree_model,
    feature_names=feature_names,
    class_names=["Below Median", "Median or Above"],
    filled=True,
    rounded=True,
    fontsize=7
)

plt.title("Decision Tree Visualization")
plt.show()

# Step 14: Save Model
joblib.dump(
    model,
    "saved_models/decision_tree_model.pkl"
)

print("\nDecision Tree Model Saved Successfully!")
