
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay
)

import joblib

# Step 1: Load Dataset
df = pd.read_csv("dataset/StudentPerformanceFactors.csv")

print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

print("\nExam Score Statistics:")
print(df["Exam_Score"].describe())

print("\nUnique Exam Scores:")
print(sorted(df["Exam_Score"].unique()))

print("\nMinimum Exam Score:")
print(df["Exam_Score"].min())

print("\nMaximum Exam Score:")
print(df["Exam_Score"].max())

# Step 2: Remove Duplicates
df = df.drop_duplicates()

# Step 3: Create Binary Target
threshold = df["Exam_Score"].median()

df["Pass_Fail"] = (
    df["Exam_Score"] >= threshold
).astype(int)

print("\nClassification Threshold:", threshold)

print("\nPass/Fail Distribution:")
print(df["Pass_Fail"].value_counts())
print("\nPass/Fail Distribution:")
print(df["Pass_Fail"].value_counts())

# Step 4: Separate Features and Target
X = df.drop(["Exam_Score", "Pass_Fail"], axis=1)
y = df["Pass_Fail"]

# Step 5: Identify Column Types
numerical_columns = X.select_dtypes(include=np.number).columns
categorical_columns = X.select_dtypes(exclude=np.number).columns

# Step 6: Preprocessing Pipelines
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

# Step 8: Create Logistic Regression Pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Step 9: Train Model
model.fit(X_train, y_train)

print("\nLogistic Regression Model Trained Successfully!")

# Step 10: Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nActual Values:")
print(y_test.head(10).values)

print("\nPredicted Values:")
print(y_pred[:10])

print("\nPass Probabilities:")
print(np.round(y_prob[:10], 3))

# Step 11: Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)

print("\nModel Evaluation Results:")
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("ROC-AUC:", round(roc_auc, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Step 12: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Fail", "Pass"]
)

disp.plot(cmap="Blues")
plt.title("Logistic Regression - Confusion Matrix")
plt.show()

# Step 13: Save Model
joblib.dump(
    model,
    "saved_models/logistic_regression_model.pkl"
)

print("\nLogistic Regression Model Saved Successfully!")
