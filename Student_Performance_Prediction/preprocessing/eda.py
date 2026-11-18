
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load Dataset
df = pd.read_csv("dataset/StudentPerformanceFactors.csv")

# Step 2: Dataset Overview
print("Dataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns.tolist())

# Step 3: Distribution of Exam Scores
plt.figure(figsize=(8, 5))
sns.histplot(df["Exam_Score"], bins=20, kde=True)
plt.title("Distribution of Student Exam Scores")
plt.xlabel("Exam Score")
plt.ylabel("Number of Students")
plt.show()

# Step 4: Study Hours vs Exam Score
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Hours_Studied", y="Exam_Score")
plt.title("Study Hours vs Exam Score")
plt.show()

# Step 5: Attendance vs Exam Score
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Attendance", y="Exam_Score")
plt.title("Attendance vs Exam Score")
plt.show()

# Step 6: Previous Scores vs Exam Score
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Previous_Scores", y="Exam_Score")
plt.title("Previous Scores vs Exam Score")
plt.show()

# Step 7: Correlation Heatmap (Numerical Features)
numerical_df = df.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(12, 8))
sns.heatmap(
    numerical_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Feature Correlation Heatmap")
plt.show()

print("\nEDA Completed Successfully!")
