import pandas as pd
from sklearn.model_selection import train_test_split

# Load final features and target
X = pd.read_csv("data/X_final.csv")
y = pd.read_csv("data/y_final.csv")["Churn Label"]

# Convert target labels into numeric values
y = y.map({"No": 0, "Yes": 1})

# Stratified Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Display dataset shapes
print("Training and Testing Split")
print("-" * 40)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# Display target distribution
print("\nTraining Target Distribution:")
print(y_train.value_counts(normalize=True) * 100)

print("\nTesting Target Distribution:")
print(y_test.value_counts(normalize=True) * 100)

# Save split datasets
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)

y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("\nTrain-Test Split Completed Successfully!")