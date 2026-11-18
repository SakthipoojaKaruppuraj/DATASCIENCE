
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import joblib

# Step 1: Load Dataset
df = pd.read_csv("dataset/StudentPerformanceFactors.csv")

print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

# Step 2: Remove Duplicates
df = df.drop_duplicates()

# Step 3: Select Numerical Features
X = df.select_dtypes(include=np.number).drop(
    columns=["Exam_Score"],
    errors="ignore"
)

print("\nSelected Features:")
print(X.columns.tolist())

# Step 4: Handle Missing Values
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# Step 5: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

print("\nFeature Scaling Completed!")

# Step 6: Create K-Means Model
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Step 7: Train Model and Assign Clusters
clusters = kmeans.fit_predict(X_scaled)

print("\nK-Means Clustering Completed!")

# Step 8: Add Cluster Labels
df["Cluster"] = clusters

print("\nCluster Distribution:")
print(df["Cluster"].value_counts().sort_index())

# Step 9: Calculate Cluster Centers
print("\nCluster Centers:")
print(kmeans.cluster_centers_)

# Step 10: Analyze Cluster Characteristics
cluster_summary = df.groupby("Cluster").agg({
    "Hours_Studied": "mean",
    "Attendance": "mean",
    "Previous_Scores": "mean",
    "Exam_Score": "mean"
}).round(2)

print("\nCluster Performance Summary:")
print(cluster_summary)

# Step 11: Evaluate Clustering
inertia = kmeans.inertia_
silhouette = silhouette_score(X_scaled, clusters)

print("\nK-Means Evaluation:")
print("Inertia:", round(inertia, 2))
print("Silhouette Score:", round(silhouette, 4))

# Step 12: PCA for 2D Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(9, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    cmap="viridis",
    alpha=0.6
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Student Clustering")
plt.colorbar(label="Cluster")
plt.show()

# Step 13: Elbow Method
wcss = []

for k in range(1, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)
    wcss.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS / Inertia")
plt.title("Elbow Method")
plt.show()

# Step 14: Save Results
df.to_csv(
    "dataset/student_clustered.csv",
    index=False
)

# Step 15: Save Model Components
joblib.dump(
    {
        "imputer": imputer,
        "scaler": scaler,
        "kmeans": kmeans,
        "pca": pca,
        "features": X.columns.tolist()
    },
    "saved_models/kmeans_model.pkl"
)

print("\nClustered Dataset Saved Successfully!")
print("K-Means Model Saved Successfully!")
