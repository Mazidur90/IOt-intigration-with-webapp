import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import plotly.express as px

# === CONFIG ===
FILE_PATH = r"C:\Users\lock3\Downloads\obvious_anomalies_dataset.csv"
FEATURES = ['Leaf_Temperature', 'Leaf_Moisture', 'Air_Temperature', 'Air_Humidity']

# === LOAD DATA ===
df = pd.read_csv(FILE_PATH)
print(f"✅ Loaded {len(df)} rows.")

# === SCALING ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[FEATURES])

# === KMEANS ===
kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
kmeans.fit(X_scaled)
df["kmeans_distance"] = np.linalg.norm(X_scaled - kmeans.cluster_centers_[kmeans.labels_], axis=1)
threshold_kmeans = np.percentile(df["kmeans_distance"], 90)
df["kmeans_anomaly"] = (df["kmeans_distance"] > threshold_kmeans).astype(int)

# === ISOLATION FOREST ===
isoforest = IsolationForest(contamination=0.1, random_state=42)
df["isoforest_score"] = isoforest.fit_predict(X_scaled)
df["isoforest_anomaly"] = (df["isoforest_score"] == -1).astype(int)

# === EVALUATE ===
df["was_detected"] = ((df["kmeans_anomaly"] == 1) | (df["isoforest_anomaly"] == 1)).astype(int)

total_true = df["is_anomaly"].sum()
detected_kmeans = df[(df["is_anomaly"] == 1) & (df["kmeans_anomaly"] == 1)].shape[0]
detected_iforest = df[(df["is_anomaly"] == 1) & (df["isoforest_anomaly"] == 1)].shape[0]
detected_both = df[(df["is_anomaly"] == 1) & (df["kmeans_anomaly"] == 1) & (df["isoforest_anomaly"] == 1)].shape[0]
missed = df[(df["is_anomaly"] == 1) & (df["was_detected"] == 0)]

# === PRINT SUMMARY ===
print("\n📊 Detection Summary:")
print(f"✅ Total True Anomalies: {total_true}")
print(f"🔍 Detected by KMeans: {detected_kmeans}")
print(f"🌲 Detected by Isolation Forest: {detected_iforest}")
print(f"🎯 Detected by Both: {detected_both}")
print(f"❌ Missed by Both: {len(missed)}")

# === SHOW MISSED ROWS ===
if len(missed):
    print("\n⚠️ Missed Anomaly Row Numbers:")
    print(missed.index.to_list())

# === OPTIONAL: PLOT PCA ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]
df["type"] = df.apply(
    lambda row: "Both" if row["kmeans_anomaly"] and row["isoforest_anomaly"]
    else "KMeans" if row["kmeans_anomaly"]
    else "iForest" if row["isoforest_anomaly"]
    else "Normal", axis=1
)

fig = px.scatter(df, x="PCA1", y="PCA2", color="type", symbol="is_anomaly",
                 title="PCA of Sensor Data with Anomaly Detection",
                 hover_data=FEATURES)
fig.show()
