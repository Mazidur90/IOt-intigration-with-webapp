import pandas as pd
import plotly.graph_objects as go

# === CONFIG ===
FILE_PATH = r"C:\Users\lock3\Downloads\obvious_with_detection.csv"  

# === LOAD DATA ===
df = pd.read_csv(FILE_PATH)
print(f"✅ Loaded {len(df)} rows with detection flags.")

# === REQUIRED COLUMNS CHECK ===
required_cols = ["is_anomaly", "kmeans_anomaly", "isoforest_anomaly", "was_detected"]
missing = [col for col in required_cols if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# === METRIC COUNTS ===
TP = df[(df["is_anomaly"] == 1) & (df["was_detected"] == 1)].shape[0]
FN = df[(df["is_anomaly"] == 1) & (df["was_detected"] == 0)].shape[0]
FP = df[(df["is_anomaly"] == 0) & (df["was_detected"] == 1)].shape[0]
TN = df[(df["is_anomaly"] == 0) & (df["was_detected"] == 0)].shape[0]

# === METRIC SCORES ===
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("\n📊 Detection Metrics:")
print(f" True Positives (TP): {TP}")
print(f" False Negatives (FN): {FN}")
print(f" Flse Positives (FP): {FP}")
print(f" True Negatives (TN): {TN}")
print(f"\n Precision: {precision:.2f}")
print(f" Recall: {recall:.2f}")
print(f" F1 Score: {f1:.2f}")

# === CHART 1: Detection Bar Chart ===
total_true = df["is_anomaly"].sum()
detected_kmeans = df[(df["is_anomaly"] == 1) & (df["kmeans_anomaly"] == 1)].shape[0]
detected_iforest = df[(df["is_anomaly"] == 1) & (df["isoforest_anomaly"] == 1)].shape[0]
detected_both = df[(df["is_anomaly"] == 1) & (df["kmeans_anomaly"] == 1) & (df["isoforest_anomaly"] == 1)].shape[0]

fig1 = go.Figure(data=[
    go.Bar(name='Detected', x=['KMeans', 'Isolation Forest', 'Both'], y=[detected_kmeans, detected_iforest, detected_both]),
    go.Bar(name='Missed', x=['KMeans', 'Isolation Forest', 'Both'], y=[
        total_true - detected_kmeans,
        total_true - detected_iforest,
        total_true - detected_both
    ])
])
fig1.update_layout(
    title="🔍 Anomaly Detection Breakdown",
    barmode='group',
    xaxis_title="Detection Method",
    yaxis_title="Number of Anomalies"
)
fig1.show()

# === CHART 2: Confusion-Style Bar ===
fig2 = go.Figure(data=[
    go.Bar(x=["True Positives", "False Negatives", "False Positives", "True Negatives"],
           y=[TP, FN, FP, TN],
           marker_color=["green", "red", "orange", "blue"])
])
fig2.update_layout(
    title="📊 Confusion Matrix-style Evaluation",
    yaxis_title="Number of Samples"
)
fig2.show()

