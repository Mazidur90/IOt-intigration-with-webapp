import pandas as pd
import numpy as np
from dash import Dash, dcc, html, dash_table
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

# === CONFIG ===
FILE_PATH = r"C:\Users\lock3\Downloads\obvious_with_detection.csv"  # UPDATE THIS IF NEEDED
FEATURES = ['Leaf_Temperature', 'Leaf_Moisture', 'Air_Temperature', 'Air_Humidity']

# === LOAD DATA ===
df = pd.read_csv(FILE_PATH)
df['was_detected'] = ((df['kmeans_anomaly'] == 1) | (df['isoforest_anomaly'] == 1)).astype(int)

# === PCA ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[FEATURES])
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

# === Hover Text ===
df['hover_text'] = df.apply(lambda row:
    f"Leaf Temp: {row['Leaf_Temperature']:.2f}°C<br>"
    f"Leaf Moist: {row['Leaf_Moisture']:.2f}%<br>"
    f"Air Temp: {row['Air_Temperature']:.2f}°C<br>"
    f"Air Hum: {row['Air_Humidity']:.2f}%<br>"
    f"KMeans Dist: {row.get('kmeans_distance', 0):.2f}<br>"
    f"iForest Score: {row.get('isoforest_score', 0):.4f}", axis=1)

# === Anomaly Type ===
df["anomaly_type"] = "Normal"
df.loc[df['kmeans_anomaly'] == 1, "anomaly_type"] = "KMeans"
df.loc[df['isoforest_anomaly'] == 1, "anomaly_type"] = "Isolation Forest"
df.loc[(df['kmeans_anomaly'] == 1) & (df['isoforest_anomaly'] == 1), "anomaly_type"] = "Both"

# === Detection Metrics ===
TP = df[(df["is_anomaly"] == 1) & (df["was_detected"] == 1)].shape[0]
FN = df[(df["is_anomaly"] == 1) & (df["was_detected"] == 0)].shape[0]
FP = df[(df["is_anomaly"] == 0) & (df["was_detected"] == 1)].shape[0]
TN = df[(df["is_anomaly"] == 0) & (df["was_detected"] == 0)].shape[0]
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# === Evaluation Graphs ===
detected_kmeans = df[(df["is_anomaly"] == 1) & (df["kmeans_anomaly"] == 1)].shape[0]
detected_iforest = df[(df["is_anomaly"] == 1) & (df["isoforest_anomaly"] == 1)].shape[0]
detected_both = df[(df["is_anomaly"] == 1) & (df["kmeans_anomaly"] == 1) & (df["isoforest_anomaly"] == 1)].shape[0]
total_true = df["is_anomaly"].sum()

fig_summary = px.scatter(
    df, x="PCA1", y="PCA2", color="anomaly_type", symbol="anomaly_type",
    hover_name="hover_text", template="plotly_dark",
    title="📊 PCA Scatter Plot (KMeans vs Isolation Forest)"
)
fig_summary.update_traces(marker=dict(size=9, line=dict(width=1, color='white')))

fig1 = go.Figure(data=[
    go.Bar(name='Detected', x=['KMeans', 'Isolation Forest', 'Both'], y=[detected_kmeans, detected_iforest, detected_both]),
    go.Bar(name='Missed', x=['KMeans', 'Isolation Forest', 'Both'], y=[
        total_true - detected_kmeans,
        total_true - detected_iforest,
        total_true - detected_both
    ])
])
fig1.update_layout(title="🔍 Anomaly Detection Breakdown", barmode='group')

fig2 = go.Figure(data=[
    go.Bar(x=["True Positives", "False Negatives", "False Positives", "True Negatives"],
           y=[TP, FN, FP, TN],
           marker_color=["green", "red", "orange", "blue"])
])
fig2.update_layout(title="📊 Confusion Matrix-style Bar Chart", yaxis_title="Count")

# === Dash Table ===
df_anomalies = df[df["anomaly_type"] != "Normal"].copy()
table_columns = FEATURES + ["kmeans_distance", "isoforest_score", "anomaly_type"]
table_data = df_anomalies[table_columns].round(2).to_dict("records")

# === DASH LAYOUT ===
app = Dash(__name__)
app.title = "Anomaly Dashboard"

app.layout = html.Div([
    html.H2("🌿 Sensor Anomaly Detection Dashboard", style={"textAlign": "center", "color": "white"}),
    dcc.Graph(figure=fig_summary),
    html.Div([
        html.H3("📋 Anomaly Detection Table", style={"color": "white"}),
        dash_table.DataTable(
            columns=[{"name": col, "id": col} for col in table_columns],
            data=table_data,
            style_table={"overflowX": "auto"},
            style_cell={"backgroundColor": "#1e1e1e", "color": "white", "textAlign": "left"},
            style_header={"backgroundColor": "#333", "color": "white", "fontWeight": "bold"},
            page_size=10
        )
    ]),
    html.Div([
        dcc.Graph(figure=fig1),
        dcc.Graph(figure=fig2),
        html.Div([
            html.H4("📈 Evaluation Scores", style={"color": "white"}),
            html.P(f"Precision: {precision:.2f}", style={"color": "white"}),
            html.P(f"Recall: {recall:.2f}", style={"color": "white"}),
            html.P(f"F1 Score: {f1:.2f}", style={"color": "white"}),
        ], style={"padding": "10px", "backgroundColor": "#222"})
    ])
], style={"backgroundColor": "#111", "padding": "20px"})

# === RUN APP ===
if __name__ == "__main__":
    app.run(debug=True, port=8053)

