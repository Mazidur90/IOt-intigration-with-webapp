import pandas as pd
import numpy as np
from dash import Dash, dcc, html, dash_table
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

# === CONFIG ===
FILE_PATH = r"C:\Users\lock3\Downloads\obvious_with_detection.csv"
FEATURES = ['Leaf_Temperature', 'Leaf_Moisture', 'Air_Temperature', 'Air_Humidity']

# === LOAD DATA ===
df = pd.read_csv(FILE_PATH)

# === Hybrid OR Detection: Anomaly if detected by either KMeans or Isolation Forest ===
df['hybrid_anomaly'] = ((df['kmeans_anomaly'] == 1) | (df['isoforest_anomaly'] == 1)).astype(int)

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
    f"Hybrid: {row['hybrid_anomaly']}<br>", axis=1)

# === Evaluation Labels ===
def get_eval_type(row):
    if row["is_anomaly"] == 1 and row["hybrid_anomaly"] == 1:
        return "True Positive"
    elif row["is_anomaly"] == 1 and row["hybrid_anomaly"] == 0:
        return "False Negative"
    elif row["is_anomaly"] == 0 and row["hybrid_anomaly"] == 1:
        return "False Positive"
    else:
        return "True Negative"

df["evaluation_type"] = df.apply(get_eval_type, axis=1)

# === Metrics based ONLY on true anomalies ===
TP = ((df["is_anomaly"] == 1) & (df["hybrid_anomaly"] == 1)).sum()
FN = ((df["is_anomaly"] == 1) & (df["hybrid_anomaly"] == 0)).sum()

precision = TP / (TP + FN) if (TP + FN) > 0 else 0  # TRUE PRECISION based only on real anomalies
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# === PCA SCATTER ===
fig = px.scatter(
    df, x="PCA1", y="PCA2", color="evaluation_type", symbol="evaluation_type",
    hover_name="hover_text", template="plotly_dark",
    title="📊 Hybrid (KMeans OR IsoForest) PCA Evaluation"
)
fig.update_traces(marker=dict(size=10, line=dict(width=1, color='white')))

# === Confusion Chart ===
fig_conf = go.Figure(data=[
    go.Bar(x=["True Positive", "False Negative"], y=[TP, FN], marker_color=["green", "red"])
])
fig_conf.update_layout(title="📊 Hybrid Confusion Chart (Only True Anomaly Cases)", yaxis_title="Count")

# === Table: All Detected ===
df_detected = df[df["hybrid_anomaly"] == 1].copy()
table_columns = FEATURES + ["hybrid_anomaly", "evaluation_type"]
table_data = df_detected[table_columns].round(2).to_dict('records')

# === Table: False Negatives ===
df_fn_table = df[df["evaluation_type"] == "False Negative"].copy()
fn_table_columns = FEATURES + ["hybrid_anomaly", "is_anomaly", "evaluation_type"]
fn_table_data = df_fn_table[fn_table_columns].round(2).to_dict("records")

# === DASH APP ===
app = Dash(__name__)
app.title = "Hybrid Anomaly Dashboard"

app.layout = html.Div([
    html.H2("🌿 Hybrid Anomaly Detection (OR Method)", style={"textAlign": "center", "color": "white"}),

    dcc.Graph(figure=fig),

    html.H3("📋 Detected Anomalies", style={"color": "white"}),
    dash_table.DataTable(
        columns=[{"name": col, "id": col} for col in table_columns],
        data=table_data,
        style_table={"overflowX": "auto"},
        style_cell={"backgroundColor": "#1e1e1e", "color": "white", "textAlign": "left"},
        style_header={"backgroundColor": "#333", "color": "white", "fontWeight": "bold"},
        page_size=10
    ),

    html.H3("❌ Missed Anomalies (False Negatives)", style={"color": "white", "marginTop": "40px"}),
    dash_table.DataTable(
        columns=[{"name": col, "id": col} for col in fn_table_columns],
        data=fn_table_data,
        style_table={"overflowX": "auto"},
        style_cell={"backgroundColor": "#1e1e1e", "color": "white", "textAlign": "left"},
        style_header={"backgroundColor": "#333", "color": "white", "fontWeight": "bold"},
        page_size=10
    ),

    dcc.Graph(figure=fig_conf),

    html.Div([
        html.H4("📈 Evaluation Metrics (Based on Ground Truth Only)", style={"color": "white"}),
        html.P(f"Precision (TP / (TP+FN)): {precision:.2f}", style={"color": "white"}),
        html.P(f"Recall (TP / (TP+FN)): {recall:.2f}", style={"color": "white"}),
        html.P(f"F1 Score: {f1:.2f}", style={"color": "white"})
    ], style={"padding": "10px", "backgroundColor": "#222", "marginTop": "20px"})

], style={"backgroundColor": "#111", "padding": "20px"})

# === RUN ===
if __name__ == "__main__":
    app.run(debug=True, port=8056)


