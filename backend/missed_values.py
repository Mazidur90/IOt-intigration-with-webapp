import pandas as pd
import numpy as np
from dash import Dash, dcc, html, dash_table
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

# === CONFIG ===
FILE_PATH = r"C:\Users\lock3\Downloads\obvious_with_detection.csv"  # UPDATE IF NEEDED
FEATURES = ['Leaf_Temperature', 'Leaf_Moisture', 'Air_Temperature', 'Air_Humidity']

# === LOAD DATA ===
df = pd.read_csv(FILE_PATH)
df["hybrid_anomaly"] = ((df["kmeans_anomaly"] == 1) | (df["isoforest_anomaly"] == 1)).astype(int)

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
    f"Air Hum: {row['Air_Humidity']:.2f}%", axis=1)

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

# === Metrics (based only on true anomalies) ===
total_anomalies = (df["is_anomaly"] == 1).sum()
tp = ((df["is_anomaly"] == 1) & (df["hybrid_anomaly"] == 1)).sum()
fn = ((df["is_anomaly"] == 1) & (df["hybrid_anomaly"] == 0)).sum()

precision = tp / total_anomalies if total_anomalies > 0 else 0
recall = tp / total_anomalies if total_anomalies > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# === PCA SCATTER ===
fig = px.scatter(
    df, x="PCA1", y="PCA2", color="evaluation_type", symbol="evaluation_type",
    hover_name="hover_text", template="plotly_dark",
    title=" Hybrid Detection PCA — TP, FN, FP, TN Visualized"
)
fig.update_traces(marker=dict(size=10, line=dict(width=1, color='white')))

# === Confusion Chart ===
type_counts = df["evaluation_type"].value_counts()
fig_conf = go.Figure(data=[
    go.Bar(x=type_counts.index, y=type_counts.values,
           marker_color=["green" if x=="True Positive" else "red" if x=="False Negative" else "orange" if x=="False Positive" else "blue" for x in type_counts.index])
])
fig_conf.update_layout(title=" Hybrid Confusion Bar", yaxis_title="Count")

# === Table: Only Detected Anomalies ===
df_detected = df[df["hybrid_anomaly"] == 1].copy()
table_columns = FEATURES + ["kmeans_anomaly", "isoforest_anomaly", "evaluation_type"]
table_data = df_detected[table_columns].round(2).to_dict('records')

# === Table: FP & FN Breakdown ===
df_eval_table = df[df["evaluation_type"].isin(["False Negative", "False Positive"])].copy()
eval_table_columns = FEATURES + ["kmeans_anomaly", "isoforest_anomaly", "evaluation_type"]
eval_table_data = df_eval_table[eval_table_columns].round(2).to_dict("records")

# === DASH APP ===
app = Dash(__name__)
app.title = "Hybrid Detection"

app.layout = html.Div([
    html.H2("🌿 Hybrid Anomaly Detection Dashboard", style={"textAlign": "center", "color": "white"}),

    dcc.Graph(figure=fig),

    html.H3("📋 Anomalies Detected (KMeans or Isolation Forest)", style={"color": "white"}),
    dash_table.DataTable(
        columns=[{"name": col, "id": col} for col in table_columns],
        data=table_data,
        style_table={"overflowX": "auto"},
        style_cell={"backgroundColor": "#1e1e1e", "color": "white", "textAlign": "left"},
        style_header={"backgroundColor": "#333", "color": "white", "fontWeight": "bold"},
        page_size=10
    ),

    html.H3(" False Positives & False Negatives", style={"color": "white", "marginTop": "40px"}),
    dash_table.DataTable(
        columns=[{"name": col, "id": col} for col in eval_table_columns],
        data=eval_table_data,
        style_table={"overflowX": "auto"},
        style_cell={"backgroundColor": "#1e1e1e", "color": "white", "textAlign": "left"},
        style_header={"backgroundColor": "#333", "color": "white", "fontWeight": "bold"},
        page_size=10
    ),

    dcc.Graph(figure=fig_conf),

    html.Div([
        html.H4("📈 Hybrid Evaluation Metrics", style={"color": "white"}),
        html.P(f"Total Injected Anomalies: {total_anomalies}", style={"color": "white"}),
        html.P(f"Detected by Hybrid: {tp}", style={"color": "white"}),
        html.P(f"Missed by Hybrid: {fn}", style={"color": "white"}),
        html.P(f"Precision: {precision:.2f}", style={"color": "white"}),
        html.P(f"Recall: {recall:.2f}", style={"color": "white"}),
        html.P(f"F1 Score: {f1:.2f}", style={"color": "white"}),
    ], style={"padding": "10px", "backgroundColor": "#222", "marginTop": "20px"})

], style={"backgroundColor": "#111", "padding": "20px"})

# === RUN ===
if __name__ == "__main__":
    app.run(debug=True, port=8057)
