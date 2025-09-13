import pandas as pd
import numpy as np
from dash import Dash, dcc, html, dash_table
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go

# === CONFIG ===
FILE_PATH = r"C:\Users\lock3\Downloads\obvious_with_detection.csv"  # UPDATE IF NEEDED
FEATURES = ['Leaf_Temperature', 'Leaf_Moisture', 'Air_Temperature', 'Air_Humidity']

# === LOAD DATA ===
df = pd.read_csv(FILE_PATH)

# === ISOFOREST DETECTION ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[FEATURES])
model = IsolationForest(contamination=0.1, random_state=42)
df["isoforest_anomaly"] = model.fit_predict(X_scaled)
df["isoforest_score"] = model.decision_function(X_scaled)
df["isoforest_anomaly"] = df["isoforest_anomaly"].apply(lambda x: 1 if x == -1 else 0)
df["isoforest_detected"] = df["isoforest_anomaly"]

# === PCA ===
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
    f"iForest Score: {row.get('isoforest_score', 0):.4f}", axis=1)

# === Evaluation Labels ===
def get_eval_type(row):
    if row["is_anomaly"] == 1 and row["isoforest_anomaly"] == 1:
        return "True Positive"
    elif row["is_anomaly"] == 1 and row["isoforest_anomaly"] == 0:
        return "False Negative"
    elif row["is_anomaly"] == 0 and row["isoforest_anomaly"] == 1:
        return "False Positive"
    else:
        return "True Negative"

df["evaluation_type"] = df.apply(get_eval_type, axis=1)

# === Metrics ===
TP = (df["evaluation_type"] == "True Positive").sum()
FN = (df["evaluation_type"] == "False Negative").sum()
FP = (df["evaluation_type"] == "False Positive").sum()
TN = (df["evaluation_type"] == "True Negative").sum()

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# === PCA SCATTER by evaluation_type ===
fig = px.scatter(
    df, x="PCA1", y="PCA2", color="evaluation_type", symbol="evaluation_type",
    hover_name="hover_text", template="plotly_dark",
    title="📊 Isolation Forest PCA — TP, FN, FP, TN Visualized"
)
fig.update_traces(marker=dict(size=10, line=dict(width=1, color='white')))

# === Confusion Bar ===
fig_conf = go.Figure(data=[
    go.Bar(x=["True Positive", "False Negative", "False Positive", "True Negative"],
           y=[TP, FN, FP, TN],
           marker_color=["green", "red", "orange", "blue"])
])
fig_conf.update_layout(title="📊 Isolation Forest Confusion Bar", yaxis_title="Count")

# === Table: Only Detected by Isolation Forest ===
df_detected = df[df["isoforest_anomaly"] == 1].copy()
table_columns = FEATURES + ["isoforest_score", "evaluation_type"]
table_data = df_detected[table_columns].round(2).to_dict('records')

# === Table: FP & FN Breakdown ===
df_eval_table = df[df["evaluation_type"].isin(["False Negative", "False Positive"])].copy()
eval_table_columns = FEATURES + ["isoforest_anomaly", "is_anomaly", "evaluation_type"]
eval_table_data = df_eval_table[eval_table_columns].round(2).to_dict("records")

# === DASH APP ===
app = Dash(__name__)
app.title = "Isolation Forest Dashboard"

app.layout = html.Div([
    html.H2("🌲 Isolation Forest Anomaly Detection Dashboard", style={"textAlign": "center", "color": "white"}),

    dcc.Graph(figure=fig),

    html.H3("📋 Detected by Isolation Forest", style={"color": "white"}),
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
        html.H4("📈 Isolation Forest Evaluation Metrics", style={"color": "white"}),
        html.P(f"Precision: {precision:.2f}", style={"color": "white"}),
        html.P(f"Recall: {recall:.2f}", style={"color": "white"}),
        html.P(f"F1 Score: {f1:.2f}", style={"color": "white"}),
    ], style={"padding": "10px", "backgroundColor": "#222", "marginTop": "20px"})
], style={"backgroundColor": "#111", "padding": "20px"})

# === RUN ===
if __name__ == "__main__":
    app.run(debug=True, port=8055)
