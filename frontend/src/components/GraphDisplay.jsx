import React, { useEffect, useState } from "react";
import Papa from "papaparse";
import Plot from "react-plotly.js";

export default function GraphDisplay({ dataKey, color, label }) {
  const [data, setData] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch("/leaf-data.csv");
        const text = await res.text();
        Papa.parse(text, {
          header: true,
          skipEmptyLines: true,
          complete: (result) => {
            const rows = result.data;
            const filtered = rows
              .map((r) => ({
                x: new Date(r.timestamp),
                y: parseFloat(r[dataKey]),
              }))
              .filter((point) => !isNaN(point.y));
            setData(filtered);
          },
        });
      } catch (err) {
        console.error(`❌ Error loading ${dataKey}:`, err);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, [dataKey]);

  return (
    <div className="bg-white rounded-lg p-3 shadow">
      <h3 className="text-md font-semibold text-gray-800 mb-1">{label}</h3>
      <Plot
        data={[
          {
            x: data.map((d) => d.x),
            y: data.map((d) => d.y),
            type: "scatter",
            mode: "lines+markers",
            marker: { color },
            name: label,
          },
        ]}
        layout={{
          width: "100%",
          height: 200,
          margin: { t: 20, b: 30, l: 40, r: 10 },
          xaxis: { title: "Timestamp" },
          yaxis: { title: label },
        }}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
        config={{ displayModeBar: false }}
      />
    </div>
  );
}
