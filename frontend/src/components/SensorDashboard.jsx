import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import Papa from "papaparse";
import Plot from "react-plotly.js";
import GraphDisplay from "./GraphDisplay";

export default function SensorDashboard() {
  const [leafData, setLeafData] = useState(null);
  const [airData, setAirData] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [plotData, setPlotData] = useState(null);

  useEffect(() => {
    const fetchCSVData = async () => {
      try {
        const res = await fetch("/leaf-data.csv");
        const text = await res.text();
        Papa.parse(text, {
          header: true,
          skipEmptyLines: true,
          complete: (result) => {
            const rows = result.data;
            const lastRow = rows[rows.length - 1];
            const {
              timestamp,
              sensor_type,
              Air_Temperature,
              Air_Humidity,
              Leaf_Temperature,
              Leaf_Moisture,
            } = lastRow;

            if (sensor_type === "leaf") {
              setLeafData({
                Leaf_Temperature: parseFloat(Leaf_Temperature),
                Leaf_Moisture: parseFloat(Leaf_Moisture),
              });
            } else {
              setAirData({
                Air_Temperature: parseFloat(Air_Temperature),
                Air_Humidity: parseFloat(Air_Humidity),
              });
            }

            setLastUpdated(timestamp);
          },
        });
      } catch (err) {
        console.error("Error reading CSV:", err);
      }
    };

    fetchCSVData();
    const interval = setInterval(fetchCSVData, 10000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const fetchPlot = async () => {
      try {
        const res = await fetch("http://localhost:5001/plotly-manual-clustering");
        const json = await res.json();
        setPlotData(json);
      } catch (err) {
        console.error("⚠️ Could not load plot:", err);
      }
    };
    fetchPlot();
  }, []);

  const formatTime = (iso) => (iso ? new Date(iso).toLocaleString() : "");

  const triggerScripts = async () => {
    try {
      const res = await fetch("http://localhost:5000/run-scripts", {
        method: "POST",
      });
      const msg = await res.text();
      alert(msg);
    } catch {
      alert("Failed to start scripts.");
    }
  };

  const startFlaskClustering = async () => {
    try {
      const res = await fetch("http://localhost:5000/start-flask", {
        method: "POST",
      });
      const msg = await res.text();
      alert(msg);
    } catch {
      alert("⚠️ Could not start Flask clustering server.");
    }
  };

  const graphConfigs = [
    {
      key: "Air_Temperature",
      label: "🌬️ Air Temperature (°C)",
      color: "#ef4444",
    },
    {
      key: "Air_Humidity",
      label: "💨 Air Humidity (%)",
      color: "#8b5cf6",
    },
    {
      key: "Leaf_Temperature",
      label: "🌿 Leaf Temperature (°C)",
      color: "#10b981",
    },
    {
      key: "Leaf_Moisture",
      label: "💧 Leaf Moisture (%)",
      color: "#3b82f6",
    },
  ];

  return (
    <>
      <div className="p-6 text-gray-800 min-h-screen bg-gradient-to-tr from-green-50 to-blue-50">
        <h1 className="text-3xl font-bold mb-8 text-center text-orange-500">
          🌿 Environmental Sensor Dashboard
        </h1>

        <div className="absolute top-4 right-4 flex gap-2 z-10">
          <button
            onClick={triggerScripts}
            className="w-12 h-12 bg-gray-300 hover:bg-green-500 text-white rounded-full shadow"
            title="Start Data Collection"
          >
            ⏻
          </button>
          <button
            onClick={startFlaskClustering}
            className="w-12 h-12 bg-yellow-400 hover:bg-yellow-500 text-white rounded-full shadow"
            title="Start Flask Clustering Server"
          >
            
          </button>
        </div>

        <div className="flex gap-6 px-6">
          <div className="w-[200px] flex flex-col gap-4">
            <motion.div className="p-4 bg-green-100 rounded-xl shadow">
              <h2 className="font-semibold">🌡️ Leaf Temperature</h2>
              <p className="text-2xl font-bold">
                {leafData?.Leaf_Temperature ?? "No data"} °C
              </p>
            </motion.div>
            <motion.div className="p-4 bg-blue-100 rounded-xl shadow">
              <h2 className="font-semibold">💧 Leaf Moisture</h2>
              <p className="text-2xl font-bold">
                {leafData?.Leaf_Moisture ?? "No data"} %
              </p>
            </motion.div>
            <motion.div className="p-4 bg-red-100 rounded-xl shadow">
              <h2 className="font-semibold">🌬️ Air Temperature</h2>
              <p className="text-2xl font-bold">
                {airData?.Air_Temperature ?? "No data"} °C
              </p>
            </motion.div>
            <motion.div className="p-4 bg-purple-100 rounded-xl shadow">
              <h2 className="font-semibold">💨 Air Humidity</h2>
              <p className="text-2xl font-bold">
                {airData?.Air_Humidity ?? "No data"} %
              </p>
            </motion.div>
          </div>

          <div className="flex-1 flex flex-col gap-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {graphConfigs.map((cfg, i) => (
                <GraphDisplay
                  key={i}
                  dataKey={cfg.key}
                  color={cfg.color}
                  label={cfg.label}
                />
              ))}
            </div>

            <div className="h-[320px] bg-white/90 rounded-xl shadow p-3 mt-6">
              <div className="flex justify-between items-center mb-2">
                <h3 className="text-lg font-semibold text-gray-700">
                  📊 Manual Clustering (last 100 rows)
                </h3>
                <button
                  onClick={async () => {
                    try {
                      const res = await fetch("http://localhost:5001/plotly-manual-clustering");
                      const json = await res.json();
                      setPlotData(json);
                    } catch {
                      alert("⚠️ Could not refresh plot.");
                    }
                  }}
                  className="px-3 py-1 bg-gray-400 hover:bg-gray-500 text-white rounded shadow text-sm"
                >
                  🔄 Refresh
                </button>
              </div>

              <div className="w-full h-full">
                {plotData ? (
                  <Plot
                    data={plotData.data}
                    layout={{ ...plotData.layout, responsive: true, title: "" }}
                    useResizeHandler
                    style={{ width: "100%", height: "100%" }}
                    config={{ displayModeBar: false }}
                  />
                ) : (
                  <p className="text-gray-500 text-center">Loading clustering chart...</p>
                )}
              </div>
            </div>
          </div>
        </div>

        <div className="text-center text-sm text-gray-600 mt-8">
          {lastUpdated ? (
            <>
              <p>
                Last update: <strong>{formatTime(lastUpdated)}</strong>
              </p>
              <p>
                {new Date() - new Date(lastUpdated) > 60000
                  ? "⚠️ No new data in the last minute."
                  : "✅ Live data is up-to-date."}
              </p>
            </>
          ) : (
            <p>Waiting for sensor data...</p>
          )}
        </div>
      </div>
    </>
  );
}
