import React from "react";

export default function DashEmbed({ endpoint = "http://localhost:8050", title = "Anomaly Detection Dashboard" }) {
  return (
    <div className="w-full h-[80vh] rounded-2xl overflow-hidden shadow-lg border border-gray-200">
      <iframe
        src={endpoint}
        title={title}
        className="w-full h-full"
        loading="lazy"
        sandbox="allow-scripts allow-same-origin"
      >
        
      </iframe>
    </div>
  );
}
