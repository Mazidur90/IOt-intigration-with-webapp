const express = require("express");
const path = require("path");

const app = express();
const PORT = process.env.PORT || 3000;

// Serve frontend dist folder
app.use(express.static(path.join(__dirname, "../frontend/dist")));

// Serve the CSV file directly from backend folder
app.use("/leaf.csv", express.static(path.join(__dirname, "leaf.csv")));

// Catch-all fallback (Fix for Express v5+)
app.use((req, res) => {
  res.sendFile(path.join(__dirname, "../frontend/dist/index.html"));
});

app.listen(PORT, () => {
  console.log(`🚀 Server running at http://localhost:${PORT}`);
});


