import React, { useState } from "react";

export default function Classifier() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("http://127.0.0.1:8000/classify", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    setResult(data);
    setLoading(false);
  };

  return (
    <div className="p-6 text-center">
      <h1 className="text-2xl font-bold mb-4">🎌 Anime Character Classifier</h1>

      <input
        type="file"
        accept="image/*"
        onChange={(e) => setFile(e.target.files[0])}
        className="block mx-auto mb-4"
      />

      <button
        onClick={handleUpload}
        disabled={!file || loading}
        className="bg-purple-600 text-white px-4 py-2 rounded-lg hover:bg-purple-700 disabled:opacity-50"
      >
        {loading ? "Classifying..." : "Upload & Classify"}
      </button>

      {result && result.best_prediction && (
        <div className="mt-6">
          <h2 className="text-xl font-semibold mb-2">
            Predicted: {result.best_prediction.character_name}
          </h2>
          <p>Confidence: {(result.best_prediction.confidence * 100).toFixed(2)}%</p>

          {result.annotated_image && (
            <img
              src={`data:image/jpeg;base64,${result.annotated_image}`}
              alt="Annotated Result"
              className="rounded-2xl shadow-lg mt-4 mx-auto max-w-md"
            />
          )}
        </div>
      )}
    </div>
  );
}
