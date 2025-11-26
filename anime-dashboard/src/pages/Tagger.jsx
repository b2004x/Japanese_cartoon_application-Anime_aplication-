import React, { useState } from "react";

export default function Tagger() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
      setError("");
    }
  };

  const handleSubmit = async () => {
    if (!image) {
      setError("Please select an image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", image);

    try {
      const res = await fetch("http://127.0.0.1:8000/caption_tag/", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Server error while processing image.");

      const data = await res.json();
      console.log("TAG 0:", data.tags[0]);
      console.log("TAG 1:", data.tags[1]);
      console.log("TAG 2:", data.tags[2]);
      console.log("TAG LENGTH:", data.tags.length);
      setResult(data);
    } catch (err) {
      console.error(err);
      setError("Failed to process the image. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center p-6 min-h-screen bg-gray-50">
      <h1 className="text-3xl font-bold mb-6 text-gray-800">
        🏷️ Anime Tagging & Caption Generator
      </h1>

      <div className="bg-white shadow-xl rounded-2xl p-6 w-full max-w-2xl">
        {/* Upload Area */}
        <div className="flex flex-col items-center gap-4">
          {preview ? (
            <img
              src={preview}
              alt="Preview"
              className="w-64 h-64 object-cover rounded-xl shadow-md border border-gray-200"
            />
          ) : (
            <div className="w-64 h-64 flex items-center justify-center bg-gray-100 rounded-xl border border-dashed border-gray-400 text-gray-500">
              No image selected
            </div>
          )}

          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="mt-3 block text-sm text-gray-700"
          />

          <button
            onClick={handleSubmit}
            disabled={loading}
            className={`mt-4 px-6 py-2 rounded-xl font-semibold shadow-md transition-all ${
              loading
                ? "bg-gray-400 cursor-not-allowed"
                : "bg-purple-600 hover:bg-purple-700 text-white"
            }`}
          >
            {loading ? "Analyzing..." : "Generate Caption & Tags"}
          </button>

          {error && (
            <p className="text-red-500 mt-3 font-medium">{error}</p>
          )}
        </div>

        {/* Results */}
        {result && (
          <div className="mt-6 bg-gray-100 rounded-xl p-4 shadow-inner">
            <h2 className="text-xl font-semibold text-gray-700 mb-2">
              📝 Caption
            </h2>
            <p className="text-gray-800 italic mb-4">
              "{result.caption || 'No caption generated.'}"
            </p>

            <h2 className="text-xl font-semibold text-gray-700 mb-2">
              🏷️ Tags
            </h2>
            <div className="flex flex-wrap gap-2">
              {result.tags && result.tags.length > 0 ? (
                result.tags.map((tag, i) => (
                  <span
                    key={i}
                    style={{ margin: "4px" }}
                    className="bg-purple-100 text-purple-700 px-3 py-1 rounded-full text-sm font-medium"
                  >
                    {tag}
                  </span>
                ))
              ) : (
                <p className="text-gray-500">No tags found.</p>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}