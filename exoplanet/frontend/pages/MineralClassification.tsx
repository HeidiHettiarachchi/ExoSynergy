import { useState } from "react";
import ImageUploader from "./component/ImageUploader";
import MineralResultsDisplay from "./component/MineralResultsDisplay";
import "./MineralClassification.css";

export default function MineralClassification() {
  const [results, setResults] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleImageUpload = async (file: File) => {
    setLoading(true);
    setResults(null);

    try {
      const formData = new FormData();
      formData.append("image", file);
      formData.append("min_area", "50");
      formData.append("return_image", "true");

      // Call your backend API
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Inference failed");
      }

      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error("Error during inference:", error);
      setResults({
        error:
          "Failed to process image. Make sure the backend server is running on http://localhost:8000",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mineral-page-wrapper">
      <div className="mineral-page-container">
        <div className="mineral-content-wrapper">
          {/* Header */}
          <div className="mineral-header">
            <div className="mineral-header-content">
              <div className="mineral-header-text">
                <h1>Mineral Classification</h1>
                <p>AI-powered CRISM hyperspectral mineral analysis</p>
              </div>
            </div>
          </div>

          {/* Main Content Grid */}
          <div className="mineral-grid">
            {/* Upload Section */}
            <div>
              <div className="mineral-card">
                <div className="mineral-card-header">
                  <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                    />
                  </svg>
                  <h2>Upload Sample Image</h2>
                </div>
                <ImageUploader
                  onImageUpload={handleImageUpload}
                  loading={loading}
                />
              </div>
            </div>

            {/* Results Section */}
            <div className="mineral-card">
              <MineralResultsDisplay results={results} loading={loading} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
