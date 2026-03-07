import { useState } from "react";
import ImageUploader from "./component/ImageUploader";
import ResultDetailsPanel from "./component/ResultDetailsPanel";
import PlanetMineralGlobe from "./component/Planetmineralglobe";
import "./MineralClassification.css";
// import MineralResultsDisplay from "./component/MineralResultsDisplay";

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
      console.log("[MineralClassification] inference response:", data);
      // Normalize backend response: if the backend returns the class distribution array
      // directly, wrap it under statistics.class_distribution so the rest of the UI
      // (which expects results.statistics.class_distribution) works unchanged.
      if (Array.isArray(data)) {
        setResults({ statistics: { class_distribution: data } });
      } else if (data && Array.isArray(data.statistics?.class_distribution)) {
        setResults(data);
      } else if (data && Array.isArray(data.class_distribution)) {
        // sometimes the backend may return a top-level class_distribution
        setResults({ statistics: { class_distribution: data.class_distribution } });
      } else {
        // fallback: set raw data and let PlanetMineralGlobe attempt normalization
        setResults(data);
      }
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
            {/* Upload Section - full width */}
            <div style={{ gridColumn: "1 / -1", marginBottom: 12 }}>
              <ImageUploader onImageUpload={handleImageUpload} loading={loading} />
            </div>


            {/* Planet globe (right, larger) */}
            <div className="mineral-card" style={{ minHeight: 640, gridColumn: "1 / -1" }}>
              <div style={{ width: "100%", height: "100%" }}>
                <PlanetMineralGlobe
                  results={Array.isArray(results) ? results : (results?.statistics?.class_distribution ?? results?.class_distribution ?? null)}
                  onResults={(data:any) => { setResults(data); }}
                  onUploadState={(b:boolean) => setLoading(b)}
                />
              </div>
            </div>

            {/* Result details full-width below globe */}
            <div className="mineral-card" style={{ gridColumn: "1 / -1", marginTop: 12 }}>
              <ResultDetailsPanel results={results} loading={loading} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
