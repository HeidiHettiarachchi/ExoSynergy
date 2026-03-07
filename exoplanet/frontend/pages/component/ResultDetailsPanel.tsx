import { useState } from "react";
import "./MineralResultsDisplay.css";

// Aggregate and clean up class_distribution entries that may contain
// many repeated rows from the backend. This groups by mineral name
// (case-insensitive), sums percentage (or pixel_count if percentage
// missing), and returns a sorted array suitable for display.
function normalizeName(rawName: string) {
  if (!rawName) return "Unknown";
  // remove zero-width / BOM characters, collapse whitespace, trim
    const cleaned = rawName
      .normalize("NFKD")
      // remove common invisible / zero-width characters and BOM
      .replace(/[\u200B\u200C\u200D\uFEFF]/g, "")
      .replace(/\s+/g, " ")
      .trim();
  return cleaned || "Unknown";
}

function aggregateDistribution(
  dist: Array<{ mineral_name: string; percentage?: number; pixel_count?: number }>,
  totalPixels?: number
) {
  if (!Array.isArray(dist)) return [];
  const map = new Map<string, { name: string; percentage: number; pixel_count: number; count: number }>();
  let hasPercentage = false;

  for (const entry of dist) {
    const raw = entry?.mineral_name ?? "";
    const name = normalizeName(raw);
    const key = name.toLowerCase();
    const pct = typeof entry.percentage === "number" ? entry.percentage : NaN;
    const px = typeof entry.pixel_count === "number" ? entry.pixel_count : 0;
    if (!Number.isNaN(pct)) hasPercentage = true;

    const existing = map.get(key);
    if (existing) {
      existing.pixel_count += px;
      existing.percentage += Number.isNaN(pct) ? 0 : pct;
      existing.count += 1;
    } else {
      map.set(key, { name, percentage: Number.isNaN(pct) ? 0 : pct, pixel_count: px, count: 1 });
    }
  }

  const arr = Array.from(map.values()).map((v) => {
    // If percentages weren't provided by backend (all zeros), but we have
    // pixel counts and totalPixels, derive a percentage.
    let finalPct = v.percentage;
    if (!hasPercentage && totalPixels && totalPixels > 0) {
      finalPct = (v.pixel_count / totalPixels) * 100;
    }
    return { name: v.name, percentage: finalPct, pixel_count: v.pixel_count, count: v.count };
  });

  // Sort descending by percentage
  arr.sort((a, b) => b.percentage - a.percentage);
  return arr;
}

interface Detection {
  mineral_class: number;
  mineral_name: string;
  bbox: { x: number; y: number; width: number; height: number };
  area: number;
  center: { x: number; y: number };
}

interface Statistics {
  total_minerals_detected: number;
  total_regions: number;
  image_size: { width: number; height: number };
  class_distribution: Array<{
    mineral_class: number;
    mineral_name: string;
    pixel_count: number;
    percentage: number;
  }>;
  confidence_stats: { mean: number; min: number; max: number };
}

interface ResultsDisplayProps {
  results: {
    success?: boolean;
    detections?: Detection[];
    statistics?: Statistics;
    annotated_image?: string;
    segmentation_map?: string;
    error?: string;
  } | null;
  loading: boolean;
}

export default function ResultDetailsPanel({ results, loading }: ResultsDisplayProps) {
  const [isImageModalOpen, setIsImageModalOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<"annotated" | "segmentation">(
    "annotated"
  );

  if (loading) {
    return (
      <div className="loading-state">
        <div className="loading-spinner">
          <div className="spinner-ring"></div>
          <div className="spinner-ring active"></div>
        </div>
        <p className="loading-text">Analyzing mineral sample...</p>
        <p className="loading-subtext">This may take a few moments</p>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="empty-state">
        <div className="empty-icon">
          <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            />
          </svg>
        </div>
        <div className="empty-text">
          <h3>No results yet</h3>
          <p>Upload an image to begin analysis</p>
        </div>
      </div>
    );
  }

  if (results.error) {
    return (
      <div className="error-card">
        <div className="error-content">
          <div className="error-icon">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          </div>
          <div className="error-text">
            <h3>Error</h3>
            <p>{results.error}</p>
          </div>
        </div>
      </div>
    );
  }

  const { statistics, detections, annotated_image, segmentation_map } = results;

  return (
    <div className="results-display">
      {/* Statistics Overview */}
      {statistics && (
        <div className="stats-grid">
          <div className="stat-card">
            <p className="stat-label">Minerals Found</p>
            <p className="stat-value">{statistics.total_minerals_detected}</p>
          </div>
          <div className="stat-card">
            <p className="stat-label">Total Regions</p>
            <p className="stat-value">{statistics.total_regions}</p>
          </div>
          <div className="stat-card">
            <p className="stat-label">Avg Confidence</p>
            <p className="stat-value">{(statistics.confidence_stats.mean * 100).toFixed(1)}%</p>
          </div>
          <div className="stat-card">
            <p className="stat-label">Image Size</p>
            <p className="stat-value" style={{ fontSize: "1.3rem" }}>
              {statistics.image_size.width}×{statistics.image_size.height}
            </p>
          </div>
        </div>
      )}

      {/* Image Results */}
      {(annotated_image || segmentation_map) && (
        <div className="results-section">
          <div className="section-header">
            <h4 className="section-title">
              <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                />
              </svg>
              Analysis Visualization
            </h4>
            {annotated_image && segmentation_map && (
              <div className="tab-switcher">
                <button
                  onClick={() => setActiveTab("annotated")}
                  className={`tab-button ${activeTab === "annotated" ? "active" : ""}`}>
                  Annotated
                </button>
                <button
                  onClick={() => setActiveTab("segmentation")}
                  className={`tab-button ${activeTab === "segmentation" ? "active" : ""}`}>
                  Segmentation
                </button>
              </div>
            )}
          </div>

          <div className="image-viewer" onClick={() => setIsImageModalOpen(true)}>
            {activeTab === "annotated" && annotated_image && (
              <img src={`data:image/png;base64,${annotated_image}`} alt="Annotated analysis" className="result-image" />
            )}
            {activeTab === "segmentation" && segmentation_map && (
              <img src={`data:image/png;base64,${segmentation_map}`} alt="Segmentation map" className="result-image" />
            )}
            <div className="image-overlay">
              <div className="overlay-content">
                <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7"
                  />
                </svg>
                <p>Click to enlarge</p>
              </div>
            </div>
          </div>

          {/* Image Modal */}
          {isImageModalOpen && (
            <div className="modal-backdrop" onClick={() => setIsImageModalOpen(false)}>
              <button onClick={() => setIsImageModalOpen(false)} className="modal-close">
                <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
              <div className="modal-image-container" onClick={(e) => e.stopPropagation()}>
                {activeTab === "annotated" && annotated_image && (
                  <img src={`data:image/png;base64,${annotated_image}`} alt="Full size" className="modal-image" />
                )}
                {activeTab === "segmentation" && segmentation_map && (
                  <img src={`data:image/png;base64,${segmentation_map}`} alt="Full size" className="modal-image" />
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Detections List */}
      {detections && detections.length > 0 && (
        <div className="results-section">
          <h4 className="section-title">Detected Mineral Regions ({detections.length})</h4>
          <div className="detections-list">
            {detections.slice(0, 20).map((detection, index) => (
              <div key={index} className="detection-item">
                <div className="detection-info">
                  <h4>{detection.mineral_name}</h4>
                  <p>Area: {detection.area.toLocaleString()} px | Position: ({detection.center.x}, {detection.center.y})</p>
                </div>
                <span className="detection-class">Class {detection.mineral_class}</span>
              </div>
            ))}
            {detections.length > 20 && (
              <p className="more-detections">+ {detections.length - 20} more detections</p>
            )}
          </div>
        </div>
      )}

      {/* Mineral Distribution (aggregated) */}
      {statistics && statistics.class_distribution.length > 0 && (
        <div className="results-section">
          <h4 className="section-title">Mineral Composition Distribution</h4>
          <div className="distribution-list">
            {(() => {
              const aggregated = aggregateDistribution(statistics.class_distribution as any);
              // If the backend provided percentages (0-100) we used them; if
              // pixel_count values were used as fallback, they may not sum to
              // 100. We simply display the relative numbers as provided.
              const TOP_N = 12;
              const top = aggregated.slice(0, TOP_N);
              const rest = aggregated.slice(TOP_N);
              const restSum = rest.reduce((s, r) => s + r.percentage, 0);

              return (
                <>
                  {top.map((m, idx) => {
                    // Clamp width to [0,100] in case of noisy input
                    const pct = Math.max(0, Math.min(100, m.percentage));
                    return (
                      <div key={`${m.name}-${idx}`} className="distribution-item">
                        <div className="distribution-header">
                          <span className="mineral-name">{m.name}</span>
                          <span className="percentage">{pct.toFixed(2)}%</span>
                        </div>
                        <div className="progress-bar-bg">
                          <div className="progress-bar" style={{ width: `${pct}%` }}></div>
                        </div>
                      </div>
                    );
                  })}

                  {rest.length > 0 && (
                    <div className="distribution-item other-item">
                      <div className="distribution-header">
                        <span className="mineral-name">Other</span>
                        <span className="percentage">{restSum.toFixed(2)}%</span>
                      </div>
                      <div className="progress-bar-bg">
                        <div className="progress-bar" style={{ width: `${Math.max(0, Math.min(100, restSum))}%` }}></div>
                      </div>
                    </div>
                  )}
                </>
              );
            })()}
          </div>
        </div>
      )}
    </div>
  );
}
