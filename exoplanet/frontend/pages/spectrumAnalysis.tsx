import React, { useEffect, useRef, useState } from "react";
import type { JSX } from "react";
import "../pages/spectrumAnalysis.css";
import planetVid from "../src/assets/Vid.mp4";

// Icons
import { TbSettingsAutomation, TbChartDots3 } from "react-icons/tb";
import { MdOutlineFileUpload } from "react-icons/md";
import { FaCheckCircle } from "react-icons/fa";
import { TbTemperature } from "react-icons/tb";
import { MdOutlineScience } from "react-icons/md";
import { GiRingedPlanet } from "react-icons/gi";


// Interfaces
interface GasProfile {
  [gas: string]: number;
}
interface Biosignature {
  name: string;
  detected: boolean;
  reason: string;
  gases_involved: string[];
}
interface PlanetSimilarity {
  planet: string;
  score: number;
  similarity: number;
}
interface Habitability {
  score: number;
  grade: string;
  category: string;
  biosignatures?: Biosignature[];
  summary: string;
  factor_scores: Record<string, number>;
  planet_similarity?: PlanetSimilarity;
  profile: {
    planet_type: string;
    dominant_gas_fingerprint: string;

    greenhouse_intensity: string;
    greenhouse_heating_index?: number;
    greenhouse_effect?: number;

    atmospheric_density?: string;
    thermal_stability?: string;
    temperature_potential?: string;

    toxicity_index: number;
    toxicity_label: string;

    similar_atmospheres?: PlanetSimilarity[];
}
}
interface AnalysisResult {
  major_gases: Record<string, number>;
  trace_gases: Record<string, number>;
  gas_profile: GasProfile;
  habitability?: Habitability;
}

export default function SpectrumAnalysis(): JSX.Element {

  // Constants
  const videoRef = useRef<HTMLVideoElement | null>(null);

  // Preprocessing
  const [progress, setProgress] = useState<number>(0);
  const [fileUploaded, setFileUploaded] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [processStatus, setProcessStatus] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [rowCount, setRowCount] = useState<number | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [dataType, setDataType] = useState<string>("direct");

  // Analysis
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);

  /* Video playback speed */
  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = 0.5;
  }, []);

  /* Bar Animation */
  const animateProgress = (): void => {
    setProgress(0);
    let value = 0;
    const interval = window.setInterval(() => {
      value += Math.floor(Math.random() * 8) + 4;
      if (value >= 100) {
        value = 100;
        clearInterval(interval);
      }
      setProgress(value); 
    }, 120);
  };

  /* File Upload Handler */
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>): void => {
    const file = e.target.files?.[0];
    if (!file) return;

    setSelectedFile(file);
    setFileUploaded(true);
    setError(null);
    setProcessStatus(null);
    setRowCount(null);
    setAnalysisResult(null);
  };

  const handlePreprocess = async (): Promise<void> => {
    if (!selectedFile) {
      setError("");
      return;
    }

    setLoading(true);
    setError(null);
    setProcessStatus("Uploading and Processing Data...");
    animateProgress();

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("data_type", dataType);

      const res: Response = await fetch("http://127.0.0.1:8000/preprocess", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) throw new Error(data.detail || "Processing failed");

      setRowCount(data.row_count || 0);
      setProcessStatus("Processing Completed Successfully");
      setProgress(100);

      // if backend returned analysis results, show them immediately
      if (data.analysis) {
        setAnalysisResult(data.analysis as AnalysisResult);
      }

    } catch (err: any) {
      setError(err.message || "Processing failed.");
      setProcessStatus("Processing failed. Please check file format and data type.");
      setProgress(0);
    } finally {
      setLoading(false);
    }
  };

  // Filter gases
const sortedGases =
  analysisResult?.gas_profile
    ? Object.entries(analysisResult.gas_profile).sort((a, b) => b[1] - a[1])
    : [];

const topGases = sortedGases.slice(0, 2).map(g => g[0]);     // Highest 2
const traceGases = sortedGases.slice(2, 4).map(g => g[0]);   // Next 2

  return (
    <div className="spectrumDashboard">
      <section className="dashboard-hero">

        {/* Background Video */}
        <video
          ref={videoRef}
          className="bg-video-sa"
          src={planetVid}
          autoPlay
          loop
          muted
          playsInline
        />

        <div className="dashboard-overlay"></div>

        <div className="dashboard-content">
          <div className="glass-effect">

            <p className="system-tag">SPECTRUM · ANALYSIS · PROFILING · MODULE</p>

            <h1 className="dashboard-title">
              Atmospheric Analyzer
              <span>Dashboard</span>
            </h1>

            <div className="divider-line"></div>
            <hr className="divider2"/>

            <div className="function-col ">

            {/* ================= DATA LOADING PANEL ================= */}
            <div className="glass-effect preprocessing-panel">
                <div className="panel-heading">
                  <div className="sec-icon">
                    <TbSettingsAutomation style={{ color: "#2195f380", width: "30px", height: "30px" }} />
                  </div>
                  <h2>Spectrum Submission</h2>
                </div>

                <label className={`upload-box ${fileUploaded ? "uploaded" : ""}`}>
                  <input type="file" accept=".csv" onChange={handleFileUpload} hidden />
                  <div className="upload-icon">
                    {fileUploaded ? <FaCheckCircle className="upload-success-icon" /> : <MdOutlineFileUpload className="icons" />}
                  </div>
                  <p className="upload-title">{fileUploaded ? "File Uploaded" : "Upload Spectrum Data"}</p>
                  <p className="upload-subtitle">{fileUploaded ? selectedFile?.name : "Click to browse or drag & drop"}</p>
                </label>

                <div className="row-card">
                  <label>Data type:</label>
                  <select value={dataType} onChange={(e) => setDataType(e.target.value)}>
                    <option value="direct">Direct Imaging</option>
                    <option value="eclipse">Eclipse</option>
                    <option value="transmission">Transmission</option>
                  </select>
                </div>

                {processStatus && (
                  <div className={`status-card ${error ? "error" : "success"}`}>
                    <p className="status-text">{processStatus}</p>
                  </div>
                )}

                {rowCount !== null && (
                  <div className="row-card">
                    <p>Total data points:</p>
                    <span>{rowCount}</span>
                  </div>
                )}

                {loading && (
                  <div className="progress-container">
                    <div className="progress-header">
                      <span className="processing-text">Processing {selectedFile?.name}...</span>
                      <span className="progress-percent">{progress}%</span>
                    </div>
                    <div className="progress-bar">
                      <div className="progress-fill" style={{ width: `${progress}%` }} />
                    </div>
                  </div>
                )}

                <button className="buttons" onClick={handlePreprocess} disabled={loading || !fileUploaded}>
                  <TbChartDots3 style={{ width: "20px", height: "20px" }} />
                  {loading ? "Processing..." : "Load & Analyze Data"}
                </button>
            </div>

            {/* ================= ATMOSPHERIC VISUALIZATION ================= */}
            <div style={{marginTop: "1px"}}>
              <div className="atmo-panel preprocessing-panel">

                <div className="panel-heading" style={{ margin: "25px auto", alignItems: "center"}}>
                  <h2>Atmospheric Gas Profile</h2>
                </div>

                <div className="atmo-content">

                  {/* Planet */}
                  <div className="planet-container">
                    <div className="ring ring-1"></div>
                    <div className="ring ring-2"></div>
                    <div className="ring ring-3"></div>

                    <div className="planet">
                      <div className="planet-texture"></div>
                      <div className="planet-clouds"></div>
                      <div className="planet-glow"></div>
                      <div className="planet-light"></div>
                    </div>

                    <div className="label stratosphere">
                      <span className="dot"></span> UPPER ATMOSPHERE ({topGases.join(", ")})
                    </div>

                    <div className="label exosphere">
                      <span className="dot blue"></span> ATMOSPHERIC LAYER ({traceGases.join(", ")})
                    </div>
                  </div>

                  {/* Gas Bars */}
                  <div className="gas-list">
                    {(analysisResult
                      ? Object.entries(analysisResult.gas_profile).sort((a, b) => b[1] - a[1])
                      : ["H2O", "CO2", "CH4", "O2", "N2", "CO", "NH3"]  
                    ).map((item) => {
                      const gas = analysisResult ? item[0] : item;
                      const value = analysisResult ? item[1] : null;

                      return (
                        <div key={String(gas)} className="gas-card">
                          <div className="gas-header">
                            <div>
                              <h4>{gas}</h4>
                              <p>{analysisResult && value !== null ? `${(value as number).toFixed(3)}%` : "-"}</p>
                            </div>
                          </div>

                          <div className="gas-bar">
                            <div
                              className="gas-fill"
                              style={{
                                width: analysisResult && value !== null ? `${Math.min(value as number, 100)}%` : "0%",
                                opacity: analysisResult ? 1 : 0.2
                              }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

              </div>
            </div>
            </div>

            {/* ================= HABITABILITY & BIOSIGNATURES ================= */}
            <div className="function-col-2">
              <div className="glass-effect preprocessing-panel" style={{ animation: "none", boxShadow: "none" }}>

                <div className="panel-heading">
                  <h2 style={{textAlign: "center", margin: "auto"}}>Biosignatures & Habitability</h2>
                </div>

                {!analysisResult?.habitability ? (
                  <p className="preAnalysis-result">
                    Run analysis to generate habitability report
                  </p>
                ) : (

                  <div>
                    {/* ================= TOP SEC : SCORE ================= */}
                    <div
                      className="preprocessing-panel glass-effect"
                      style={{ animation: "none", boxShadow: "none", alignItems: "center", marginTop: "8px", padding: "40px 30px" }}>

                      <div className="score-circle">
                        <svg viewBox="0 0 120 120">
                          <defs>
                            <linearGradient id="habitGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                              <stop offset="0%" stopColor="#6fd4ffd1" />
                              <stop offset="50%" stopColor="#20a9c853" />
                              <stop offset="100%" stopColor="#185dbe" />
                            </linearGradient>

                            <filter id="glow">
                              <feGaussianBlur stdDeviation="3" result="coloredBlur" />
                              <feMerge>
                                <feMergeNode in="coloredBlur" />
                                <feMergeNode in="SourceGraphic" />
                              </feMerge>
                            </filter>
                          </defs>

                          <circle
                            cx="60"
                            cy="60"
                            r="50"
                            stroke="#1e293b"
                            strokeWidth="10"
                            fill="none"
                          />
                          <circle
                            cx="60"
                            cy="60"
                            r="50"
                            stroke="url(#habitGradient)"
                            strokeWidth="10"
                            fill="none"
                            strokeDasharray={314}
                            strokeDashoffset={
                              314 -
                              (314 * (analysisResult?.habitability?.score ?? 0)) / 100
                            }
                            strokeLinecap="round"
                            transform="rotate(-90 60 60)"
                            filter="url(#glow)"
                            style={{ transition: "stroke-dashoffset 1s ease" }}
                          />

                        </svg>

                        <div className="score-text">
                          <h1>{(analysisResult?.habitability?.score ?? 0).toFixed(1)}</h1>
                          <span>Atmospheric<br /> Habitability <br /> Index</span>
                        </div>
                      </div>

                      <div style={{ width: "100%" }}>
                        <div className="row-card grade-box" style={{background: "#185dbe63", marginBottom: "10px", padding: "12px"}}>
                          <p>Grade</p>
                          <p style={{ fontWeight: "100" }}>
                            {analysisResult?.habitability?.grade ?? "N/A"}
                          </p>
                        </div>

                        <div className="row-card grade-box" style={{ background: "#20a9c853" }}>
                          <p>Habitability</p>
                          <p style={{ fontWeight: "100" }}>
                            {analysisResult?.habitability?.category ?? "Unknown"}
                          </p>
                        </div>
                      </div>

                    </div>

                    {/* ================= BOTTOM SEC : BIOSIGNATURE SEC ================= */}
                    <div className="biosignature-section">

                      <h3>Detected Biosignatures</h3>

                      <div className="biosignature-list">                        
                         {(analysisResult?.habitability?.biosignatures ?? []).map((b, index) => (
                          <div
                            key={index}
                            className= "biosignature-card" >

                            <div style={{fontWeight: "600", display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "10px"}}>
                              <p style={{opacity: "0.9"}}>{b.name}</p>
                              <span
                                className={`biosignature-status ${
                                  b.detected ? "detected" : "not-detected"
                                }`}>
                                {b.detected ? "Detected" : "Not Detected"}
                              </span>
                            </div>

                            <div className="biosignature-reason">
                              {b.reason}
                            </div>

                            <div className="biosignature-gases">
                              {(b.gases_involved ?? []).map((gas, i) => (
                                <span key={i} className="gas-chip">
                                  {gas}
                                </span>
                              ))}
                            </div>
                          </div>
                        ))}
                      </div>

                      <div className="biosignature-card biosignature-reason">
                        <h4 style={{marginBottom: "8px", color: "#ffffffa0"}}>Scientific Summary</h4>
                        <p>
                          {analysisResult?.habitability?.summary ??
                            "No summary generated."}
                        </p>
                      </div>

                    </div>
                  </div>
                )}
              </div>

              {/* ======================= Atmospheric Similarity Section ======================== */}
              <div className="glass-effect preprocessing-panel" style={{ animation: "none", boxShadow: "none" }}>
                <div className="panel-heading">
                  
                  <h2 style={{ textAlign: "center", margin: "auto" }}>
                    Atmospheric & Similarity Analysis
                  </h2>
                </div>

                {!analysisResult?.habitability ? (
                  <p className="preAnalysis-result">
                    Run analysis to generate similarity report
                  </p>
                ) : (
                  <>
                    {analysisResult?.habitability?.profile && (
                      <div style={{marginTop: "13px"}}>

                        {/* Atmospheric Profile */}
                        <div className="profile-card" style={{padding: "30px 20px"}}>
                          <h3 style={{marginBottom: "20px", color: "#ffffff76"}}>Atmospheric Profile</h3>

                          <div className="planet-type-row">
                            <div className="planet-type-left">
                              <GiRingedPlanet className="planet-type-icon" />
                              <span className="planet-type-label">Planet Type</span>
                            </div>

                            <div className="planet-type-value">
                              {analysisResult.habitability.profile.planet_type}
                            </div>
                          </div>

                          {/* Gas Fingerprint */}
                          <div className="gas-section">
                            <p className="sub-title">Gas Fingerprint</p>

                            <div className="gas-badges">
                              {Object.entries(analysisResult.gas_profile)
                                .sort((a, b) => b[1] - a[1])
                                .slice(0, 5)
                                .map(([gas, value], i) => (
                                  <div key={i} className={`gas-badge gas-color-${i % 6}`}>
                                    <span className="gas-name">{gas}</span>
                                    <span className="gas-percent">{value.toFixed(2)}%</span>
                                  </div>
                                ))}
                            </div>
                          </div>

                          {/* Greenhouse + Toxicity */}
                          <div className="profile-stats">
                            <div className="stat-card greenhouse-card">
                              <div className="stat-icon">
                                <TbTemperature />
                              </div>

                              <div className="stat-info">
                                <p className="stat-title">Greenhouse Effect</p>
                                <span className="stat-value">
                                  {analysisResult.habitability.profile.greenhouse_intensity}
                                </span>
                              </div>
                            </div>

                            <div className="stat-card toxicity-card">
                              <div className="stat-icon">
                                <MdOutlineScience />
                              </div>

                              <div className="stat-info">
                                <p className="stat-title">Toxicity Level</p>
                                <span className="stat-value">
                                  {analysisResult.habitability.profile.toxicity_label} 
                                  ({analysisResult.habitability.profile.toxicity_index})
                                </span>
                              </div>
                            </div>
                          </div>

                          {/* Atmospheric Indicators */}
{/* Atmospheric Conditions */}
<div className="atmospheric-conditions">

  <div className="condition-row">
    <span className="condition-label">Greenhouse Heating Index</span>
    <span className="condition-value">
      {analysisResult.habitability.profile.greenhouse_heating_index?.toFixed(2)}
    </span>
  </div>

  <div className="condition-row">
    <span className="condition-label">Greenhouse Effect</span>
    <span className="condition-value">
      {analysisResult.habitability.profile.greenhouse_effect?.toFixed(2)}
    </span>
  </div>

  <div className="condition-row">
    <span className="condition-label">Atmospheric Density</span>
    <span className="condition-value">
      {analysisResult.habitability.profile.atmospheric_density}
    </span>
  </div>

  <div className="condition-row">
    <span className="condition-label">Thermal Stability</span>
    <span className="condition-value">
      {analysisResult.habitability.profile.thermal_stability}
    </span>
  </div>

  <div className="condition-row">
    <span className="condition-label">Temperature Potential</span>
    <span className="condition-value">
      {analysisResult.habitability.profile.temperature_potential}
    </span>
  </div>

</div>
                        </div>

                        {/* Solar System Similarity */}
                        {analysisResult?.habitability?.profile?.similar_atmospheres && (
                          <div className="profile-card"  style={{marginTop: "13px"}}>

                            <h3 style={{marginBottom: "25px", color: "#ffffffa0"}}>
                              Atmospheric Similarity
                            </h3>

                            {analysisResult.habitability.profile.similar_atmospheres
                              .slice(0, 4)
                              .map((planet, i) => (
                                <div key={i} className="similarity-row">
                                <div className="similarity-top">
                                  <span className="planet-name">{planet.planet}</span>
                                  <span className="similarity-value">{planet.similarity.toFixed(1)}%</span>
                                </div>

                                <div className="similarity-bar">
                                  <div
                                    className="similarity-fill"
                                    style={{ width: `${planet.similarity}%` }}
                                  />
                                </div>
                              </div>

                              ))}

                          </div>
                        )}

                      </div>
                    )}

                  </>
                )}
              </div>
            </div>

          </div>
        </div>
      </section>
    </div>
  );
}