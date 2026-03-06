import React, { useEffect, useRef, useState } from "react";
import type { JSX } from "react";
import "../pages/spectrumAnalysis.css";
import planetVid from "../src/assets/Vid.mp4";

// Icons
import { TbSettingsAutomation, TbChartDots3 } from "react-icons/tb";
import { MdOutlineFileUpload } from "react-icons/md";
import { FaCheckCircle } from "react-icons/fa";

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
    dominant_gas_fingerprint:string;
    greenhouse_intensity: string;
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

  // Spectrum data analysis
  // const handleAnalyze = async (): Promise<void> => {
  //   if (!selectedFile) {
  //     setError("");
  //     return;
  //   }

  //   setLoading(true);
  //   setError(null);
  //   setProcessStatus("Running Atmospheric Analysis...");

  //   try {
  //     const formData = new FormData();
  //     formData.append("file", selectedFile);

  //     const res: Response = await fetch(`http://127.0.0.1:8000/analyze/${dataType}`, {
  //       method: "POST",
  //       body: formData,
  //     });

  //     const data: AnalysisResult = await res.json();

  //     if (!res.ok) throw new Error((data as any).detail || "Failed Analysing Spectrum Data");

  //     setAnalysisResult(data);
  //     setProcessStatus("Atmospheric Profiling Completed");

  //   } catch (err: any) {
  //     setError(err.message || "Failed Analysing Spectrum Data");
  //     setProcessStatus("Failed Analysing Spectrum Data");
  //   } finally {
  //     setLoading(false);
  //   }
  // };


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

            {/* ================= PREPROCESS PANEL ================= */}
            <div className="glass-effect preprocessing-panel">
                <div className="panel-heading">
                  <TbSettingsAutomation className="icons" />
                  <h2>Spectrum Preprocessing</h2>
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

            <div>
              {/* ================= ATMOSPHERIC VISUALIZATION ================= */}
              <div className="atmo-panel preprocessing-panel">

                <div className="panel-heading" style={{ marginTop: "10px" }}>
                  <h2>Atmospheric Gas Profile</h2>
                </div>

                {/* button hides itself once analysis has been performed */}
                {/* {analysisResult === null && (
                  <button className="buttons" onClick={handleAnalyze} disabled={loading || !fileUploaded}
                    style={{ marginTop: "10px", opacity: "0.7" }}>
                    Predict Gases
                  </button>
                )} */}

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

                    <div className="label exosphere">
                      <span className="dot"></span> EXOSPHERE (H, He)
                    </div>

                    <div className="label stratosphere">
                      <span className="dot blue"></span> STRATOSPHERE (CH4, CO2)
                    </div>
                  </div>

                  {/* Gas Bars */}
                  <div className="gas-list">
                    {(analysisResult
                      ? Object.entries(analysisResult.gas_profile).sort((a, b) => b[1] - a[1])
                      : ["H2O", "CO2", "CH4", "O2", "N2", "CO", "NH3"]  
                    ).map((item, index) => {
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

                  <div className="function-col-2">

                    {/* ================= LEFT SIDE : SCORE ================= */}
                    <div
                      className="preprocessing-panel glass-effect"
                      style={{ animation: "none", boxShadow: "none", alignItems: "center" }}
                    >

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
                          <span>Habitability Index</span>
                        </div>

                      </div>

                      <div style={{ width: "100%" }}>

                        <div
                          className="row-card grade-box"
                          style={{
                            background: "#185dbe63",
                            marginBottom: "10px",
                            padding: "10px",
                          }}
                        >
                          <p>Grade</p>
                          <p style={{ fontWeight: "100" }}>
                            {analysisResult?.habitability?.grade ?? "N/A"}
                          </p>
                        </div>

                        <div
                          className="row-card grade-box"
                          style={{ background: "#20a9c853" }}
                        >
                          <p>Habitability</p>
                          <p style={{ fontWeight: "100" }}>
                            {analysisResult?.habitability?.category ?? "Unknown"}
                          </p>
                        </div>

                      </div>

                    </div>

                    {/* ================= RIGHT SIDE : BIOSIGNATURE SEC ================= */}
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
                      <div className= "function-col-2" style={{marginTop: "13px"}}>

                        {/* Atmospheric Profile */}
                        <div className="profile-card">
                          <h3 style={{marginBottom: "8px", color: "#ffffffa0"}}>Atmospheric Profile</h3>

                          <div className="row-card grade-box"
                            style={{background: "#185dbe63", marginBottom: "10px", padding: "10px", marginTop: "20px"}}>
                            <p>Planet Type:</p>
                            <p>{analysisResult.habitability.profile.planet_type}</p>
                          </div>

                          {/* Gas Fingerprint */}
                          <div className="gas-section">
                            <p className="sub-title">Gas Fingerprint</p>

                            <div className="gas-badges">
                              {analysisResult.habitability.profile.dominant_gas_fingerprint
                                .split(",")
                                .map((gas, i) => (
                                  <div key={i} className="gas-badge">
                                    {gas.trim()}
                                  </div>
                                ))}
                            </div>
                          </div>

                          {/* Greenhouse + Toxicity */}
                          <div className="profile-stats">

                            <div className="stat-box">
                              <p className="sub-title">Green House</p>

                              <span className="greenhouse-badge">
                                {analysisResult.habitability.profile.greenhouse_intensity}
                              </span>
                            </div>

                            <div className="stat-box">
                              <p className="sub-title">Toxicity</p>
                              <span className="toxicity-badge">
                                {analysisResult.habitability.profile.toxicity_label} (
                                {analysisResult.habitability.profile.toxicity_index})
                              </span>
                            </div>
                          </div>
                        </div>

                        {/* Solar System Similarity */}
                        {analysisResult?.habitability?.profile?.similar_atmospheres && (
                          <div className="profile-card">

                            <h3 style={{marginBottom: "25px", color: "#ffffffa0"}}>
                              Atmospheric Similarity
                            </h3>

                            {analysisResult.habitability.profile.similar_atmospheres
                              .slice(0, 4)
                              .map((planet, i) => (
                                <div key={i} className="similarity-row">
                                  <div className="similarity-top">
                                    <span style={{color: "#ffffffa0"}}>{planet.planet}</span>
                                    <span style={{color: "#ffffffa0"}}>{planet.similarity.toFixed(1)}%</span>
                                  </div>

                                  <div className="gas-bar">
                                    <div
                                      className="gas-fill"
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