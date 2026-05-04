import React, { useEffect, useRef, useState } from "react";
import type { JSX } from "react";
import "../pages/spectrumAnalysis.css";
import planetVid from "../src/assets/Vidbg.mp4";

// Icons
import { TbSettingsAutomation, TbChartDots3 } from "react-icons/tb";
import { MdOutlineFileUpload } from "react-icons/md";
import { FaCheckCircle } from "react-icons/fa";
import { TbTemperature } from "react-icons/tb";
import { MdOutlineScience } from "react-icons/md";
import { GiRingedPlanet } from "react-icons/gi";

// 3D animation
import * as THREE from "three";

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
  const planetType = analysisResult?.habitability?.profile?.planet_type || "Unknown";

  // 3D Planet 
  const [labelPositions, setLabelPositions] = useState<any[]>([]);
  const planetRef = useRef<HTMLDivElement | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);

  // Planet Similarity
  const normalizePlanetName = (name: string) => {
  return name.trim().toLowerCase();
  };
  const planetImages: Record<string, string> = {
    earth: "/src/assets/planets/Earth.jpg",
    venus: "/src/assets/planets/Venus.jpg",
    mars: "/src/assets/planets/Venus.jpg",
    jupiter: "/src/assets/planets/jupiter.png",
    saturn: "/src/assets/planets/Saturn.png",
    uranus: "/src/assets/planets/Uranus.png",
    neptune: "/src/assets/planets/iceGiant.png",
  };

  // 3D Planet Effects
  useEffect(() => {
    if (!planetRef.current) return;

    if (rendererRef.current) {
      planetRef.current.innerHTML = "";
      rendererRef.current.dispose();
    }

    const scene = new THREE.Scene();

    const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
    camera.position.z = 1.9;

    const renderer = new THREE.WebGLRenderer({
      alpha: true,
      antialias: true,
    });

    renderer.setSize(390, 390);
    renderer.setPixelRatio(window.devicePixelRatio);

    planetRef.current.innerHTML = "";
    planetRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // ================= PLANET =================
    const geometry = new THREE.SphereGeometry(1, 64, 64);

    const texture = new THREE.TextureLoader().load(
      "/src/assets/texture.png"
    );

    texture.colorSpace = THREE.SRGBColorSpace;

    const material = new THREE.MeshStandardMaterial({
      map: texture,
    });

    const sphere = new THREE.Mesh(geometry, material);

    const gasAnchors: THREE.Vector3[] = [
      new THREE.Vector3(2.82, 1.25, 0),   // 0
      new THREE.Vector3(2.32, 0.28, 0.5), // 1
      new THREE.Vector3(2.01, -0.3, 0.6), // 2
      new THREE.Vector3(0.4, -0.5, 0.6),  // 3
      new THREE.Vector3(0.4, 0.4, 0.2),  // 4
    ];
    scene.add(sphere);

    // ================= ATMOSPHERE GLOW =================
    const glowGeo = new THREE.SphereGeometry(1.12, 64, 64);
    const glowMat = new THREE.ShaderMaterial({
      transparent: true,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide,
      uniforms: {
        glowColor: { value: new THREE.Color(0x00E1F5) },
      },
      vertexShader: `
        varying vec3 vNormal;
        varying vec3 vPosition;

        void main() {
          vNormal = normalize(normalMatrix * normal);
          vPosition = (modelViewMatrix * vec4(position, 1.0)).xyz;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform vec3 glowColor;
        varying vec3 vNormal;
        varying vec3 vPosition;

        void main() {
          float intensity = pow(0.7 - dot(vNormal, normalize(vPosition)), 2.5);
          gl_FragColor = vec4(glowColor, intensity);
        }
      `,
    });

    const glowMesh = new THREE.Mesh(glowGeo, glowMat);
    scene.add(glowMesh);

    // ================= LIGHT =================
    const light = new THREE.DirectionalLight(0xffffff, 1.2);
    light.position.set(3, 2, 5);
    scene.add(light);

    scene.add(new THREE.AmbientLight(0xffffff, 0.4));

    // ================= MOUSE INTERACTION =================
    let isDragging = false;
    let prevX = 0;

    planetRef.current.addEventListener("mousedown", (e) => {
      isDragging = true;
      prevX = e.clientX;
    });

    window.addEventListener("mouseup", () => {
      isDragging = false;
    });

    window.addEventListener("mousemove", (e) => {
      if (!isDragging) return;

      const delta = e.clientX - prevX;
      sphere.rotation.y += delta * 0.005;
      glowMesh.rotation.y += delta * 0.005;
      prevX = e.clientX;
    });

  const animate = () => {
    requestAnimationFrame(animate);

    if (!isDragging) {
      sphere.rotation.y += 0.0015;
    }

    sphere.updateMatrixWorld();
    camera.updateMatrixWorld();

    const tempV = new THREE.Vector3();

    const newPositions = gasAnchors.map((pos) => {
      tempV.copy(pos);
      tempV.project(camera);

      const x = (tempV.x * 0.5 + 0.5) * renderer.domElement.clientWidth;
      const y = (-tempV.y * 0.5 + 0.5) * renderer.domElement.clientHeight;

      return {
        x,
        y,
        visible: tempV.z < 1,
        screenX: tempV.x,
      };
    });

    setLabelPositions(newPositions);
    renderer.render(scene, camera);
  };
    animate();
    return () => {
      renderer.dispose();
      geometry.dispose();
      material.dispose();
      
    };
  }, []);

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

      const res: Response = await fetch("https://exosynergy-backend.onrender.com//preprocess", {
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

  // Sort Gases
  const sortedGases = analysisResult?.gas_profile
    ? Object.entries(analysisResult.gas_profile).sort((a, b) => b[1] - a[1])
    : [];

  const top3Gases = sortedGases.slice(0, 3);
  const next2Gases = sortedGases.slice(3, 5);

  // Planet type description
  const planetTypeData: Record<
  string,
  { image: string; description: string }
  > = {
    "Gas Giant": {
      image: "/src/assets/planets/gasGiant.png",
      description:
        "Massive planet dominated by Hydrogen (H2) and Helium (He) with thick gaseous layers and no solid surface.",
    },
    "Ice Giant / Sub-Neptune": {
      image: "/src/assets/planets/iceGiant.png",
      description:
        "Intermediate-sized planet rich in volatiles like Water (H2O), methane (CH4), and ammonia (NH3).",
    },
    "Earth-like": {
      image: "/src/assets/planets/Earth.jpg",
      description:
        "Balanced Nitrogen-Oxygen atmosphere with potential for life-supporting conditions.",
    },
    "Venus-like (CO2-dominated)": {
      image: "/src/assets/planets/Venus.jpg",
      description:
        "Dense CO2 atmosphere with extreme greenhouse heating and high pressure.",
    },
    "Rocky / Mixed Atmosphere": {
      image: "/src/assets/planets/Rocky.png",
      description:
        "Solid surface with a mix of gases and variable environmental conditions.",
    },
    "Hycean World Candidate": {
      image: "/src/assets/planets/Hycean.jpg",
      description:
        "Ocean-covered planet with a hydrogen-rich atmosphere, considered a strong candidate for hosting life.",
    },
    "Volcanically Active Rocky": {
      image: "/src/assets/planets/Rocky.png",
      description:
        "Rocky planet with active volcanism, releasing gases like SO2 and H2S into the atmosphere.",
    },
    "Titan-like (N2-dominated)": {
      image: "/src/assets/planets/Rocky.png",
      description:
        "Cold world dominated by nitrogen with thick hazy atmosphere, similar to Saturn’s moon Titan.",
    },

    "Unknown": {
      image: "/src/assets/planets/Rocky.png",
      description: "Planet classification is uncertain based on available data.",
    },
  };

  const selectedPlanet = planetTypeData[planetType] || planetTypeData["Unknown"];


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

          {/*================== Left Phase============================= */}
            <div className="left-panel-stack">

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

              {/* ================= PLANET TYPE WITH DESCRIPTION ================= */}
              <div className="planet-type-card">
              <div className="planet-type-image">
                <img src={selectedPlanet.image} alt={planetType} />
              </div>

              {/* CONTENT */}
              <div className="planet-type-content">
                <div className="planet-type-title-row">
                  <h3 style={{color: "white"}}>{planetType}</h3>
                </div>

                <p className="planet-type-description">
                  {selectedPlanet.description}
                </p>
              </div>

              </div>

              {/* ================= BIOSIGNATURE & HABITABILITY SECTION ================= */}
              <div className="glass-effect preprocessing-panel" style={{marginTop: "2px"}}>

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
            </div>

            {/* ================= ATMOSPHERIC VISUALIZATION ================= */}
            <div>

            {/*=================== PLANET SETION WITH GAS BARS ==========================*/}
            <div style={{marginTop: "1px"}}>
              <div className="atmo-panel preprocessing-panel">

                <div className="panel-heading"
                  style={{
                    margin: "0px auto",
                    marginBottom: "25px",
                    width: "100%",
                    display: "flex",
                    justifyContent: "center",
                    alignItems: "center",
                    padding: "12px 0",
                    background: "rgba(0, 1, 25, 0.63)",
                    backdropFilter: "blur(10px)",
                    border: "1px solid rgba(255, 255, 255, 0.08)",
                    borderRadius: "12px 12px 0 0",
                  }}>
                  <h2>Atmospheric Gas Profile</h2>
                </div>

                {/* Planet */}
                  <div className="planet-container">
                    <div ref={planetRef} className="planet-3d"></div>
                    <div className="gas-overlay">

                      {/* ================= RIGHT SIDE (TOP 3 GASES) ================= */}
                      {top3Gases.map(([gas, value], i) => {
                        const pos = labelPositions[i];
                        if (!pos?.visible) return null;

                        return (
                          <div
                            key={gas}
                            className="gas-label-wrapper"
                            style={{
                              position: "absolute",
                              left: pos.x,
                              top: pos.y,
                              transform: "translate(0px, 50%)", 
                            }}
                          >
                            <svg className="gas-connector-svg" width="120" height="80">
                              <g stroke="#e1eef4c4" strokeWidth="1" fill="none">
                                <line x1="0" y1="0" x2="80" y2="0" />
                                <line x1="80" y1="0" x2="80" y2="40" />
                              </g>

                              <circle cx="0" cy="0" r="5" fill="#6fd4ffb5" />
                              <circle cx="0" cy="0" r="10" fill="#bce9fc56" opacity="0.5" />
                            </svg>

                            <div className="gas-tag-right">
                              <span className="gas-name">{gas}</span>
                              <span className="gas-value">{value.toFixed(1)}%</span>
                            </div>
                          </div>
                        );
                      })}
                      {/* ================= LEFT SIDE (NEXT 2 GASES) ================= */}
                      {next2Gases.map(([gas, value], i) => {
                        const pos = labelPositions[i + 3];
                        if (!pos?.visible) return null;

                        return (
                          <div
                            key={gas}
                            className="gas-label-wrapper"
                            style={{
                              position: "absolute",
                              left: pos.x,
                              top: pos.y,
                              transform: "translate(0, 0)", 
                            }}
                          >
                            {/* DOT + LINE SYSTEM */}
                            <svg className="gas-connector-svg" width="140" height="80">

                              <g stroke="#e1eef4c4" strokeWidth="1" fill="none">
                                
                                {/* L-SHAPE  */}
                                <line x1="0" y1="0" x2="-70" y2="0" />
                                <line x1="-70" y1="0" x2="-80" y2="-20" />
                              </g>

                              {/* DOT */}
                              <circle cx="0" cy="0" r="5" fill="#6fd4ffb5" />
                              <circle cx="0" cy="0" r="10" fill="#bce9fc56" opacity="0.5" />
                            </svg>

                            {/* LABEL  */}
                            <div className="gas-tag-left">
                              <span className="gas-name">{gas}</span>
                              <span className="gas-value">{value.toFixed(2)}%</span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                <div className="atmo-content">

                  {/* Gas Bars */}
                  <div className="gas-list">
                    {(analysisResult
                      ? Object.entries(analysisResult.gas_profile)
                          .filter(([_, value]) => Number(value) > 0.001) // ✅ STRICT FILTER
                          .sort((a, b) => Number(b[1]) - Number(a[1]))
                      : ["H2O", "CO2", "CH4", "O2", "N2", "CO", "NH3"]
                    ).map((item) => {
                      const gas = analysisResult ? item[0] : item;
                      const value = analysisResult ? item[1] : null;

                      return (
                        <div key={String(gas)} className="gas-card">
                          <div className="gas-header">
                            <div>
                              <h4>{gas}</h4>
                              <p>
                                {analysisResult && value !== null
                                  ? `${Number(value).toFixed(3)}%`
                                  : "-"}
                              </p>
                            </div>
                          </div>

                          <div className="gas-bar">
                            <div
                              className="gas-fill"
                              style={{
                                width:
                                  analysisResult && value !== null
                                    ? `${Math.min(Number(value), 100)}%`
                                    : "0%",
                                opacity: analysisResult ? 1 : 0.2,
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

            {/* ======================= ATMOSPHERIC PROFILING & SIMILARITY SEC ======================== */}
              <div className="glass-effect preprocessing-panel" style={{ animation: "none", boxShadow: "none" }}>

                <div className="panel-heading">
                  <h2 style={{ textAlign: "center", margin: "auto" }}>
                    Atmospheric Profiling & Similarity Analysis
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
                          <h3 style={{marginBottom: "20px", color: "#ffffff76"}}>Atmospheric Conditions</h3>                       

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

                          {/* Atmospheric Conditions */}
                          <div className="atmospheric-conditions">

                            <div className="condition-row">
                              <span className="condition-label">Greenhouse Heating Index</span>
                              <span className="condition-value">
                                {analysisResult.habitability.profile.greenhouse_heating_index?.toFixed(2)}
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
                        <div className="similarity-container">
                          {analysisResult?.habitability?.profile?.similar_atmospheres && (() => {

                            const planets = analysisResult.habitability.profile.similar_atmospheres;
                            const topPlanet = planets[0];
                            const otherPlanets = planets.slice(1, 3);

                            return (
                              <div className="similarity-grid">

                                {/* ================= LEFT: TOP MATCH ================= */}
                                <div className="similarity-main">

                                  <img
                                    src={
                                    planetImages[normalizePlanetName(topPlanet.planet)] ||
                                    "/src/assets/planets/default.png"
                                  }
                                    alt={topPlanet.planet}
                                    className="planet-image"
                                  />

                                  <div className="planet-info">
                                    <h2>{topPlanet.planet}</h2>

                                    <div className="main-bar">
                                      <div
                                        className="main-fill"
                                        style={{ width: `${topPlanet.similarity}%` }}
                                      />
                                    </div>

                                    <span className="main-percentage">
                                      {topPlanet.similarity.toFixed(1)}% Similar
                                    </span>

                                    {/* optional characteristics */}
                                    <div className="planet-tags">
                                      <span>Atmospheric Match</span>
                                      <span>Composition Similarity</span>
                                    </div>
                                  </div>

                                </div>

                                {/* ================= RIGHT: OTHER MATCHES ================= */}
                                <div className="similarity-side">

                                  {otherPlanets.map((p, i) => (
                                    <div key={i} className="side-card">

                                      <div className="side-top">
                                        <span>{p.planet}</span>
                                        <span>{p.similarity.toFixed(1)}%</span>
                                      </div>

                                      <div className="side-bar">
                                        <div
                                          className="side-fill"
                                          style={{ width: `${p.similarity}%` }}
                                        />
                                      </div>

                                    </div>
                                  ))}

                                </div>

                              </div>
                            );
                          })()}
                        </div>

                      </div>
                    )}

                  </>
                )}
              </div>
            </div>          
              
            </div>

          </div>
        </div>
      </section>
    </div>
  );
}