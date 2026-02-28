import React, { useEffect, useRef, useState } from "react";
import { JSX } from "react";
import "../pages/spectrumAnalysis.css";
import planetVid from "../src/assets/Vid.mp4";

// Icons
import { TbSettingsAutomation, TbChartDots3 } from "react-icons/tb";
import { MdOutlineFileUpload } from "react-icons/md";
import { FaCheckCircle } from "react-icons/fa";

export default function SpectrumAnalysis(): JSX.Element {

  // Constants
  const videoRef = useRef<HTMLVideoElement | null>(null);

  // Preprocess Constants
  const [progress, setProgress] = useState<number>(0);
  const [fileUploaded, setFileUploaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [processStatus, setProcessStatus] = useState<string | null>(null);
  const [processedDataPath, setProcessedDataPath] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [rowCount, setRowCount] = useState<number | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  // Video Speed
  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = 0.5;
  }, []);

  // Animate progress bar
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

  // Handle file upload
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setSelectedFile(file);
    setFileUploaded(true);
    setError(null);
    setProcessStatus("");
  };

  // Handle preprocessing
  const handlePreprocess = async (): Promise<void> => {
    if (!selectedFile) {
      setError("Please upload a CSV file first.");
      return;
    }

    setLoading(true);
    setError(null);
    setProcessStatus("Uploading and Processing Data...");
    animateProgress();

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("data_type", "direct");

      const res: Response = await fetch("http://127.0.0.1:8000/preprocess", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "Processing failed");
      }

      setRowCount(data.row_count || 0); 
      setProcessStatus("Processing Completed Successfully");
      setProgress(100);

    } catch (err: any) {
      console.error(err);
      setError(err.message || "Processing failed.");
      setProcessStatus("Preprocessing failed");
      setProgress(0);
      
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="spectrumDashboard">
      <section className="dashboard-hero">

        {/* Video */}
        <video
          ref={videoRef}
          className="bg-video-sa"
          src={planetVid}
          autoPlay
          loop
          muted
          playsInline
        />

        {/* Overlay */}
        <div className="dashboard-overlay"></div>

        {/* Content */}
        <div className="dashboard-content">

          {/* Title Section */}
          <div className="glass-effect">
            <p className="system-tag">SPECTRUM · ANALYSIS · PROFILING · MODULE</p>

            <h1 className="dashboard-title">
              Atmospheric Analyzer
              <span>Dashboard</span>
            </h1>

            <div className="divider-line"></div>
            <hr className="divider2"/>

            {/* Preprocessing Panel */}
            <div className="Function-col-1 glass-effect preprocessing-panel">
              
              {/* Header */}
              <div className="panel-heading">
                <TbSettingsAutomation className="icons" />
                <h2>Spectrum Preprocessing</h2>
              </div>

              {/* Upload Box */}
              <label className={`upload-box ${fileUploaded ? "uploaded" : ""}`}>
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleFileUpload}
                  style={{ display: "none" }}
                />
                <div className="upload-icon">
                  {fileUploaded ? <FaCheckCircle className="upload-success-icon" /> : <MdOutlineFileUpload className="icons" />}
                </div>
                <p className="upload-title">{fileUploaded ? "File Uploaded" : "Upload Spectrum Data"}</p>
                <p className="upload-subtitle">{fileUploaded ? selectedFile?.name : "Click to browse or drag & drop"}</p>
              </label>

              {/* Status Card */}
              {processStatus && (
                <div className={`status-card ${error ? "error" : "success"}`}>
                  <p className="status-text">{processStatus}</p>
                </div>
              )}

              {/* Row count */}
              {rowCount !== null && (
                <div className="row-card">
                  <p>Total data points:</p>
                  <span>{rowCount}</span>
                </div>
              )}

              {/* Progress Bar */}
              {loading && (
                <div className="progress-container">
                  <div className="progress-header">
                    <span className="processing-text">Processing {selectedFile?.name}...</span>
                    <span className="progress-percent">{progress}%</span>
                  </div>
                  <div className="progress-bar">
                    <div className="progress-fill" style={{ width: `${progress}%` }}></div>
                  </div>
                </div>
              )}

              {/* Load Button */}
              <button className="buttons" onClick={handlePreprocess} disabled={loading || !fileUploaded}>
                <TbChartDots3 style={{ width: "20px", height: "20px" }} />
                {loading ? "Processing..." : "Load Spectrum Data"}
              </button>
            </div>
            
          </div>
        </div>
      </section>
    </div>
  );
}