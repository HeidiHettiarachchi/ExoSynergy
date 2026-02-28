import React, { useEffect, useRef, useState } from "react";
import "../pages/spectrumAnalysis.css";
import planetVid from "../src/assets/Vid.mp4";
import { TbSettingsAutomation } from "react-icons/tb";
import { MdOutlineFileUpload } from "react-icons/md";
import { TbChartDots3 } from "react-icons/tb";

export default function SpectrumAnalysis() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [progress, setProgress] = useState(78);
  

  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.playbackRate = 0.5;
    }
  }, []);

  return (
    <div className="spectrumDashboard">

      <section className="dashboard-hero">

        {/* Video section*/}
        <video
          ref={videoRef}
          className="bg-video-sa"
          src={planetVid}
          autoPlay
          loop
          muted
          playsInline
        />

        {/* Dark overlay */}
        <div className="dashboard-overlay"></div>

        {/* Overlay Content */}
        <div className="dashboard-content">

        {/* Div section 1 : Title */}
          <div className="glass-effect">
            <p className="system-tag">SPECTRUM · ANALYSIS · PROFILING · MODULE</p>

            <h1 className="dashboard-title">
              Atmospheric Analyzer
              <span>Dashboard</span>
            </h1>

            <div className="divider-line"></div>

            <hr className="divider2"/>

          {/* Div sectioon 2 : Function section 1 */}          
          <div className="Function-col-1 glass-effect preprocessing-panel">
            
            {/* Panel header */}
            <div className="panel-heading">
              <TbSettingsAutomation className="icons" />
              <h2>Spectrum Preprocessing</h2>
            </div>

            {/* Upload Box */}
            <div className="upload-box">
              <div className="upload-icon">
                <MdOutlineFileUpload className="icons" />
              </div>
              <p className="upload-title">Upload Spectrum Data</p>
              <p className="upload-subtitle">Click to browse or drag & drop</p>
            </div>

            {/* Progress */}
            <div className="progress-container">
              <div className="progress-header">
                <span className="processing-text">Processing Kepler-186f data...</span>
                <span className="progress-percent">{progress}%</span>
              </div>
              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{ width: `${progress}%` }}
                ></div>
              </div>
            </div>

            {/* Button - Preprocess section */}
            <button className="buttons">
              <TbChartDots3 style={{ width: "20px", height: "20px"}} />
              Load Spectrum Data
            </button>


          </div>

          </div>


            

          </div>

      </section>
    </div>
  );
}