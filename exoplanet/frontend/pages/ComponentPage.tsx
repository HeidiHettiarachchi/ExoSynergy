import './ComponentPage.css';

// Component Images
import exodetect from "../src/assets/exodetect.png";
import minaralDetect from "../src/assets/mineraldetect.png";
import atmosDetect from "../src/assets/atmodetect.png";
import stellarDetect from "../src/assets/stellarpredict.png";


function ComponentPage() {

  return (
    <div className="component-page-wrapper">
      <div className="page-container">
        <div className="content-wrapper">

        <div className="module-grid">

          {/* ================= MODULE 1 - Exoplanet Detection ================= */}
          <div className="module-card large exo-module exo-purple">
            <div className="inner-card card1">

              <div className="card-header">
                <div>
                  <span className="tag  pulse-tag">MODULE ALPHA</span>
                  <h3>EXODIOS — Hybrid AI Based Exoplanet Detection Framework</h3>
                  <div className="sub-text">
                    Exoplanet Detection System
                  </div>
                </div>
                <span className="icon glow">◎</span>
              </div>

              <div className="image video holo-frame">
                <img src={exodetect} />
                <div className="scanline"></div>
                <div className="frame-glow"></div>
              </div>

              <p>
                Hybrid AI-Based Exoplanet Detection Framework is a system that combines multiple data-driven techniques to detect 
                and analyze exoplanets from observational data. It enhances detection accuracy by processing complex astronomical 
                signals and identifying potential exoplanetary candidates efficiently.
              </p>

              <div className="card-footer">
                <button className="btn exo-btn" onClick={() => window.open('https://exodios.onrender.com/', '_blank')}>EXPLORE MODULE</button>
              </div>

            </div>
          </div>

          {/* ================= MODULE 2 - Mineral Detection ================= */}
          <div className="module-card small exo-module exo-orange">
            <div className="inner-card">

              <div className="card-header">
                <div>
                  <span className="tag yellow pulse-tag">MODULE BETA</span>
                  <h3>HYPERSPECTRA - AI Based Martian Mineral Identifier</h3>
                  <div className="sub-text">
                    Spectral mineral classification engine
                  </div>
                </div>
                <span className="icon glow">◇</span>
              </div>

              <div className="image square holo-frame">
                <img src={minaralDetect} />
                <div className="frame-glow"></div>
              </div>

              <p>
                HYPERSPECTRA uses CRISM hyperspectral data and deep learning to identify minerals on the Martian surface.
                It generates accurate, spatially coherent mineral maps for geological analysis and exploration planning.
              </p>

              <button className="btn outline-cyan exo-btn" onClick={() => window.open('https://mineral-identification-frontend.onrender.com/', '_blank')}>EXPLORE MODULE</button>
            </div>
          </div>

          {/* ================= MODULE 3 - Atmospheric Analyzer ================= */}
          <div className="module-card small exo-module exo-blue">
            <div className="inner-card">

              <div className="card-header">
                <div>
                  <span className="tag purple pulse-tag">MODULE GAMMA</span>
                  <h3>ATMOSPHERA - Atmospheric Analysis for Biosignatures </h3>
                  <div className="sub-text">
                    Gas composition Estimation & Biosignature Analysis
                  </div>
                </div>
                <span className="icon glow">≋</span>
              </div>

              <div className="image square holo-frame">
                <img src={atmosDetect} />
                <div className="frame-glow"></div>
              </div>

              <p>
                ATMOSPHERA detects key atmospheric gases, analyzes biosignatures from spectral data. 
                It generates a structured atmospheric profile and evaluates habitability based on chemical distribution and atmospheric characteristics.
              </p>

              <button className="btn outline-purple exo-btn" onClick={() => window.location.href = 'https://exosynergy-vm6u.onrender.com/Atmosphere'}>EXPLORE MODULE</button>
            </div>
          </div>

          {/* ================= MODULE 4 - Stellar Predictor ================= */}
          <div className="module-card large exo-module exo-red">
            <div className="inner-card">

              <div className="card-header">
                <div>
                  <span className="tag red pulse-tag">MODULE DELTA</span>
                  <h3>Stellar Parameter Estimation & Stellar Suitability Classification</h3>
                  <div className="sub-text">
                    Stellar suitability evolution for hosting exoplanet
                  </div>
                </div>
                <span className="icon glow">☼</span>
              </div>

              <div className="image video holo-frame">
                <img src={stellarDetect} />
                <div className="scanline"></div>
                <div className="frame-glow"></div>
              </div>

              <p>
                Stellar Parameter Estimation & Stellar Suitability Classification is a module that estimates key stellar properties 
                and evaluates how suitable a star is for hosting potentially habitable exoplanets based on its physical characteristics.
              </p>

              <div className="card-footer">
                <button className="btn red exo-btn" onClick={() => window.open('https://star-suitability-predictor-production-5e6f.up.railway.app/', '_blank')}>EXPLORE MODULE</button>
              </div>

            </div>
          </div>

        </div>

        </div>
      </div>
    </div>
  );
}

export default ComponentPage;