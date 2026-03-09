import React from 'react';
import './ComponentPage.css';
import exodetect from '../src/assets/exodetect.jpg';
import Atmo from '../src/assets/Atmo.jpg';
import Star from '../src/assets/stellar.jpeg';
import Mars from '../src/assets/mars.jpeg';
import ExoDetection from './ExoDetection';
import ExtraMineral from './ExtraMineral';  
import AtmosProfile from './AtmosProfile';
import StellarAnalysis from './StellarAnalysis';
import { useNavigate } from 'react-router-dom';


function ComponentPage() {
const navigate = useNavigate();

const handleExpDetect = () => {
  window.open('https://exoplanet-detection-sp06.onrender.com/', '_blank');
}

const handleMineral = () => {
  window.open('https://mineral-identification-frontend.onrender.com/', '_blank');
}

const handleStellar = () => {   
    window.open('https://frontend-production-449c.up.railway.app/', '_blank');
}

const handleAtmos = () => {     
    window.open('https://atmospheric-analysis-frontend-production.up.railway.app/atmosphereProfile/', '_blank');
}


  return (
    <div className="component-page-wrapper">
      <div className="page-container">
        <div className="content-wrapper">
          
          <div className="cards-grid">
            {/* Card 1 */}
            <div className="card">
              <img className="card-image" src={exodetect} alt="Golf" />
              <h3 className="card-title">Hybrid Exoplanet Detection</h3>
              <p className="card-description">Offering a first-of-its-kind hybrid method to detect exoplanets using both direct and indirect approaches.</p>
              <button className="card-button" onClick={handleExpDetect}>Explore →</button>
            </div>

            {/* Card 2 */}
            <div className="card">
              <img className="card-image" src={Mars} alt="Basketball" />
              <h3 className="card-title">Extraterrestrial Mineral Identifier</h3>
              <p className="card-description">Styles made for your games</p>
              <button className="card-button" onClick={handleMineral}>Explore →</button>
            </div>

            {/* Card 3 */}
            <div className="card">
              <img className="card-image" src={Atmo} alt="Planrt Atmosphere" />
              <h3 className="card-title">Atmospheric Spectrum Analyzer</h3>
              <p className="card-description">Atmospheric Spectrum Analysis for Biosignatures and Profiling</p>
              <button className="card-button" onClick={handleAtmos}>Explore →</button>
            </div>

            {/* Card 4 */}
            <div className="card">
              <img className="card-image" src={Star} alt="Soccer" />
              <h3 className="card-title">Stellar Classification and Suitability Predictor.
</h3>
              <p className="card-description">Gear up for the pitch</p>
              <button className="card-button" onClick={handleStellar}>Explore →</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ComponentPage;
