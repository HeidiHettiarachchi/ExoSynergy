import atmoBg from '../src/assets/atmoBg.jpg';
import './Atmosphere.css';
import { useNavigate } from 'react-router-dom';

const Atmosphere = () => {
  const navigate = useNavigate();
  return (
    <div className="atmosphere-wrapper">
      {/* HERO SECTION */}
      <section className="atmosphere-hero">
        <div className="hero-background">
          <img src={atmoBg} alt="Atmospheric Analysis Background" className="bg-image" />
          <div className="overlay"></div>
        </div>
        
        <div className="hero-content">
          <div className="content-container">
            <div className="hero-title">
              <h1>
                <span className="atmo-prefix">ATMOSPHERA</span>
                {/* <span className="atmo-main">SPHERE</span> */}
              </h1>
              <div className="title-underline"></div>
            </div>
            
            <div className="hero-description">
              <p className="main-tagline">
                Atmospheric Analysis for Biosignatures & Profiling
              </p>
              <p className="sub-tagline">
                Advanced spectral data interpretation for exoplanetary atmospheric composition and habitability assessment
              </p>
            </div>
            
            <div className="hero-actions">
              <button className="primary-action" onClick={() => navigate('/spectrumAnalysis')}>
                <span className="btn-text">Start Analysis</span>
                <span className="btn-arrow">→</span>
              </button>
            </div>
            
            <div className="powered-by">
              <span className="powered-text">Powered by</span>
              <span className="brand-name">ExoSynergy</span>
            </div>
          </div>
        </div>
        
        {/* Animated Elements */}
        <div className="floating-particles">
          <div className="particle particle-1"></div>
          <div className="particle particle-2"></div>
          <div className="particle particle-3"></div>
          <div className="particle particle-4"></div>
          <div className="particle particle-5"></div>
        </div>
        
        <div className="atmospheric-layers">
          <div className="layer layer-1"></div>
          <div className="layer layer-2"></div>
          <div className="layer layer-3"></div>
        </div>
      </section>
    </div>
  );
};

export default Atmosphere;
