import homevid from '../src/assets/homevid.mp4';
import aboutImg from '../src/assets/space.png';
import './Home.css';
import { useNavigate } from 'react-router-dom';

const Home = () => {
  const navigate = useNavigate();

  return (
    <div className="home-wrapper">

      {/* HERO SECTION */}
      <section className="hero">
        <video src={homevid} autoPlay loop muted className="bg-vid" />
        <div className="overlay"></div>

        <div className="content">
          <h1>
            Explore the Universe with <span>ExoSynergy</span>
          </h1>
          <p>
            Intelligent analysis platform for exoplanets, atmospheres, and stellar systems.
          </p>

          <div className="hero-buttons">
            <button className="primary-btn" onClick={() => navigate('/services')}>
              Get Started
            </button>
            <button
              className="secondary-btn"
              onClick={() => {
                const el = document.getElementById("about");
                if (el) {
                  el.scrollIntoView({ behavior: "smooth" });
                }
              }}
            >
              Learn More
            </button>
          </div>
        </div>

        {/* Scroll indicator */}
        <div className="scroll-down"></div>
      </section>


      {/* ABOUT SECTION */}
      <section className="about" id="about">
        <div className="about-container">
          {/* LEFT IMAGE */}
          <div className="about-image">
            <img src={aboutImg} alt="ExoSynergy" />
          </div>

          {/* RIGHT CONTENT */}
          <div className="about-content">
            <h2>About ExoSynergy</h2>
            <p>
              ExoSynergy is an integrated space analysis platform designed to explore and interpret exoplanetary systems through multiple specialized modules. It enables users to detect exoplanets from observational data, analyze atmospheric compositions using spectral inputs, identify potential mineral signatures, and estimate key stellar parameters of host stars. By combining these capabilities into a single interactive environment, ExoSynergy simplifies complex astronomical analysis and provides meaningful insights into the structure, composition, and behavior of distant worlds.
            </p>
            <button className="primary-btn" onClick={() => navigate('/services')}>
              Explore Platform
            </button>
          </div>
        </div>
      </section>

    </div>
  );
};

export default Home;