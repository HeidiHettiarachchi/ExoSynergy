import React from 'react';
import homevid from '../src/assets/homevid.mp4';
import './Home.css';
import ComponentPage from './ComponentPage';
import { useNavigate } from 'react-router-dom';

const Home = () => {

const navigate = useNavigate();

  const handleGetStarted = () => {
    navigate('/services');
  };


  return (
    <div className="main">
      <video src={homevid} autoPlay loop muted className="bg-vid"/>
      <div className="overlay"></div>
      <div className="content">
        <h1>Welcome to ExoSynergy!</h1>
        <p>Explore astrophysics simulations and visualize exoplanet data.</p>
        <button className="button-85" onClick={handleGetStarted}>Get Started</button>
      </div>
    </div>
  );
};

export default Home;