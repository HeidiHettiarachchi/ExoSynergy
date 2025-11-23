import { useState } from 'react';
import reactLogo from './assets/react.svg';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import viteLogo from '/vite.svg';
import Home from '../pages/Home';
import Header from '../pages/component/header';
import About from '../pages/About';
import ContactUs from '../pages/ContactUs';
import Simulation from '../pages/Simulation';
import ComponentPage from '../pages/ComponentPage';
import Login from '../pages/Login';
import ExoDetect from '../pages/ExoDetection';
import ExtraMineral from '../pages/ExtraMineral';
import AtmosProfile from '../pages/AtmosProfile';
import StellarAnalysis from '../pages/StellarAnalysis';
import './App.css';

function App() {
  return (

    
    <Router>
      <Header />

      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/about" element={<About />} />
        <Route path="/simulation" element={<Simulation />} />
        <Route path="/contact" element={<ContactUs />} />
        <Route path="/login" element={<Login />} />
        <Route path="/services" element={<ComponentPage />} />
        <Route path="/exoplanetDetection" element={<ExoDetect />} />
        <Route path="/extraMineral" element={<ExtraMineral />} />
        <Route path="/atmosphereProfile" element={<AtmosProfile />} />
        <Route path="/stellarAnalysis" element={<StellarAnalysis />} />
      </Routes>
    </Router>
  );
}


export default App
