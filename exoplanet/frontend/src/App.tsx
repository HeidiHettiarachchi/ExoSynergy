import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import Home from '../pages/Home';
import Header from '../pages/component/header';
import ComponentPage from '../pages/ComponentPage';
import Atmosphere from '../pages/Atmosphere';
import ExoDetect from '../pages/ExoDetection';
import ExtraMineral from '../pages/ExtraMineral';
import AtmosProfile from '../pages/spectrumAnalysis';
import StellarAnalysis from '../pages/StellarAnalysis';
import './App.css';

function AppContent() {
  const location = useLocation();
  const hideHeaderRoutes = ['/atmosphere'];
  const shouldHideHeader = hideHeaderRoutes.includes(location.pathname);

  return (
    <>
      {!shouldHideHeader && <Header />}
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/components" element={<ComponentPage />} />
        <Route path="/services" element={<ComponentPage />} />
        <Route path="/atmosphere" element={<Atmosphere />} />
        <Route path="/exoplanetDetection" element={<ExoDetect />} />
        <Route path="/extraMineral" element={<ExtraMineral />} />
        <Route path="/atmosphereProfile" element={<AtmosProfile />} />
        <Route path="/stellarAnalysis" element={<StellarAnalysis />} />
      </Routes>
    </>
  );
}

function App() {
  return (
    <Router>
      <AppContent />
    </Router>
  );
}


export default App
