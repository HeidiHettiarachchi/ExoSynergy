import { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from '../pages/Home';
import Header from '../pages/component/Header'; // fix casing!
import About from '../pages/About';
import ContactUs from '../pages/ContactUs';
import Simulation from '../pages/Simulation';
import ComponentPage from '../pages/ComponentPage';
import Login from '../pages/Login';
import './App.css';
// comment
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
      </Routes>
    </Router>
  );
}

export default App