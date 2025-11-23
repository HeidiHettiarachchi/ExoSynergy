import React from "react";
import About from '../About';
import ContactUs from '../ContactUs';
import Simulation from '../Simulation';
import ComponentPage from '../ComponentPage';
import Login from '../Login';
import './Header.css';
import logo from '../../src/assets/logo.png';


const Header = () => {
    return (
        <nav className="header-nav">
            <div className="header-container">
             <a href="/" className="header-logo">
                    <img src={logo} className="logo-img" alt="ExoSynergy Logo" />
                    <span className="logo-text">ExoSynergy</span>
                </a>

                <button className="mobile-menu-btn" type="button">
                    <span className="sr-only">Open main menu</span>
                    <svg className="w-6 h-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <path stroke="currentColor" strokeLinecap="round" strokeWidth="2" d="M5 7h14M5 12h14M5 17h14" />
                    </svg>
                </button>

                <div className="header-menu">
                    <ul className="nav-list">
                        {/* <li><a href="/" className="nav-link active">Home</a></li> */}
                        <li><a href="/about" className="nav-link">About</a></li>
                        <li><a href="/simulation" className="nav-link">Simulation</a></li>
                        <li><a href="/contact" className="nav-link">Contact Us</a></li>

                        <li><a href="/login" className="nav-link">Log in</a></li>
                    </ul>
                </div>
            </div>
        </nav>
    )
}


export default Header;