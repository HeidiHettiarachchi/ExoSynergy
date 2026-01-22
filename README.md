AI-INTEGRATED FRAMEWORK FOR EXOPLANET DETECTION, STELLAR
& PLANETARY RESOURCE ANALYSIS
IT4010 Research Project – July 2025
Group 249
========================================================

--------------------------------------------------------
1. PROJECT DESCRIPTION
--------------------------------------------------------
Exoplanetary discovery is a cornerstone of modern astrophysics, enabling the study of
planetary system formation, habitability, and the potential for extraterrestrial life.
However, existing exoplanet detection and characterization approaches are fragmented,
noise-sensitive, and often limited to a single observational modality.

This project proposes an AI-integrated framework that unifies four critical astrophysical
and planetary analysis tasks into a single intelligent platform:

• Hybrid exoplanet detection using transit photometry and direct imaging  
• Stellar age and metallicity prediction from photometric data  
• Exoplanetary Atmospheric Spectrum Analysis for Biosignatures and Profiling using Spectral Data 
• Identifying extraterrestrial minerals and resources using spectral and imaging data  

By leveraging machine learning, deep learning, and multimodal data fusion, the system
enhances detection accuracy, reduces false positives, and enables scalable analysis of
large astronomical datasets. The framework supports future space missions, habitability
assessment, and long-term extraterrestrial exploration.

--------------------------------------------------------
2. SYSTEM COMPONENTS
--------------------------------------------------------

Component 01 – Hybrid AI-Based Exoplanet Detection & Characterization  
--------------------------------------------------
Owner : H.A.H.E.K. Hettiarachchi  
GitHub Repo: https://github.com/HeidiHettiarachchi/exo-hybrid-detection

This component develops a hybrid artificial intelligence framework that combines
indirect transit photometry and direct imaging techniques to improve the accuracy
and reliability of exoplanet detection and characterization.

By fusing time-series light curve analysis with high-contrast imaging pipelines,
the system reduces false positives, enhances faint planetary signals, and produces
robust detection confidence scores.

Key Sub-Modules:
• Transit Photometry Detection Model  
  - Preprocesses and denoises Kepler and TESS light curves  
  - Uses CNN and RNN architectures to detect periodic transit signals  
  - Predicts orbital parameters and transit characteristics  

• Direct Imaging Enhancement Model  
  - Applies PSF subtraction methods (KLIP, Annular PCA)  
  - Uses AI-based denoisers for speckle noise suppression  
  - Generates signal-to-noise ratio (SNR) and likelihood maps  

• Hybrid Fusion & Validation Engine  
  - Integrates outputs from transit and imaging pipelines  
  - Uses attention-based and cross-validation techniques  
  - Produces final detection likelihood and confidence scores  

Objectives:
• Detect exoplanets using both direct and indirect observational data  
• Reduce noise and false positives in exoplanet detection  
• Fuse multi-domain signals for higher detection confidence  
• Generate interpretable detection maps and evaluation metrics  

The novelty of this component lies in being one of the first AI-driven frameworks
to intelligently integrate transit photometry and direct imaging into a unified
detection pipeline, significantly improving robustness and reliability.
  

Component 02 – Stellar Classification & Exoplanet Suitability Prediction  
--------------------------------------------------
Owner : Fernando M.K.C  
GitHub Repo: https://github.com/KCxRULZZ/Star-Suitability-Predictor.git

This component focuses on intelligent stellar characterization and habitability
assessment using multi-band photometric data (u, g, r, i, z) from large-scale
astronomical surveys such as SDSS.

By leveraging supervised machine learning models trained on photometric
magnitudes and derived color indices, the system predicts fundamental stellar
properties and evaluates the suitability of stars for hosting exoplanets.

Key Sub-Modules:
• Spectral Type Prediction Model  
  - Classifies stars into spectral classes (O, B, A, F, G, K, M) using ugriz data  
• Effective Temperature (Teff) Prediction Model  
  - Predicts stellar surface temperature through regression models  
• Metallicity Class Prediction Model  
  - Categorizes stars into metallicity classes (Metal-poor Halo ,Thick Disk Stars ,Thin Disk Stars ,Metal-rich Stars)  
• Iron-to-Hydrogen Ratio (FeH) Prediction Model  
  - Estimates Fe/H abundance as a continuous regression output  
• Exoplanet Suitability Prediction Model  
  - Combines predicted stellar parameters (spectral type, Teff, metallicity)
    to assess a star’s suitability for hosting exoplanets  

Objectives:
• Predict stellar spectral types from photometric inputs  
• Estimate effective temperature (Teff)  
• Predict iron-to-hydrogen ratio (Fe/H) and metallicity class  
• Evaluate exoplanet hosting suitability based on stellar characteristics  

The novelty of this component lies in extending traditional stellar parameter
prediction beyond age estimation to a holistic stellar profiling pipeline,
culminating in an AI-driven exoplanet suitability score. This approach enables
scalable and cost-effective target star selection for future exoplanet surveys
without reliance on spectroscopy.
  

Component 03 – Exoplanetary Atmospheric Spectrum Analysis for Biosignatures and Profiling using Spectral Data
-------------------------------------------------------------------------------------------------------------
Owner : Tissera W A H  
GitHub Repo: https://github.com/HeidiHettiarachchi/ExoSynergy.git

This component analyzes exoplanetary atmospheric spectral data to identify the presence of key gases, estimate their relative composition, and assess potential biosignatures. It generates  atmospheric profiles, assigns confidence levels to detected gases, and compares planetary atmospheres with known reference planets. The component supports efficient and consistent atmospheric characterization while reducing manual spectral interpretation.

Key Sub-Modules:
• Spectral Preprocessing & Normalization
  - Handles noise reduction and continuum normalization of atmospheric spectra
  - Prepares spectral data for reliable gas detection

• Atmospheric Gas Detection  
  - Detects atmospheric gases based on characteristic absorption bands  
  - Identifies the availability of key gases such as H₂O, CO₂, CH₄, O₂, NH₃, and CO 

• Gas Confidence Estimation
  - Assigns confidence scores to each detected gas
  - Distinguishes dominant, trace, and uncertain gas detections

• Atmospheric Composition Profiling
  - Constructs summery chemical composition profiles from detected gases
  - Estimates relative gas contribution percentages

• Biosignature Scoring & Interpretation
  - Evaluates biologically relevant gas combinations
  - Assigns likelihood levels (None, Low, Moderate, Strong)
  - Computes biosignature likelihood levels with explainable reasoning

• Planetary Atmosphere Comparison
  - Compares atmospheric profiles with reference planets
  - Identifies closest planetary analog based on similarity scores

• Visualization & Result Presentation
  - Displays gas composition, confidence levels, and biosignature scores
  - Supports clear interpretation through graphical outputs

Objectives:
• Detect and identify key atmospheric gases from exoplanetary spectral data
• Estimate relative gas composition and generate atmospheric composition profiles
• Evaluate biosignature relevance based on detected gas combinations
• Compare exoplanetary atmospheres with known planetary reference profiles
• Reduce manual spectral analysis and improve consistency in atmospheric interpretation 
• Create a unified framework for exoplanetary spectral data analysis

The novelty of this component lies in its end-to-end integration of atmospheric gas detection, confidence estimation, biosignature scoring, and planetary comparison into a single automated framework. Unlike existing approaches that focus on isolated detection tasks, this system provides a holistic and explainable atmospheric characterization pipeline, enabling faster, consistent, and scalable analysis of exoplanetary atmospheres.
  

Component 04 – Identifying extraterrestrial minerals and resources using spectral and imaging data  
--------------------------------------------------
Owner : C I Abeywickrama  
GitHub Repo:

This component focuses on automatically identifying planetary minerals and generating detailed mineral maps using deep learning applied to hyperspectral data. It supports planetary research by enabling fast, consistent, and scalable surface analysis without manual interpretation.

Key Sub-Modules:
• Hyperspectral Data Processing  
  - Loads and structures CRISM and planetary hyperspectral datasets  
  - Normalizes spectral values and removes noise and inconsistencies  

• Mineral Segmentation Model  
  - Uses a U-Net based deep learning architecture 
  - Performs pixel-level mineral identification and segmentation
  - Integrates both spectral and spatial information
 

• Spectral Inference Extension
  - Applies learned mineral spectral features for analysis beyond Mars  
  - Supports probabilistic mineral inference from non-spatial spectra 

• Mineral Mapping & Visualization  
  - Generates interpretable mineral maps from model outputs  
  - Displays dominant minerals, confidence levels, and mixed regions
  - Highlights uncertain areas for expert review

Objectives:
• Automatically identify minerals from hyperspectral planetary data 
• Generate pixel-level mineral segmentation maps 
• Reduce manual analysis and expert dependency  
• Integrate spectral and spatial features for higher accuracy
• Support spectral inference beyond Mars without surface mapping
  

The novelty of this component is that this introduces a U-Net based framework that unifies spectral and spatial learning for hyperspectral mineral segmentation, with a physics-informed extension toward mineral inference beyond Mars.

--------------------------------------------------------
3. NOVELTY OF THE PROJECT
--------------------------------------------------------
• First AI framework to integrate both direct imaging and transit-based
  exoplanet detection into a unified system  
• Intelligent fusion of multi-domain astronomical signals  
• Automated atmospheric biosignature scoring with explainable outputs  
• Scalable stellar characterization using only photometric data  
• Combined supervised and unsupervised learning for extraterrestrial
  mineral and resource mapping  
• End-to-end AI-driven astrophysical analysis pipeline  

--------------------------------------------------------
4. WHY THIS PROJECT MATTERS
--------------------------------------------------------
• Improves accuracy and reliability of exoplanet detection  
• Enables scalable stellar characterization without costly spectroscopy  
• Advances automated habitability and biosignature assessment  
• Supports future space exploration and in-situ resource utilization (ISRU)  
• Demonstrates real-world application of AI in astrophysics and planetary science  

--------------------------------------------------------
5. DATASETS & RESOURCES
--------------------------------------------------------
Transit Photometry Data:
• Kepler Mission – https://archive.stsci.edu/kepler/
• TESS Mission   – https://archive.stsci.edu/tess/

Direct Imaging Data:
• VLT/SPHERE
• Keck/NIRC2
• Starshade Telescope (Simulated / Callibrated Data)

Stellar Photometric Data:
• Kaggle Stellar UGRIZ – https://www.kaggle.com/datasets/diraf0/sloan-digital-sky-survey-dr18?resource=download

Atmospheric Spectral Data:
• NASA Exoplanet Archive - https://exoplanetarchive.ipac.caltech.edu/cgi-bin/atmospheres/nph-firefly?atmospheres
• HITRAN - https://hitran.org/lbl/#

Planetary Mineral Data:
• Mars orbital data explorer - https://ode.rsl.wustl.edu/mars/

--------------------------------------------------------
6. SYSTEM DIAGRAM & LOGO
--------------------------------------------------------
Project Logo:
https://github.com/HeidiHettiarachchi/ExoSynergy/blob/main/exoplanet/frontend/src/assets/logo.png

System / Conceptual Diagram:
https://github.com/HeidiHettiarachchi/ExoSynergy/blob/main/exoplanet/frontend/src/assets/System-Diagram.png

--------------------------------------------------------
7. GROUP DETAILS
--------------------------------------------------------

1. H.A.H.E.K. Hettiarachchi  
   Registration No : IT22323248  
   Component       : Hybrid AI-Based Exoplanet Detection  

2. Fernando M.K.C  
   Registration No : IT22346254  
   Component       : Stellar Age & Metallicity Prediction  

3. Tissera W A H  
   Registration No : IT22026866  
   Component       : Exoplanetary Atmospheric Spectrum Analysis for Biosignatures and Profiling using Spectral Data  

4. C I Abeywickrama  
   Registration No : IT22343048  
   Component       : Identifying extraterrestrial minerals and resources using spectral and imaging data 

--------------------------------------------------------
8. TECHNOLOGIES USED
--------------------------------------------------------
• Python  
• TensorFlow / PyTorch  
• Scikit-learn  
• AstroPy, VIP-HCI  
• NumPy, Pandas, SciPy  
• Matplotlib, Plotly  
• Machine Learning & Deep Learning  
• Computer Vision & Spectral Analysis  

--------------------------------------------------------
9. LICENSE & USAGE
--------------------------------------------------------
This repository is intended for academic and research purposes under the
SLIIT IT4010 Research Project. Proper citation is required for any reuse of code,
models, or derived results.

========================================================
END OF README
========================================================
