import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter

# ==============================
# MASTER PREPROCESSOR
# ==============================
def preprocess_spectrum(input_csv_path, data_type):
    """
    data_type: 'direct' | 'eclipse' | 'transmission'
    """

    df = pd.read_csv(input_csv_path)

    if data_type == "direct":
        return preprocess_direct_imaging(df)

    elif data_type == "eclipse":
        return preprocess_eclipse(df)

    elif data_type == "transmission":
        return preprocess_transmission(df)

    else:
        raise ValueError("Invalid data_type. Use: direct | eclipse | transmission")


# ==============================
# DIRECT IMAGING
# ==============================
def preprocess_direct_imaging(df):

    df = df[["CENTRALWAVELNG", "FLAM"]].rename(columns={
        "CENTRALWAVELNG": "wave_length",
        "FLAM": "F_lambda"
    })

    df = df.dropna()
    df = df[df["F_lambda"] > 0]
    df = df.sort_values("wave_length")

    wavelength = df["wave_length"].values
    flux = df["F_lambda"].values

    # Continuum fitting
    spline = UnivariateSpline(wavelength, flux, s=0.5 * len(wavelength))
    continuum = spline(wavelength)

    # Normalization
    df["flux_norm"] = flux / continuum

    # Smoothing
    df["flux_smooth"] = savgol_filter(df["flux_norm"], window_length=9, polyorder=2)

    # ML features
    features = extract_spectral_features(wavelength, df["flux_smooth"].values)

    return features


# ==============================
# ECLIPSE DATA
# ==============================
def preprocess_eclipse(df):

    df = df[[
        "Central Wave.\n(microns)",
        "Band Width\n(microns)",
        "Eclipse Depth\n(%)"
    ]].rename(columns={
        "Central Wave.\n(microns)": "wave_length",
        "Band Width\n(microns)": "band_width",
        "Eclipse Depth\n(%)": "eclipse_depth"
    })

    df = df.dropna()
    df = df.sort_values("wave_length")

    # Normalize eclipse depth
    df["depth_norm"] = (df["eclipse_depth"] - df["eclipse_depth"].mean()) / df["eclipse_depth"].std()

    # Feature engineering
    features = {
        "mean_depth": df["depth_norm"].mean(),
        "std_depth": df["depth_norm"].std(),
        "max_depth": df["depth_norm"].max(),
        "min_depth": df["depth_norm"].min(),
        "mean_bandwidth": df["band_width"].mean(),
        "spectral_coverage": df["wave_length"].max() - df["wave_length"].min()
    }

    return pd.DataFrame([features])


# ==============================
# TRANSMISSION DATA
# ==============================
def preprocess_transmission(df):

    df = df[[
        "Planet Rad. (+err)",
        "Planet Rad. (-err)",
        "Planet Rad. Prov.",
        "Transit Mid-Point\n(days)",
        "Transit Mid-Point (+err)"
    ]].rename(columns={
        "Planet Rad. (+err)": "rad_plus",
        "Planet Rad. (-err)": "rad_minus",
        "Planet Rad. Prov.": "rad_prov",
        "Transit Mid-Point\n(days)": "transit_mid",
        "Transit Mid-Point (+err)": "transit_err"
    })

    df = df.dropna()

    # Physical features
    df["radius_uncertainty"] = df["rad_plus"] + df["rad_minus"]
    df["transit_stability"] = 1 / (df["transit_err"] + 1e-6)

    features = {
        "mean_radius": df["rad_prov"].mean(),
        "mean_uncertainty": df["radius_uncertainty"].mean(),
        "transit_variability": df["transit_mid"].std(),
        "mean_stability": df["transit_stability"].mean(),
    }

    return pd.DataFrame([features])


# ==============================
# FEATURE EXTRACTION (FOR ML)
# ==============================
def extract_spectral_features(wavelength, flux):

    return pd.DataFrame([{
        "flux_mean": np.mean(flux),
        "flux_std": np.std(flux),
        "flux_max": np.max(flux),
        "flux_min": np.min(flux),
        "spectral_range": np.max(wavelength) - np.min(wavelength),
        "slope": np.polyfit(wavelength, flux, 1)[0]
    }])