import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter

def preprocess_spectrum(input_csv_path_or_df, data_type):
   
    if isinstance(input_csv_path_or_df, pd.DataFrame):
        df = input_csv_path_or_df.copy()
    else:
        df = pd.read_csv(input_csv_path_or_df)
    
    df.columns = df.columns.str.strip()
    
    try:
        if data_type == "direct":
            return preprocess_direct_imaging(df)
        elif data_type == "eclipse":
            return preprocess_eclipse(df)
        elif data_type == "transmission":
            return preprocess_transmission(df)
        else:
            raise ValueError("Invalid data_type. Use: direct | eclipse | transmission")
    except KeyError as ke:
        raise ValueError(f"Preprocessing failed for '{data_type}' data, missing column: {ke}")
    except Exception:
        raise


# ------------------------------
# Direct Imaging Preprocessing
# ------------------------------
def preprocess_direct_imaging(df):
    df = df[["CENTRALWAVELNG", "FLAM"]].dropna()
    df = df[df["FLAM"] > 0].sort_values("CENTRALWAVELNG")
    
    if len(df) < 4:
        return create_empty_spectral_features()
    
    wavelength = df["CENTRALWAVELNG"].values
    flux = df["FLAM"].values
    
    try:
        spline = UnivariateSpline(wavelength, flux, s=0.5 * len(wavelength))
        continuum = spline(wavelength)
    except Exception:
        return create_empty_spectral_features()
    
    flux_norm = flux / continuum
    flux_smooth = savgol_filter(flux_norm, window_length=9, polyorder=2)
    
    return extract_spectral_features_binned(wavelength, flux_smooth)

# ------------------------------
# Eclipse Data Preprocessing
# ------------------------------
def preprocess_eclipse(df):
    required = ["CENTRALWAVELNG", "ESPECLIPDEP", "ESPECLIPDEPERR1", "ESPECLIPDEPERR2"]
    missing = set(required) - set(df.columns)

    if missing:
        raise KeyError(f"missing {missing}")
    
    df = df[required]
    
    for col in df.columns:
        df.loc[:, col] = pd.to_numeric(df[col], errors="coerce")
    
    df = df.dropna(subset=["CENTRALWAVELNG", "ESPECLIPDEP"])
    df = df.sort_values("CENTRALWAVELNG")
    
    if len(df) < 2:
        return create_empty_spectral_features()
    
    wavelength = df["CENTRALWAVELNG"].values
    depth = df["ESPECLIPDEP"].values
    depth_err = df["ESPECLIPDEPERR1"].fillna(0).values + df["ESPECLIPDEPERR2"].fillna(0).values
    
    return extract_spectral_features_binned(wavelength, depth, depth_err)

# --------------------------------
# Transmission Data Preprocessing
# --------------------------------
def preprocess_transmission(df):
    cols = [
        "CENTRALWAVELNG", "PL_TRANDEP", "PL_TRANDEPERR1", "PL_TRANDEPERR2",
        "PL_RATROR", "PL_RATRORERR1", "PL_RATRORERR2",
        "PL_RADJ", "PL_RADJERR1", "PL_RADJERR2",
        "ST_RAD", "ST_RADERR1", "ST_RADERR2"
    ]
    
    missing = set(cols) - set(df.columns)

    if missing:
        raise KeyError(f"missing {missing}")
    df = df[cols]
    
    for col in df.columns:
        df.loc[:, col] = pd.to_numeric(df[col], errors="coerce")
    
    df = df.dropna(subset=["CENTRALWAVELNG", "PL_TRANDEP", "PL_RATROR", "PL_RADJ"])
    df = df.sort_values("CENTRALWAVELNG")
    
    if len(df) < 2:
        return create_empty_spectral_features()
    
    wavelength = df["CENTRALWAVELNG"].values
    depth = df["PL_TRANDEP"].values
    depth_err = df["PL_TRANDEPERR1"].fillna(0).values + df["PL_TRANDEPERR2"].fillna(0).values
    
    return extract_spectral_features_binned(wavelength, depth, depth_err)


# Spectral Binning 
def create_empty_spectral_features():
    features_dict = {}
    for i in range(208):
        features_dict[f"lambda_{i}"] = 0.0
    return pd.DataFrame([features_dict])


def bin_spectrum(wavelength, intensity, n_bins=52):

    wl_min, wl_max = wavelength.min(), wavelength.max()
    bin_edges = np.linspace(wl_min, wl_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    binned_intensity = np.zeros(n_bins)
    binned_width = np.zeros(n_bins)
    binned_noise = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (wavelength >= bin_edges[i]) & (wavelength < bin_edges[i+1])
        if mask.sum() > 0:
            binned_intensity[i] = np.mean(intensity[mask])
            binned_width[i] = bin_edges[i+1] - bin_edges[i]
            binned_noise[i] = np.std(intensity[mask]) if len(intensity[mask]) > 1 else 0.0
        else:
            binned_intensity[i] = 0.0
            binned_width[i] = bin_edges[i+1] - bin_edges[i]
            binned_noise[i] = 0.0
    
    return binned_intensity, binned_width, binned_noise, bin_centers


def extract_spectral_features_binned(wavelength, intensity, intensity_err=None):

    # Normalize intensity
    if intensity.max() > 0:
        intensity = (intensity - intensity.mean()) / (intensity.std() + 1e-8)
    
    if intensity_err is None:
        intensity_err = np.abs(np.gradient(intensity))
    
    # Bin spectrum
    binned_flux, bin_widths, binned_noise, bin_centers = bin_spectrum(
        wavelength, intensity, n_bins=52
    )
    
    # Normalize all components
    binned_flux = (binned_flux - np.mean(binned_flux)) / (np.std(binned_flux) + 1e-8)
    binned_noise = (binned_noise - np.mean(binned_noise)) / (np.std(binned_noise) + 1e-8)
    bin_centers = (bin_centers - np.mean(bin_centers)) / (np.std(bin_centers) + 1e-8)
    bin_widths = (bin_widths - np.mean(bin_widths)) / (np.std(bin_widths) + 1e-8)
    
    # Concatenate all components into 208 features
    features = np.concatenate([
        binned_flux,      
        bin_widths,       
        binned_noise,     
        bin_centers       
    ]).astype(np.float32) 
    
    # Create DataFrame with lambda_X column names (matches training format)
    features_dict = {f"lambda_{i}": float(features[i]) for i in range(208)}
    return pd.DataFrame([features_dict])


def extract_spectral_features(wavelength, flux):
    return extract_spectral_features_binned(wavelength, flux, intensity_err=None)