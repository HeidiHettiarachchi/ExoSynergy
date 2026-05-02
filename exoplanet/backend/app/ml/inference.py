import torch
import numpy as np
import os
from .model import GasMLP

MODEL_PATH = "app/ml/gas_model.pt"

_model = None
_x_mean = None
_x_std = None
_y_mean = None
_y_std = None

ALL_GASES = ["H2O", "CO2", "CH4", "CO", "NH3", "H2", "He", "N2", "O2", "O3", "SO2", "H2S"]

# Physical constraints for each gas in different observational regimes
GAS_CONSTRAINTS = {
    "direct": {
        "H2O": {"min": 0.0, "max": 50.0},
        "CO2": {"min": 0.0, "max": 30.0},
        "CH4": {"min": 0.0, "max": 20.0},
        "CO": {"min": 0.0, "max": 20.0},
        "NH3": {"min": 0.0, "max": 10.0},
        "H2": {"min": 0.0, "max": 95.0},
        "He": {"min": 0.0, "max": 95.0},
        "N2": {"min": 0.0, "max": 100.0},
        "O2": {"min": 0.0, "max": 50.0},
        "O3": {"min": 0.0, "max": 1.0},
        "SO2": {"min": 0.0, "max": 5.0},
        "H2S": {"min": 0.0, "max": 5.0},
    },
    "eclipse": {
        "H2O": {"min": 0.0, "max": 40.0},
        "CO2": {"min": 0.0, "max": 35.0},
        "CH4": {"min": 0.0, "max": 25.0},
        "CO": {"min": 0.0, "max": 25.0},
        "NH3": {"min": 0.0, "max": 15.0},
        "H2": {"min": 0.0, "max": 95.0},
        "He": {"min": 0.0, "max": 95.0},
        "N2": {"min": 0.0, "max": 100.0},
        "O2": {"min": 0.0, "max": 45.0},
        "O3": {"min": 0.0, "max": 0.5},
        "SO2": {"min": 0.0, "max": 8.0},
        "H2S": {"min": 0.0, "max": 8.0},
    },
    "transmission": {
        "H2O": {"min": 0.0, "max": 35.0},
        "CO2": {"min": 0.0, "max": 30.0},
        "CH4": {"min": 0.0, "max": 20.0},
        "CO": {"min": 0.0, "max": 20.0},
        "NH3": {"min": 0.0, "max": 12.0},
        "H2": {"min": 0.0, "max": 95.0},
        "He": {"min": 0.0, "max": 95.0},
        "N2": {"min": 0.0, "max": 100.0},
        "O2": {"min": 0.0, "max": 40.0},
        "O3": {"min": 0.0, "max": 0.8},
        "SO2": {"min": 0.0, "max": 6.0},
        "H2S": {"min": 0.0, "max": 6.0},
    }
}


def load_model(input_dim):

    # Loading the gas prediction model and normalizing statistics.
    global _model, _x_mean, _x_std, _y_mean, _y_std
    
    # Load model
    _model = GasMLP(input_dim=input_dim)
    _model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    _model.eval()
    
    # Load normalization statistics
    ml_dir = os.path.dirname(os.path.abspath(__file__))
    
    x_mean_path = os.path.join(ml_dir, "x_mean.npy")
    x_std_path = os.path.join(ml_dir, "x_std.npy")
    y_mean_path = os.path.join(ml_dir, "y_mean.npy")
    y_std_path = os.path.join(ml_dir, "y_std.npy")
    
    if os.path.exists(x_mean_path):
        _x_mean = np.load(x_mean_path)
    else:
        _x_mean = np.zeros(input_dim)
        
    if os.path.exists(x_std_path):
        _x_std = np.load(x_std_path)
    else:
        _x_std = np.ones(input_dim)
        
    if os.path.exists(y_mean_path):
        _y_mean = np.load(y_mean_path)
    else:
        _y_mean = np.zeros(12)
        
    if os.path.exists(y_std_path):
        _y_std = np.load(y_std_path)
    else:
        _y_std = np.ones(12)


def pad_features_to_312(features_array):
    # Padding 208 features to 312 dimensions 
    if features_array.shape[1] == 312:
        return features_array
    
    if features_array.shape[1] == 208:
        # Pad 208 features to 312 with zeros
        pad_size = 312 - 208
        features_array = np.pad(features_array, ((0, 0), (0, pad_size)), mode='constant')
    elif features_array.shape[1] < 208:
        # Pad to 312
        pad_size = 312 - features_array.shape[1]
        features_array = np.pad(features_array, ((0, 0), (0, pad_size)), mode='constant')
    else:
        # Truncate to 312
        features_array = features_array[:, :312]
    
    return features_array


def normalize_features(features_array):
    """Normalize input features using training statistics."""
    if _x_mean is None or _x_std is None:
        return features_array
    
    return (features_array - _x_mean) / (_x_std + 1e-8)


def inverse_normalize_output(y_log_space):
    """Inverse normalize output from log space using training statistics."""
    if _y_mean is None or _y_std is None:
        return 10 ** y_log_space
    
    # Inverse normalization: y_original = y_normalized * y_std + y_mean
    y_normalized = y_log_space 
    y_original_log = y_normalized * _y_std + _y_mean
    
    # Convert from log10 space to linear
    y_linear = 10 ** y_original_log
    
    return y_linear


def apply_observational_constraints(predictions, data_type):
    # Apply observational-type-specific constraints to predictions.
    if data_type not in GAS_CONSTRAINTS:
        data_type = "direct"  # Default
    
    constraints = GAS_CONSTRAINTS[data_type]
    constrained = {}
    
    for i, gas in enumerate(ALL_GASES):
        value = float(predictions[i])
        gas_constraint = constraints.get(gas, {"min": 0.0, "max": 100.0})
        
        # Apply min/max constraints
        constrained[gas] = max(
            gas_constraint["min"],
            min(value, gas_constraint["max"])
        )
    
    return constrained


def apply_physics_validation(predictions, data_type):

    # Apply additional physics-based validation.    
    h2_he_total = predictions.get("H2", 0) + predictions.get("He", 0)
    
    # For gas giants/hot atmospheres: H2 + He should typically be > 50%
    if data_type == "eclipse" and h2_he_total < 40:
        # Boost H2 if He is low
        if predictions.get("He", 0) < 10:
            predictions["H2"] = max(predictions["H2"], 40)
    
    # Ensure toxic gases don't exceed realistic bounds
    toxic_gases = ["H2S", "SO2", "NH3"]
    total_toxic = sum(predictions.get(g, 0) for g in toxic_gases)
    if total_toxic > 20:
        for gas in toxic_gases:
            predictions[gas] *= (20 / total_toxic)
    
    # O2 and O3 relationship: O3 should be much less than O2
    if predictions.get("O2", 0) > 0:
        max_o3 = predictions["O2"] * 0.01  # O3 at most 1% of O2
        predictions["O3"] = min(predictions.get("O3", 0), max_o3)
    
    return predictions


def predict_major_gases(features_df, data_type="transmission"):
    
    if data_type not in ["direct", "eclipse", "transmission"]:
        data_type = "transmission" 
    
    # Get features as numpy array
    x = features_df.values
    if isinstance(x, np.ndarray) and len(x.shape) == 2:
        x = x[0]  # Take first row if 2D
    
    # Reshape for batch processing
    x = x.reshape(1, -1)
    
    # Pad 208 features to 312 dimensions
    x = pad_features_to_312(x)
    
    # Normalize features using training statistics
    x_normalized = normalize_features(x)
    
    # Clip to prevent extreme values
    x_normalized = np.clip(x_normalized, -5, 5)
    
    # Convert to tensor and predict
    x_tensor = torch.tensor(x_normalized, dtype=torch.float32)
    
    with torch.no_grad():
        y_log_normalized = _model(x_tensor).numpy()[0]  # Still in normalized log space
    
    # Inverse normalize to get predictions in linear space
    y_linear = inverse_normalize_output(y_log_normalized)
    
    # Clip negative values
    y_linear = np.clip(y_linear, 1e-12, None)
    
    # Create initial prediction dict
    predictions = {ALL_GASES[i]: float(y_linear[i]) for i in range(len(ALL_GASES))}
    
    # Apply observational-type-specific constraints
    predictions_array = np.array([predictions[gas] for gas in ALL_GASES])
    constrained = apply_observational_constraints(predictions_array, data_type)
    predictions.update(constrained)
    
    # Apply physics validation
    predictions = apply_physics_validation(predictions, data_type)
    
    # Normalize to sum to 100%
    total = sum(predictions.values())
    if total > 0:
        predictions = {gas: (value / total) * 100.0 for gas, value in predictions.items()}
    else:
        # Fallback to default atmosphere if all zeros
        if data_type == "direct":
            predictions = {"H2": 85.0, "He": 15.0, "H2O": 0.01}
        elif data_type == "eclipse":
            predictions = {"H2": 80.0, "He": 15.0, "H2O": 5.0}
        else:  # transmission
            predictions = {"H2": 75.0, "He": 15.0, "H2O": 10.0}
        
        total = sum(predictions.values())
        predictions = {gas: (value / total) * 100.0 for gas, value in predictions.items()}
    
    return predictions