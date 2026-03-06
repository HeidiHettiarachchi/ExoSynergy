import torch
import numpy as np
from .model import GasMLP

MODEL_PATH = "app/ml/gas_model.pt"

_model = None
_expected_input_dim = None

ALL_GASES = [
    "H2O","CO2","CH4","CO","NH3",
    "H2","He","N2","O2","O3","SO2","H2S"
]

def load_model(input_dim):
    global _model, _expected_input_dim
    _expected_input_dim = input_dim
    _model = GasMLP(input_dim=input_dim)
    _model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    _model.eval()

def predict_major_gases(features_df):

    x = torch.tensor(features_df.values, dtype=torch.float32)

    if x.shape[1] != _expected_input_dim:
        if x.shape[1] < _expected_input_dim:
            pad = torch.zeros((x.shape[0], _expected_input_dim - x.shape[1]))
            x = torch.cat([x, pad], dim=1)
        else:
            x = x[:, :_expected_input_dim]

    with torch.no_grad():
        y = _model(x).numpy()[0]

    y = np.clip(y, 0, None)

    return {ALL_GASES[i]: float(y[i]) for i in range(len(ALL_GASES))}