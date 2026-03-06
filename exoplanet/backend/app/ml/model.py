import torch
import torch.nn as nn
import numpy as np

# -------------------------------------------------------------------
# Gas indices — must match attrs['gases'] order in your HDF5:
# H2O, CO2, CH4, CO, NH3, H2, He, N2, O2, O3, SO2, H2S
# -------------------------------------------------------------------
BULK_IDX  = [5, 6]           # H2, He  — dominate the atmosphere
MAJOR_IDX = [7, 8]           # N2, O2  — secondary bulk
TRACE_IDX = [0, 1, 2, 3, 4, 9, 10, 11]  # H2O, CO2, CH4, CO, NH3, O3, SO2, H2S


class GasMLP(nn.Module):
    """
    Deeper, wider MLP with a physically aware output layer.

    Key changes vs original:
    - Much wider first layers to preserve spectral detail
    - BatchNorm instead of Dropout for better gradient flow
    - Custom output: log-softmax per gas group so trace gases
      aren't crushed to zero by a global softmax over 12 gases
    """
    def __init__(self, input_dim, output_dim=12):
        super().__init__()

        # Shared feature extractor — wider to handle large spectral input
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )

        # Separate heads per gas group — prevents bulk gases from
        # dominating the gradient and zeroing out trace gas heads
        self.bulk_head  = nn.Linear(128, len(BULK_IDX))   # H2, He
        self.major_head = nn.Linear(128, len(MAJOR_IDX))  # N2, O2
        self.trace_head = nn.Linear(128, len(TRACE_IDX))  # all trace gases

    def forward(self, x):
        features = self.encoder(x)

        # Each head gets its own softmax so trace gases
        # compete only within their own group, not against H2 at 80%
        bulk  = torch.softmax(self.bulk_head(features),  dim=-1)   # sums to 1
        major = torch.softmax(self.major_head(features), dim=-1)   # sums to 1
        trace = torch.softmax(self.trace_head(features), dim=-1)   # sums to 1

        # Fixed budget allocation per group based on physical priors
        # These are soft targets — the model learns within each group
        # but the inter-group split is constrained by physics
        bulk_budget  = 0.96   # H2 + He  ~96% for gas giants
        major_budget = 0.035  # N2 + O2  ~3.5%
        trace_budget = 0.005  # all traces ~0.5%

        bulk  = bulk  * bulk_budget
        major = major * major_budget
        trace = trace * trace_budget

        # Reconstruct full 12-gas vector in correct order:
        # H2O, CO2, CH4, CO, NH3, H2, He, N2, O2, O3, SO2, H2S
        out = torch.zeros(x.shape[0], 12, device=x.device)
        out[:, BULK_IDX]  = bulk
        out[:, MAJOR_IDX] = major
        out[:, TRACE_IDX] = trace

        # Normalize to exactly 100%
        out = out / out.sum(dim=-1, keepdim=True) * 100.0

        return out