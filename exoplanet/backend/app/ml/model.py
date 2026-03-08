import torch
import torch.nn as nn
import numpy as np

BULK_IDX  = [5, 6]          
MAJOR_IDX = [7, 8]          
TRACE_IDX = [0, 1, 2, 3, 4, 9, 10, 11]  


class GasMLP(nn.Module):

    def __init__(self, input_dim, output_dim=12):
        super().__init__()

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

        self.bulk_head  = nn.Linear(128, len(BULK_IDX))   
        self.major_head = nn.Linear(128, len(MAJOR_IDX))  
        self.trace_head = nn.Linear(128, len(TRACE_IDX))  

    def forward(self, x):
        features = self.encoder(x)

        bulk  = torch.softmax(self.bulk_head(features),  dim=-1)   
        major = torch.softmax(self.major_head(features), dim=-1)   
        trace = torch.softmax(self.trace_head(features), dim=-1)   

        bulk_budget  = 0.96  
        major_budget = 0.035  
        trace_budget = 0.005 

        bulk  = bulk  * bulk_budget
        major = major * major_budget
        trace = trace * trace_budget

        out = torch.zeros(x.shape[0], 12, device=x.device)
        out[:, BULK_IDX]  = bulk
        out[:, MAJOR_IDX] = major
        out[:, TRACE_IDX] = trace

        out = out / out.sum(dim=-1, keepdim=True) * 100.0

        return out