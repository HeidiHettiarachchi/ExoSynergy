import torch
import torch.nn as nn

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.net(x)


# Gas Model 
class GasMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.input = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512)
        )

        self.res1 = ResidualBlock(512)
        self.res2 = ResidualBlock(512)

        self.mid = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.15)
        )

        self.head = nn.ModuleList([
            nn.Linear(256, 1) for _ in range(12)
        ])

    def forward(self, x):
        x = self.input(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.mid(x)

        outputs = [head(x) for head in self.head]   
        x = torch.cat(outputs, dim=1)               

        return x