import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .model import GasMLP
from .dataset import ABCDataset

# -------------------------
DATA_PATH  = "app/data/training_data.hdf5"
MODEL_PATH = "app/ml/gas_model.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GAS_NAMES = [
    "H2O","CO2","CH4","CO","NH3",
    "H2","He","N2","O2","O3","SO2","H2S"
]

# Loss 
class PhysicsLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer("weights", torch.tensor([
            1.3, 1.3, 1.3, 1.3, 1.3,
            1.0, 1.0, 1.1, 1.1, 1.2, 1.2, 1.2
        ]))

    def forward(self, pred, target):
        loss = torch.nn.functional.smooth_l1_loss(
            pred, target, reduction="none"
        )
        loss = loss * self.weights
        return loss.mean()


# Load dataset
print("Loading dataset...")
dataset = ABCDataset(DATA_PATH)

print(f"Samples: {len(dataset)} | Features: {dataset.X.shape[1]}")

# Normalize X
X_mean = dataset.X.mean(dim=0)
X_std = dataset.X.std(dim=0) + 1e-8

dataset.X = (dataset.X - X_mean) / X_std

# Normalize Y
y_mean = dataset.y.mean(dim=0)
y_std = dataset.y.std(dim=0) + 1e-8

dataset.y = (dataset.y - y_mean) / y_std

# Save scalers
np.save("app/ml/x_mean.npy", X_mean.numpy())
np.save("app/ml/x_std.npy", X_std.numpy())
np.save("app/ml/y_mean.npy", y_mean.numpy())
np.save("app/ml/y_std.npy", y_std.numpy())

print("Normalization stats saved.")

# Split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
val_loader = DataLoader(val_set, batch_size=256, shuffle=False)

# Load Model
model = GasMLP(input_dim=dataset.X.shape[1]).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=7e-5,   
    weight_decay=1e-5
)

loss_fn = PhysicsLoss().to(DEVICE)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    patience=6,
    factor=0.4
)

best_val = float("inf")
patience = 20
counter = 0

# Training loop
for epoch in range(200):

    model.train()
    train_loss = 0

    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)

        pred = model(X)

        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Model Evaluation
    model.eval()
    val_loss = 0

    y_true_all, y_pred_all = [], []

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)

            pred = model(X)
            loss = loss_fn(pred, y)

            val_loss += loss.item()

            y_true_all.append(y.cpu().numpy())
            y_pred_all.append(pred.cpu().numpy())

    val_loss /= len(val_loader)
    scheduler.step(val_loss)

    y_true = np.vstack(y_true_all)
    y_pred = np.vstack(y_pred_all)

    # METRICS
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    cos_sim = np.mean(
        np.sum(y_true * y_pred, axis=1) /
        (np.linalg.norm(y_true, axis=1) * np.linalg.norm(y_pred, axis=1) + 1e-8)
    )

    per_gas_mae = np.mean(np.abs(y_true - y_pred), axis=0)

    # LOGGING
    if epoch % 10 == 0 or epoch < 5:
        print(f"\nEpoch {epoch}")
        print(f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
        print(f"MAE : {mae:.5f}")
        print(f"RMSE: {rmse:.5f}")
        print(f"R2  : {r2:.5f}")
        print(f"Cosine Similarity: {cos_sim:.5f}")

        print("\nPer-gas MAE:")
        for i, gas in enumerate(GAS_NAMES):
            print(f"{gas:5s}: {per_gas_mae[i]:.5f}")

    # SAVE BEST MODEL
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), MODEL_PATH)
        counter = 0
    else:
        counter += 1

    if counter >= patience:
        print(f"\nEarly stopping at epoch {epoch}")
        break

print("\n✓ Training complete. Model saved.")