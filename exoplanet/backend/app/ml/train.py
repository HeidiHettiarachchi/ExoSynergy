import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
from .model import GasMLP
from .dataset import ABCDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_PATH  = "app/data/training_data.hdf5"
MODEL_PATH = "app/ml/gas_model.pt"


class LogSpaceMSELoss(nn.Module):
    def __init__(self, epsilon=1e-2):   # FIX 1: raised epsilon from 1e-4 to 1e-2
        super().__init__()              # small epsilon caused log(~0) → -inf → NaN
        self.epsilon = epsilon

    def forward(self, pred, target):
        # FIX 2: clamp predictions before log to prevent log(negative) 
        pred   = torch.clamp(pred,   min=self.epsilon)
        target = torch.clamp(target, min=self.epsilon)
        log_pred   = torch.log(pred)
        log_target = torch.log(target)
        return torch.mean((log_pred - log_target) ** 2)


# -----------------------------
# Dataset & Preprocessing
# -----------------------------
print(f"Loading dataset from {DATA_PATH}...")
full_set = ABCDataset(DATA_PATH)
print(f"✓ Loaded {len(full_set)} samples with {full_set.X.shape[1]} features")

# FIX 3: check for NaNs in labels BEFORE training
nan_mask = torch.isnan(full_set.y).any(dim=1) | torch.isinf(full_set.y).any(dim=1)
if nan_mask.any():
    print(f"⚠ Removing {nan_mask.sum().item()} samples with NaN/Inf labels")
    full_set.X = full_set.X[~nan_mask]
    full_set.y = full_set.y[~nan_mask]

print("Normalizing features...")
X_mean = full_set.X.mean(dim=0)
X_std  = full_set.X.std(dim=0)
X_std[X_std == 0] = 1
full_set.X = (full_set.X - X_mean) / X_std

# FIX 4: check for NaNs in features after normalization
feat_nan = torch.isnan(full_set.X).any(dim=1)
if feat_nan.any():
    print(f"⚠ Removing {feat_nan.sum().item()} samples with NaN features")
    full_set.X = full_set.X[~feat_nan]
    full_set.y = full_set.y[~feat_nan]

print(f"✓ Features normalized: mean={full_set.X.mean():.6f}, std={full_set.X.std():.6f}")
print(f"✓ Clean samples remaining: {len(full_set.X)}")

# Train / Validation Split
train_size = int(0.8 * len(full_set.X))
val_size   = len(full_set.X) - train_size
train_set, val_set = random_split(full_set, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=256, shuffle=False)
print(f"✓ Split: {train_size} train, {val_size} validation")

# -----------------------------
# Model
# -----------------------------
model = GasMLP(input_dim=full_set.X.shape[1])

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-5
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    patience=10,
    factor=0.5
)

loss_fn = LogSpaceMSELoss(epsilon=1e-2)

# -----------------------------
# Training Loop
# -----------------------------
best_val_loss      = float('inf')
patience           = 25
early_stop_counter = 0

for epoch in range(300):

    # -------- TRAIN --------
    model.train()
    train_loss = 0

    for X, y in train_loader:
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()

        # FIX 5: gradient clipping prevents exploding gradients → NaN weights
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # -------- VALIDATION --------
    model.eval()
    val_loss = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for X, y in val_loader:
            pred = model(X)
            loss = loss_fn(pred, y)

            val_loss += loss.item()
            y_true.append(y.numpy())
            y_pred.append(pred.numpy())

    val_loss /= len(val_loader)
    scheduler.step(val_loss)

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    # FIX 6: guard metrics calculation against any remaining NaNs
    if np.isnan(y_pred).any():
        print(f"Epoch {epoch:3d} | ⚠ NaN in predictions — skipping metrics")
        continue

    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    if (epoch + 1) % 10 == 0 or epoch < 5:
        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"RMSE: {rmse:.6f} | "
            f"MAE: {mae:.6f} | "
            f"R2: {r2:.6f}"
        )

    if val_loss < best_val_loss:
        best_val_loss      = val_loss
        torch.save(model.state_dict(), MODEL_PATH)
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    if early_stop_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
        break

print(f"\n✓ Training complete. Best model saved to: {MODEL_PATH}")