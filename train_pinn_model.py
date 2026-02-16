"""
train_pinn_model.py
-------------------
Trains a Physics-Informed Neural Network (PINN) surrogate for the series-RLC
bandpass filter.

Network input:  (R, C, L, freq_hz)   — all log-scaled and normalised
Network output: (vout_mag, vout_phase_deg)

Physics loss — Kirchhoff's Voltage Law (KVL) at the output node
-----------------------------------------------------------------
For the series-RLC circuit driven by Vin=1 (AC source), KVL gives:

    Vin = V_R + V_L + V_C        (phasor domain)

where V_C = Vout (the quantity the network predicts).  In magnitude/phase
form this is hard to enforce directly, so we work in the complex domain:

    H(jω) = Vout / Vin = 1 / (1 − ω²LC + jωRC)

The KVL residual is |1 − H_pred · (1 − ω²LC + jωRC)|², which should be
zero for any physically valid prediction.  The physics loss penalises
non-zero residuals, steering the network toward KVL-consistent outputs
even in regions with sparse training data.

Usage:
    python train_pinn_model.py                      # defaults
    python train_pinn_model.py --epochs 500 --lr 3e-4 --physics-weight 0.1
"""

import argparse
import os
import math

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(ROOT, "data", "rlc_bandpass_dataset.csv")
MODEL_DIR = os.path.join(ROOT, "models")
FIG_DIR = os.path.join(ROOT, "figures")

# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class PINNSurrogate(nn.Module):
    """
    Fully-connected surrogate network.

    Inputs  (4): log10(R), log10(C), log10(L), log10(freq)
    Outputs (2): vout_mag, vout_phase_deg
    """

    def __init__(self, hidden: int = 128, num_layers: int = 5):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(4, hidden), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Physics loss — KVL residual
# ---------------------------------------------------------------------------

def kvl_residual_loss(
    predictions: torch.Tensor,
    R: torch.Tensor,
    C: torch.Tensor,
    L: torch.Tensor,
    freq: torch.Tensor,
) -> torch.Tensor:
    """
    Compute KVL residual in the complex phasor domain.

    The analytical transfer function is:
        H(jω) = 1 / (1 − ω²LC + jωRC)

    We reconstruct H_pred from the network's magnitude & phase output,
    then penalise: |H_pred · D − 1|²  where D = 1 − ω²LC + jωRC.
    """
    mag = predictions[:, 0]       # predicted |Vout|
    phase_deg = predictions[:, 1] # predicted ∠Vout (degrees)
    phase_rad = phase_deg * (math.pi / 180.0)

    # Predicted Vout as a complex phasor
    h_real = mag * torch.cos(phase_rad)
    h_imag = mag * torch.sin(phase_rad)

    omega = 2.0 * math.pi * freq

    # Denominator D = 1 − ω²LC + jωRC
    d_real = 1.0 - (omega ** 2) * L * C
    d_imag = omega * R * C

    # H_pred * D  (complex multiplication)
    prod_real = h_real * d_real - h_imag * d_imag
    prod_imag = h_real * d_imag + h_imag * d_real

    # KVL says H * D = 1 + 0j  (since Vin = 1∠0°)
    residual = (prod_real - 1.0) ** 2 + prod_imag ** 2
    return residual.mean()


# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------

def load_dataset(csv_path: str, subsample: int = 0):
    """
    Load the CSV and return log-scaled input tensors and target tensors,
    plus the raw physical-unit tensors needed for the physics loss.
    """
    df = pd.read_csv(csv_path)

    if subsample > 0 and len(df) > subsample:
        df = df.sample(n=subsample, random_state=42).reset_index(drop=True)

    R = torch.tensor(df["R"].values, dtype=torch.float32)
    C = torch.tensor(df["C"].values, dtype=torch.float32)
    L = torch.tensor(df["L"].values, dtype=torch.float32)
    freq = torch.tensor(df["freq_hz"].values, dtype=torch.float32)

    # Targets
    mag = torch.tensor(df["vout_mag"].values, dtype=torch.float32)
    phase = torch.tensor(df["vout_phase_deg"].values, dtype=torch.float32)

    # Log-scaled network inputs
    x = torch.stack(
        [torch.log10(R), torch.log10(C), torch.log10(L), torch.log10(freq)],
        dim=1,
    )
    y = torch.stack([mag, phase], dim=1)

    # Physical-unit columns for the physics loss
    phys = torch.stack([R, C, L, freq], dim=1)

    return x, y, phys


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    epochs: int = 300,
    batch_size: int = 4096,
    lr: float = 1e-3,
    physics_weight: float = 0.05,
    subsample: int = 0,
    hidden: int = 128,
    num_layers: int = 5,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    x, y, phys = load_dataset(DATA_CSV, subsample=subsample)
    dataset = TensorDataset(x, y, phys)

    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    print(f"Training samples: {n_train:,}  |  Validation samples: {n_val:,}")

    # Model, optimiser, scheduler
    model = PINNSurrogate(hidden=hidden, num_layers=num_layers).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    mse = nn.MSELoss()
    history = {"train_data": [], "train_phys": [], "train_total": [], "val": []}

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        epoch_data_loss = 0.0
        epoch_phys_loss = 0.0
        n_batches = 0

        for xb, yb, pb in train_loader:
            xb, yb, pb = xb.to(device), yb.to(device), pb.to(device)
            pred = model(xb)

            data_loss = mse(pred, yb)
            phys_loss = kvl_residual_loss(
                pred,
                R=pb[:, 0],
                C=pb[:, 1],
                L=pb[:, 2],
                freq=pb[:, 3],
            )
            loss = data_loss + physics_weight * phys_loss

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_data_loss += data_loss.item()
            epoch_phys_loss += phys_loss.item()
            n_batches += 1

        scheduler.step()
        avg_data = epoch_data_loss / n_batches
        avg_phys = epoch_phys_loss / n_batches
        history["train_data"].append(avg_data)
        history["train_phys"].append(avg_phys)
        history["train_total"].append(avg_data + physics_weight * avg_phys)

        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for xb, yb, pb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss_sum += mse(pred, yb).item()
                val_batches += 1
        avg_val = val_loss_sum / max(val_batches, 1)
        history["val"].append(avg_val)

        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:>4d}/{epochs}  "
                f"data={avg_data:.6f}  phys={avg_phys:.6f}  "
                f"total={avg_data + physics_weight * avg_phys:.6f}  "
                f"val={avg_val:.6f}"
            )

    # --- Save model ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "pinn_rlc_surrogate.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden": hidden,
            "num_layers": num_layers,
            "physics_weight": physics_weight,
        },
        model_path,
    )
    print(f"\nModel saved to {model_path}")

    # --- Plot training curves ---
    os.makedirs(FIG_DIR, exist_ok=True)
    _plot_loss_curves(history, epochs, physics_weight)
    _plot_predictions(model, val_ds, device)

    print("Done.")


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _plot_loss_curves(history: dict, epochs: int, pw: float) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ep = range(1, epochs + 1)
    ax.semilogy(ep, history["train_data"], label="Data loss (train)")
    ax.semilogy(ep, history["train_phys"], label="Physics loss (train)")
    ax.semilogy(ep, history["train_total"], label=f"Total (pw={pw})")
    ax.semilogy(ep, history["val"], label="Data loss (val)", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("PINN Training Curves")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "training_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to {path}")


def _plot_predictions(model, val_ds, device) -> None:
    """Scatter plot: predicted vs true for magnitude and phase."""
    loader = DataLoader(val_ds, batch_size=len(val_ds))
    xb, yb, _ = next(iter(loader))
    with torch.no_grad():
        pred = model(xb.to(device)).cpu()
    true_mag, true_phase = yb[:, 0].numpy(), yb[:, 1].numpy()
    pred_mag, pred_phase = pred[:, 0].numpy(), pred[:, 1].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(true_mag, pred_mag, s=1, alpha=0.3)
    axes[0].plot([0, true_mag.max()], [0, true_mag.max()], "r--", lw=1)
    axes[0].set_xlabel("True |Vout|")
    axes[0].set_ylabel("Predicted |Vout|")
    axes[0].set_title("Magnitude: Predicted vs True")

    axes[1].scatter(true_phase, pred_phase, s=1, alpha=0.3)
    lims = [true_phase.min(), true_phase.max()]
    axes[1].plot(lims, lims, "r--", lw=1)
    axes[1].set_xlabel("True Phase (°)")
    axes[1].set_ylabel("Predicted Phase (°)")
    axes[1].set_title("Phase: Predicted vs True")

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "pred_vs_true.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Prediction scatter plot saved to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a PINN surrogate for the RLC bandpass filter."
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--physics-weight",
        type=float,
        default=0.05,
        help="Weight λ for the KVL physics loss term.",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=0,
        help="Subsample the dataset to this many rows (0 = use all).",
    )
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=5)

    args = parser.parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        physics_weight=args.physics_weight,
        subsample=args.subsample,
        hidden=args.hidden,
        num_layers=args.num_layers,
    )
