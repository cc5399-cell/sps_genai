"""Usage example for the VAE in helper_lib (Module 5 activity).

Run:  uv run python train_vae_example.py [EPOCHS]
e.g.  uv run python train_vae_example.py 5
"""
import sys

import torch
import torch.optim as optim

from helper_lib.data_loader import get_data_loader
from helper_lib.generator import generate_samples
from helper_lib.model import get_model
from helper_lib.trainer import train_vae_model, vae_loss_function
from helper_lib.utils import get_device, save_model

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 5

torch.manual_seed(42)
device = get_device()
print(f"Using device: {device} | model: VAE | epochs: {EPOCHS}")

train_loader = get_data_loader("data", batch_size=64, dataset="FashionMNIST")

vae = get_model("VAE")
criterion = vae_loss_function
optimizer = optim.Adam(vae.parameters(), lr=0.001)

train_vae_model(vae, train_loader, criterion, optimizer, device=device, epochs=EPOCHS)
save_model(vae, "checkpoints/vae.pth")
generate_samples(vae, device, num_samples=10, save_path="vae_samples.png")
