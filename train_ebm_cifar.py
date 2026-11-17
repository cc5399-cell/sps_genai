"""Assignment 4: train the energy-based model on CIFAR-10 and save it for the API.

Run:  uv run python train_ebm_cifar.py [EPOCHS]
"""
import sys

import torch

from helper_lib.data_loader import get_data_loader
from helper_lib.generator import generate_samples
from helper_lib.model import get_model
from helper_lib.trainer import train_ebm
from helper_lib.utils import get_device, save_model

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 5

torch.manual_seed(42)
device = get_device()
print(f"Using device: {device} | model: EBM (CIFAR-10, 32x32) | epochs: {EPOCHS}")

train_loader = get_data_loader("data", batch_size=128)   # CIFAR-10 training images in [0, 1]

model = get_model("EBM")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.0, 0.999))

train_ebm(model, train_loader, optimizer, device=device, epochs=EPOCHS)
save_model(model, "models/ebm_cifar10.pth")
generate_samples(model, device, num_samples=10, langevin_steps=256, save_path="ebm_cifar10_samples.png")
