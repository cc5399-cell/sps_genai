"""Assignment 4: train the diffusion model on CIFAR-10 and save it for the API.

Run:  uv run python train_diffusion_cifar.py [EPOCHS]
"""
import sys

import torch
import torch.nn as nn

from helper_lib.data_loader import get_data_loader
from helper_lib.generator import generate_samples
from helper_lib.model import get_model
from helper_lib.trainer import train_diffusion
from helper_lib.utils import get_device, save_model

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 10

torch.manual_seed(42)
device = get_device()
print(f"Using device: {device} | model: Diffusion (CIFAR-10, 32x32) | epochs: {EPOCHS}")

train_loader = get_data_loader("data", batch_size=64)   # CIFAR-10 training images in [0, 1]

model = get_model("DiffusionCIFAR")
criterion = nn.L1Loss()
optimizer = torch.optim.AdamW(model.network.parameters(), lr=1e-3, weight_decay=1e-4)

train_diffusion(model, train_loader, criterion, optimizer, device=device, epochs=EPOCHS)
save_model(model, "models/diffusion_cifar10.pth")
generate_samples(model, device, num_samples=10, diffusion_steps=100, save_path="diffusion_cifar10_samples.png")
