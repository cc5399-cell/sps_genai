"""Usage example for the diffusion model in helper_lib (Module 8 activity).

Run:  uv run python train_diffusion_example.py [EPOCHS]
"""
import sys

import torch
import torch.nn as nn

from helper_lib.data_loader import get_diffusion_data_loader
from helper_lib.generator import generate_samples
from helper_lib.model import get_model
from helper_lib.trainer import train_diffusion
from helper_lib.utils import get_device, save_model

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 10

torch.manual_seed(42)
device = get_device()
print(f"Using device: {device} | model: Diffusion | epochs: {EPOCHS}")

train_loader = get_diffusion_data_loader("data", batch_size=64)

model = get_model("Diffusion")
criterion = nn.L1Loss()
optimizer = torch.optim.AdamW(model.network.parameters(), lr=1e-3, weight_decay=1e-4)

train_diffusion(model, train_loader, criterion, optimizer, device=device, epochs=EPOCHS)
save_model(model, "checkpoints/diffusion.pth")
generate_samples(model, device, num_samples=10, diffusion_steps=100, save_path="diffusion_samples.png")
