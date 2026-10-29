"""Usage example for the GAN in helper_lib (Module 6 activity).

Run:  uv run python train_gan_example.py [EPOCHS] [DATASET]
e.g.  uv run python train_gan_example.py 5 CIFAR10
"""
import sys

import torch

from helper_lib.data_loader import get_gan_data_loader
from helper_lib.generator import generate_samples
from helper_lib.model import get_model
from helper_lib.trainer import train_gan
from helper_lib.utils import get_device, save_model

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
DATASET = sys.argv[2] if len(sys.argv) > 2 else "CIFAR10"

torch.manual_seed(42)
device = get_device()
print(f"Using device: {device} | model: GAN | dataset: {DATASET} | epochs: {EPOCHS}")

train_loader = get_gan_data_loader("data", batch_size=64, dataset=DATASET)
gan = get_model("GAN")

train_gan(gan, train_loader, criterion=None, optimizer=None, device=device, epochs=EPOCHS)
save_model(gan, "checkpoints/gan.pth")
generate_samples(gan, device, num_samples=10, save_path="gan_samples.png")
