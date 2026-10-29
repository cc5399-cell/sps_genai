"""Assignment 3: train the spec GAN on MNIST and save it for the API.

Run:  uv run python train_mnist_gan.py [EPOCHS]
"""
import sys

import torch

from helper_lib.data_loader import get_gan_data_loader
from helper_lib.generator import generate_samples
from helper_lib.model import get_model
from helper_lib.trainer import train_standard_gan
from helper_lib.utils import get_device, save_model

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 15
WEIGHTS_PATH = "models/mnist_gan.pth"

torch.manual_seed(42)
device = get_device()
print(f"Using device: {device} | model: MNISTGAN | epochs: {EPOCHS}")

train_loader = get_gan_data_loader("data", batch_size=128, dataset="MNIST")
gan = get_model("MNISTGAN")
print(gan)

train_standard_gan(gan, train_loader, device=device, epochs=EPOCHS)

gan.to("cpu")
save_model(gan, WEIGHTS_PATH)
generate_samples(gan, "cpu", num_samples=10, save_path="mnist_gan_samples.png")
