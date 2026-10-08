"""Assignment 2: train the spec CNN on CIFAR-10 (resized to 64x64) and save it for the API.

Run:  uv run python train_classifier.py [EPOCHS]
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets

from helper_lib.data_loader import get_data_loader
from helper_lib.evaluator import evaluate_model
from helper_lib.model import CIFAR10_CLASSES, get_model
from helper_lib.trainer import train_model
from helper_lib.utils import get_device, save_model

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
WEIGHTS_PATH = "models/cnn_classifier.pth"

torch.manual_seed(42)
device = get_device()
print(f"Using device: {device} | epochs: {EPOCHS}")

train_loader = get_data_loader("data", batch_size=64, train=True, image_size=64)
test_loader = get_data_loader("data", batch_size=64, train=False, image_size=64)

model = get_model("AssignmentCNN")
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, criterion, optimizer, device=device, epochs=EPOCHS)
evaluate_model(model, test_loader, criterion, device=device)

model.to("cpu")
save_model(model, WEIGHTS_PATH)

# Save a few test images to try the API with
samples_dir = Path("samples")
samples_dir.mkdir(exist_ok=True)
raw_test = datasets.CIFAR10(root="data", train=False, download=True)
for i in range(8):
    image, label = raw_test[i]
    image.save(samples_dir / f"{i}_{CIFAR10_CLASSES[label]}.png")
print(f"Sample images saved to {samples_dir}/")
