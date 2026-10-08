"""Usage example for helper_lib (Section 6 of the activity guide).

Run:  uv run python train_example.py [MODEL_NAME] [EPOCHS]
e.g.  uv run python train_example.py CNN 5
"""
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from helper_lib.data_loader import get_data_loader
from helper_lib.evaluator import evaluate_model
from helper_lib.model import get_model
from helper_lib.trainer import train_model
from helper_lib.utils import get_device, save_model

MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "CNN"   # FCNN, CNN or EnhancedCNN
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 5

torch.manual_seed(42)
device = get_device()
print(f"Using device: {device} | model: {MODEL_NAME} | epochs: {EPOCHS}")

train_loader = get_data_loader("data", batch_size=64)
test_loader = get_data_loader("data", batch_size=64, train=False)

model = get_model(MODEL_NAME)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

trained_model = train_model(model, train_loader, criterion, optimizer, device=device,
                            epochs=EPOCHS, checkpoint_dir=f"checkpoints/{MODEL_NAME.lower()}")
evaluate_model(trained_model, test_loader, criterion, device=device)
save_model(trained_model, f"checkpoints/{MODEL_NAME.lower()}.pth")
