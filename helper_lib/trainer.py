"""Training loop, moved from Module 4 Practicals 1 and 3."""
import os

import torch
from tqdm import tqdm

from helper_lib.utils import save_checkpoint


def train_model(model, data_loader, criterion, optimizer, device="cpu", epochs=10,
                checkpoint_dir=None):
    """Train the model for the given number of epochs and return it.

    If checkpoint_dir is given, a checkpoint is saved every epoch, and the
    best epoch so far (by training accuracy) is also saved in checkpoint_dir/best.
    """
    model.to(device)
    best_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct, running_total = 0, 0

        progress = tqdm(data_loader, ncols=120, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_number, (inputs, labels) in enumerate(progress):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels).sum().item()
            running_total += labels.size(0)
            running_loss += loss.item()

            if batch_number % 100 == 99:
                progress.set_postfix({
                    "avg accuracy": f"{running_correct / running_total:.3f}",
                    "avg loss": f"{running_loss / (batch_number + 1):.4f}",
                })

        epoch_loss = running_loss / len(data_loader)
        epoch_accuracy = 100 * running_correct / running_total
        print(f"Epoch {epoch + 1}: Loss={epoch_loss:.4f}, Accuracy={epoch_accuracy:.2f}%")

        if checkpoint_dir:
            path = save_checkpoint(model, optimizer, epoch + 1, epoch_loss, epoch_accuracy,
                                   checkpoint_dir)
            print(f"Checkpoint saved: {path}")
            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                save_checkpoint(model, optimizer, epoch + 1, epoch_loss, epoch_accuracy,
                                os.path.join(checkpoint_dir, "best"))
                print(f"New best model saved! Accuracy: {epoch_accuracy:.2f}%")

    print("Finished Training")
    return model
