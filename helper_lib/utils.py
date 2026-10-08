"""Utilities: device selection, saving/loading models, and checkpoints (from Practical 1)."""
import os

import torch


def get_device():
    """Use the Apple GPU (mps) if available, then NVIDIA (cuda), otherwise the CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_model(model, path="checkpoints/model.pth"):
    """Save only the model weights."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path="checkpoints/model.pth", device="cpu"):
    """Load saved weights into a model with the same architecture."""
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model


def save_checkpoint(model, optimizer, epoch, loss, accuracy, checkpoint_dir="checkpoints"):
    """Save model weights, optimizer state, epoch and metrics so training can be resumed."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "accuracy": accuracy,
    }
    name = model.__class__.__name__.lower()
    checkpoint_path = os.path.join(checkpoint_dir, f"{name}_epoch_{epoch:03d}.pth")
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_checkpoint(model, optimizer, checkpoint_path, device="cpu"):
    """Restore a model and optimizer from a checkpoint. Returns (epoch, loss, accuracy)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    accuracy = checkpoint["accuracy"]

    print(f"Loaded checkpoint from epoch {epoch}")
    print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
    return epoch, loss, accuracy
