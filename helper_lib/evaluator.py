"""Evaluation metrics, moved from Module 4 Practicals 1 and 3."""
import torch


def evaluate_model(model, data_loader, criterion, device="cpu"):
    """Return (average loss, accuracy in %) of the model on the given data."""
    model.to(device)
    model.eval()
    total_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    print(f"Test loss: {avg_loss:.4f}")
    print(f"Accuracy of the network on the {total} test images: {accuracy:.2f}%")
    return avg_loss, accuracy
