"""Data loading, moved from Module 4 Practicals 1 (FCNN) and 3 (CNN)."""
import torch
from torchvision import datasets, transforms


def get_data_loader(data_dir, batch_size=32, train=True):
    """Return a DataLoader for CIFAR-10.

    data_dir: folder where CIFAR-10 is stored (downloaded on first use).
    train: True for the 50,000 training images (shuffled),
           False for the 10,000 test images (not shuffled).
    """
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return loader
