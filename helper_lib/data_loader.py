"""Data loading, moved from the Module 4 (CIFAR-10) and Module 5 (Fashion-MNIST) notebooks."""
import torch
from torchvision import datasets, transforms


def get_data_loader(data_dir, batch_size=32, train=True, dataset="CIFAR10"):
    """Return a DataLoader for CIFAR-10 (default) or Fashion-MNIST.

    data_dir: folder where the dataset is stored (downloaded on first use).
    train: True for the training split (shuffled), False for the test split.
    dataset: "CIFAR10" (3x32x32 color images, Module 4)
             or "FashionMNIST" (1x28x28 grayscale, padded to 1x32x32, Module 5).
    """
    name = dataset.lower()
    if name == "cifar10":
        transform = transforms.Compose([transforms.ToTensor()])
        data = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform)
    elif name == "fashionmnist":
        # Pad 28x28 -> 32x32 with zeros so three stride-2 convolutions divide it evenly.
        transform = transforms.Compose([transforms.Pad(2), transforms.ToTensor()])
        data = datasets.FashionMNIST(root=data_dir, train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Use 'CIFAR10' or 'FashionMNIST'.")

    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=train)
    return loader
