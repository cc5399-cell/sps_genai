"""Data loading for the Module 4 (CIFAR-10), Module 5 (Fashion-MNIST) and Module 6 (GAN) notebooks."""
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def get_data_loader(data_dir, batch_size=32, train=True, dataset="CIFAR10", image_size=None):
    """Return a DataLoader for CIFAR-10 (default) or Fashion-MNIST.

    image_size: if set, CIFAR-10 images are resized to image_size x image_size.
    """
    name = dataset.lower()
    if name == "cifar10":
        steps = []
        if image_size:
            steps.append(transforms.Resize((image_size, image_size)))
        steps.append(transforms.ToTensor())
        data = datasets.CIFAR10(root=data_dir, train=train, download=True,
                                transform=transforms.Compose(steps))
    elif name == "fashionmnist":
        # Pad 28x28 -> 32x32 with zeros so three stride-2 convolutions divide it evenly.
        transform = transforms.Compose([transforms.Pad(2), transforms.ToTensor()])
        data = datasets.FashionMNIST(root=data_dir, train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Use 'CIFAR10' or 'FashionMNIST'.")

    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=train)


class CustomImageDataset(Dataset):
    """Images from a folder (e.g. manually downloaded CelebA), from the Module 6 GAN notebook."""

    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0  # dummy label (not used in GANs)


def get_gan_data_loader(data_dir, batch_size=64, dataset="CIFAR10"):
    """Return a shuffled DataLoader of images normalized to [-1, 1] (matches the generator's tanh).

    dataset: "CIFAR10" (3x64x64, resized from 32x32), "CelebA" (3x64x64, images in
             data_dir/img_align_celeba) or "MNIST" (1x28x28, Assignment 3).
    """
    name = dataset.lower()
    if name == "mnist":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        data = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    elif name == "cifar10":
        transform = transforms.Compose([transforms.Resize(64), transforms.CenterCrop(64),
                                        transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
        data = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    elif name == "celeba":
        transform = transforms.Compose([transforms.Resize(128), transforms.CenterCrop(64),
                                        transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
        data = CustomImageDataset(os.path.join(data_dir, "img_align_celeba"), transform=transform)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Use 'MNIST', 'CIFAR10' or 'CelebA'.")

    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0)
