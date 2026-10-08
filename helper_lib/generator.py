"""Image generation from a trained VAE (Module 5)."""
import math

import matplotlib.pyplot as plt
import numpy as np
import torch


def generate_samples(model, device, num_samples=10, save_path=None):
    """Sample random points from the latent space, decode them, and plot them on a grid.

    Points are drawn from a standard normal distribution, which is what the VAE's
    KL-divergence term trains the latent space to match.
    """
    model.to(device)
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim).to(device)
        samples = model.decoder(z).cpu().numpy()

    ncols = min(num_samples, 5)
    nrows = math.ceil(num_samples / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < num_samples:
            ax.imshow(samples[i].squeeze(), cmap="gray")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"Samples saved to {save_path}")
    plt.show()
