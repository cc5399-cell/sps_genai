"""Image generation from trained generative models: VAE (Module 5) and GAN (Module 6)."""
import math

import matplotlib.pyplot as plt
import numpy as np
import torch


def generate_samples(model, device, num_samples=10, save_path=None):
    """Sample random points in the latent space, generate images, and plot them on a grid.

    GAN: generator output is RGB in [-1, 1] (tanh), rescaled to [0, 1] for display.
    VAE: decoder output is grayscale in [0, 1].
    """
    model.to(device)
    model.eval()
    with torch.no_grad():
        if hasattr(model, "generator"):  # GAN
            z = torch.randn(num_samples, model.z_dim, device=device)
            images = model.generator(z) * 0.5 + 0.5  # undo the [-1, 1] normalization
        else:  # VAE
            z = torch.randn(num_samples, model.latent_dim, device=device)
            images = model.decoder(z)
    images = images.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()  # (N, H, W, C)

    ncols = min(num_samples, 5)
    nrows = math.ceil(num_samples / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows))
    for i, ax in enumerate(np.array(axes).reshape(-1)):
        ax.axis("off")
        if i < num_samples:
            ax.imshow(images[i].squeeze(), cmap="gray")  # cmap only applies to grayscale

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"Samples saved to {save_path}")
    plt.show()
