"""Image generation from trained generative models: VAE (Module 5), GAN (Module 6), Diffusion (Module 8), EBM (Assignment 4)."""
import math

import matplotlib.pyplot as plt
import numpy as np
import torch


def generate_samples(model, device, num_samples=10, diffusion_steps=100, langevin_steps=256, save_path=None):
    """Generate num_samples images from random noise / latent points and plot them on a grid.

    Diffusion: denoise Gaussian noise over diffusion_steps steps; output RGB in [0, 1].
    EBM: run langevin_steps of Langevin dynamics from uniform noise in [-1, 1].
    GAN: generator output is in [-1, 1] (tanh), rescaled to [0, 1] for display.
    VAE: decoder output is grayscale in [0, 1].
    """
    model.to(device)
    model.eval()
    with torch.no_grad():
        if hasattr(model, "reverse_diffusion"):  # Diffusion
            images = model.generate(num_samples, diffusion_steps)
        elif hasattr(model, "sample"):  # Energy-based model
            size = model.image_size
            x = torch.rand((num_samples, model.in_channels, size, size), device=device) * 2 - 1
            images = model.sample(x, steps=langevin_steps, step_size=10.0, noise_std=0.01) * 0.5 + 0.5
        elif hasattr(model, "generator"):  # GAN
            z = torch.randn(num_samples, model.z_dim, device=device)
            images = model.generator(z) * 0.5 + 0.5
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
