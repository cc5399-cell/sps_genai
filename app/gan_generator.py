"""Hand-written digit generation with the Assignment 3 GAN (trained on MNIST)."""
import io
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from helper_lib.model import get_model

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "models" / "mnist_gan.pth"


class DigitGenerator:
    def __init__(self, weights_path=DEFAULT_WEIGHTS):
        self.model = get_model("MNISTGAN")
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.model.eval()  # inference mode (BatchNorm uses its stored statistics)

    def generate_png(self, num_images: int = 16, seed: int | None = None, scale: int = 4) -> bytes:
        """Sample noise vectors, run the generator, and return the digits as one PNG grid."""
        rng = torch.Generator().manual_seed(seed) if seed is not None else None
        noise = torch.randn(num_images, self.model.z_dim, generator=rng)
        with torch.no_grad():
            images = (self.model.generator(noise) * 0.5 + 0.5).clamp(0, 1)  # [-1, 1] -> [0, 1]
        grid = to_pil_image(make_grid(images, nrow=min(num_images, 8), padding=2))
        grid = grid.resize((grid.width * scale, grid.height * scale), Image.NEAREST)  # enlarge 28px digits
        buffer = io.BytesIO()
        grid.save(buffer, format="PNG")
        return buffer.getvalue()

    def describe(self) -> dict:
        return {
            "generator": str(self.model.generator),
            "discriminator": str(self.model.discriminator),
            "latent_dim": self.model.z_dim,
            "output": "1x28x28 grayscale digit",
        }
