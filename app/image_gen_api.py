"""Assignment 4: image generation endpoints for the diffusion and energy-based models trained on CIFAR-10."""
import io
from pathlib import Path

import torch
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from helper_lib.model import get_model

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
PNG_RESPONSE = {200: {"content": {"image/png": {}}, "description": "A grid of generated images"}}

router = APIRouter(tags=["Image generation"])


def load_model(name, filename):
    """Load trained weights onto the CPU, or return None if they have not been trained yet."""
    path = MODELS_DIR / filename
    if not path.exists():
        return None
    model = get_model(name)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


diffusion_model = load_model("DiffusionCIFAR", "diffusion_cifar10.pth")
energy_model = load_model("EBM", "ebm_cifar10.pth")


def to_png(images, scale=4):
    """Arrange images (N, 3, H, W) in [0, 1] on a grid, enlarge them, and encode as PNG bytes."""
    grid = to_pil_image(make_grid(images.clamp(0, 1), nrow=min(len(images), 8), padding=2))
    grid = grid.resize((grid.width * scale, grid.height * scale), Image.NEAREST)
    buffer = io.BytesIO()
    grid.save(buffer, format="PNG")
    return buffer.getvalue()


@router.get("/diffusion/generate", response_class=Response, responses=PNG_RESPONSE)
def generate_with_diffusion(
    num_images: int = Query(8, ge=1, le=32, description="How many images to generate"),
    diffusion_steps: int = Query(50, ge=1, le=500, description="Reverse diffusion steps"),
    seed: int | None = Query(None, description="Optional seed for reproducible output"),
):
    """Denoise Gaussian noise step by step with the diffusion model trained on CIFAR-10."""
    if diffusion_model is None:
        raise HTTPException(status_code=503, detail="Diffusion weights not found. Run: uv run python train_diffusion_cifar.py")
    with torch.random.fork_rng(devices=[]):
        if seed is not None:
            torch.manual_seed(seed)
        images = diffusion_model.generate(num_images, diffusion_steps)
    return Response(content=to_png(images), media_type="image/png")


@router.get("/ebm/generate", response_class=Response, responses=PNG_RESPONSE)
def generate_with_ebm(
    num_images: int = Query(8, ge=1, le=32, description="How many images to generate"),
    steps: int = Query(256, ge=1, le=1000, description="Langevin dynamics steps"),
    seed: int | None = Query(None, description="Optional seed for reproducible output"),
):
    """Start from uniform noise and run Langevin dynamics on the energy model trained on CIFAR-10."""
    if energy_model is None:
        raise HTTPException(status_code=503, detail="EBM weights not found. Run: uv run python train_ebm_cifar.py")
    with torch.random.fork_rng(devices=[]):
        if seed is not None:
            torch.manual_seed(seed)
        size = energy_model.image_size
        x = torch.rand((num_images, energy_model.in_channels, size, size)) * 2 - 1
        images = energy_model.sample(x, steps=steps, step_size=10.0, noise_std=0.01) * 0.5 + 0.5
    return Response(content=to_png(images), media_type="image/png")
