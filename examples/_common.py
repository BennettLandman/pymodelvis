"""Shared helpers for the example scripts."""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.dirname(HERE))  # allow running without `pip install -e .`

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_image(path: str | None = None, size: int = 224) -> torch.Tensor:
    """Load an RGB image as a normalised [1, 3, size, size] tensor (centre crop)."""
    from PIL import Image

    path = path or os.path.join(HERE, "assets", "chelsea_cat.jpg")
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((size, size), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return ((x - IMAGENET_MEAN) / IMAGENET_STD)[None]


def load_torchvision(builder, weights_enum):
    """Try pretrained weights; fall back to random init (offline machines)."""
    try:
        return builder(weights=weights_enum), True
    except Exception as e:  # no internet / blocked download
        warnings.warn(f"pretrained weights unavailable ({type(e).__name__}); using random initialisation")
        return builder(weights=None), False


@torch.no_grad()
def calibrate_batchnorm(model: torch.nn.Module, x: torch.Tensor, n: int = 8, seed: int = 0) -> None:
    """For *untrained* demo models: set BatchNorm running stats from jittered copies of the input
    so activations have sensible scale (pretrained models do not need this)."""
    g = torch.Generator().manual_seed(seed)
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    model.train()
    for _ in range(n):
        noise = 0.1 * torch.randn(x.shape, generator=g)
        shift = torch.randint(-8, 9, (2,), generator=g)
        model(torch.roll(x, shifts=(int(shift[0]), int(shift[1])), dims=(-2, -1)) + noise)
    model.eval()


def out_path(name: str) -> str:
    return os.path.join(OUT, name)
