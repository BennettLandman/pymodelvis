"""Loaders for real pretrained models used by the examples.

Weights are looked up in ``<repo>/real_models/`` first (``tools/checkout_real_models.py``
puts torchvision weights there); otherwise they are downloaded once from public
GitHub releases (timm's ResNet-50 "RA" weights and the JAX-ported ViT-B/16) and
cached in ``real_models/``.
"""
from __future__ import annotations

import json
import os
import urllib.request

import numpy as np
import torch
import torchvision

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "real_models")
URLS = {
    "resnet50_ram-a26f946b.pth":
        "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/resnet50_ram-a26f946b.pth",
    "vit_base_patch16_224_jx.pth":
        "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth",
}
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def _fetch(name: str) -> str:
    os.makedirs(ROOT, exist_ok=True)
    p = os.path.join(ROOT, name)
    if not os.path.exists(p):
        print(f"downloading {name} …")
        urllib.request.urlretrieve(URLS[name], p)
    return p


def imagenet_classes():
    p = os.path.join(ROOT, "imagenet_classes.json")
    if os.path.exists(p):
        return json.load(open(p))
    return torchvision.models.ResNet50_Weights.IMAGENET1K_V2.meta["categories"]


def load_resnet50():
    """Returns (model, mean, std). Prefers torchvision IMAGENET1K_V2 from the checkout script."""
    m = torchvision.models.resnet50()
    tv = [f for f in os.listdir(ROOT) if f.startswith("resnet50_") and f.endswith("_fp16.pth")] if os.path.isdir(ROOT) else []
    if tv:
        sd = torch.load(os.path.join(ROOT, tv[0]), map_location="cpu")
        m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
    else:
        m.load_state_dict(torch.load(_fetch("resnet50_ram-a26f946b.pth"), map_location="cpu"))
    return m.eval(), IMAGENET_MEAN, IMAGENET_STD


def load_vit():
    """timm ViT-B/16 (ImageNet-1k, JAX port; normalisation mean = std = 0.5). Needs `pip install timm`."""
    import timm

    m = timm.create_model("vit_base_patch16_224", pretrained=False)
    m.load_state_dict(torch.load(_fetch("vit_base_patch16_224_jx.pth"), map_location="cpu"))
    return m.eval(), [0.5] * 3, [0.5] * 3


def load_chexpert_densenet():
    """TorchXRayVision DenseNet-121 trained on 7 public chest X-ray datasets (18 pathologies)."""
    import torchxrayvision as xrv

    return xrv.models.DenseNet(weights="densenet121-res224-all").eval()


def photo(name: str) -> np.ndarray:
    """A public-domain sample photo from scikit-image (chelsea, coffee, rocket, astronaut, …) as float RGB."""
    import skimage.data

    img = getattr(skimage.data, name)()
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    return img[..., :3].astype(np.float32) / 255.0


def photo_panorama(names=("chelsea", "coffee", "rocket", "astronaut"), height: int = 256) -> np.ndarray:
    from PIL import Image

    parts = []
    for n in names:
        im = Image.fromarray((photo(n) * 255).astype(np.uint8))
        w, h = im.size
        parts.append(np.asarray(im.resize((int(round(w * height / h)), height), Image.BICUBIC), np.float32) / 255)
    return np.concatenate(parts, 1)


def to_input(img: np.ndarray, mean, std, size: int = 224) -> torch.Tensor:
    """Centre-crop + resize an HxWx3 float image and normalise → [1, 3, size, size]."""
    from PIL import Image

    H, W = img.shape[:2]
    s = min(H, W)
    crop = img[(H - s) // 2:(H - s) // 2 + s, (W - s) // 2:(W - s) // 2 + s]
    im = Image.fromarray((crop * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(im, np.float32) / 255).permute(2, 0, 1)
    return ((x - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None])[None]
