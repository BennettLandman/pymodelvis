"""Model and input loading for the command line (and the examples).

Model specifications
--------------------
=====================================  ===========================================================
``resnet50``, ``vit``, ``cxr``, ``unest``   built-in aliases (see :data:`ALIASES`)
``torchvision:NAME[:WEIGHTS]``         any torchvision classification / segmentation model
                                       (``WEIGHTS`` default ``DEFAULT``; ``none`` = random init)
``timm:NAME``                          any timm model (pretrained)
``xrv:WEIGHTS``                        TorchXRayVision DenseNet (default ``densenet121-res224-all``)
``monai:BUNDLE_DIR``                   a MONAI bundle directory (``configs/inference.json``)
``path/to/file.py:Name``               a class or factory in a Python file (``--model-args`` JSON)
``package.module:Name``                a class or factory in an importable module
``path/to/model.pt``                   a pickled ``nn.Module`` saved with ``torch.save(model)``
=====================================  ===========================================================

``--weights path.pt`` loads a state dict into any of the above.

Downloaded weights are cached in ``$NEURAL_FLOW_HOME`` (default ``~/.cache/neural_flow``);
files already present in ``./real_models`` (``tools/checkout_real_models.py``) are used first.
"""
from __future__ import annotations

import glob
import importlib
import importlib.util
import json
import os
import sys
import urllib.request
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

GITHUB_WEIGHTS = {
    "resnet50_ram-a26f946b.pth":
        "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/resnet50_ram-a26f946b.pth",
    "vit_base_patch16_224_jx.pth":
        "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth",
}
CXR_SAMPLE_URL = "https://raw.githubusercontent.com/mlmed/torchxrayvision/master/tests/00000001_000.png"
UNEST_BUNDLE = "wholeBrainSeg_Large_UNEST_segmentation"

ALIASES = {
    "resnet50": "torchvision:resnet50",
    "resnet18": "torchvision:resnet18",
    "vit": "timm:vit_base_patch16_224",
    "vit_b_16": "torchvision:vit_b_16",
    "swin_t": "torchvision:swin_t",
    "densenet121": "torchvision:densenet121",
    "cxr": "xrv:densenet121-res224-all",
    "unest": f"monai:{UNEST_BUNDLE}",
}


def cache_dir() -> str:
    d = os.environ.get("NEURAL_FLOW_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "neural_flow")
    os.makedirs(d, exist_ok=True)
    return d


def search_dirs() -> List[str]:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dirs = [os.path.join(os.getcwd(), "real_models"), os.path.join(here, "real_models"), cache_dir()]
    out = []
    for d in dirs:
        if d not in out:
            out.append(d)
    return out


def find_file(name: str) -> Optional[str]:
    for d in search_dirs():
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def fetch(name: str, url: Optional[str] = None, quiet: bool = False) -> str:
    """Return a local path for a known weight file, downloading it once if needed."""
    p = find_file(name)
    if p:
        return p
    url = url or GITHUB_WEIGHTS[name]
    p = os.path.join(cache_dir(), name)
    if not quiet:
        print(f"downloading {name} → {p}", file=sys.stderr)
    tmp = p + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, p)
    return p


def imagenet_classes() -> List[str]:
    p = find_file("imagenet_classes.json")
    if p:
        return json.load(open(p))
    try:
        import torchvision

        return list(torchvision.models.ResNet50_Weights.IMAGENET1K_V2.meta["categories"])
    except Exception:
        return [f"class {i}" for i in range(1000)]


@dataclass
class LoadedModel:
    model: nn.Module
    name: str
    preset: str = "imagenet"                  # imagenet | half | xray | volume | raw
    size: int = 224
    mean: Tuple[float, ...] = IMAGENET_MEAN
    std: Tuple[float, ...] = IMAGENET_STD
    class_names: Optional[List[str]] = None
    output_types: Dict[str, str] = field(default_factory=dict)
    roi: Optional[Tuple[int, int, int]] = None
    preprocess: Any = None                    # MONAI transform for volumes
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------


def _load_state(model: nn.Module, path: str) -> None:
    sd = torch.load(path, map_location="cpu")
    for k in ("state_dict", "model", "model_state_dict", "net"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
    if isinstance(sd, nn.Module):
        sd = sd.state_dict()
    sd = {(k[7:] if k.startswith("module.") else k): (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
          for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        warnings.warn(f"loaded {os.path.basename(path)} with {len(missing)} missing / {len(unexpected)} unexpected keys")


def _torchvision(name: str, weights: Optional[str], allow_random: bool) -> LoadedModel:
    import torchvision

    if weights is None or weights.upper() == "DEFAULT":
        w = torchvision.models.get_model_weights(name).DEFAULT
    elif weights.lower() == "none":
        w = None
    else:
        w = torchvision.models.get_model_weights(name)[weights]
    try:
        m = torchvision.models.get_model(name, weights=w)
    except Exception as e:
        # offline fallback for the flagship example
        if name == "resnet50":
            m = torchvision.models.get_model(name, weights=None)
            m.load_state_dict(torch.load(fetch("resnet50_ram-a26f946b.pth"), map_location="cpu"))
        elif allow_random:
            warnings.warn(f"could not download torchvision weights for {name} ({type(e).__name__}); using random init")
            m = torchvision.models.get_model(name, weights=None)
            w = None
        else:
            raise RuntimeError(f"could not download torchvision weights for '{name}': {e}\n"
                               f"(connect to the internet, pass --weights FILE, or --allow-random-weights)") from e
    lm = LoadedModel(m.eval(), name)
    meta = getattr(w, "meta", {}) if w is not None else {}
    if meta.get("categories"):
        lm.class_names = list(meta["categories"])
    elif name == "resnet50":
        lm.class_names = imagenet_classes()
    try:
        tr = w.transforms() if w is not None else None
        if tr is not None and hasattr(tr, "mean"):
            lm.mean, lm.std = tuple(tr.mean), tuple(tr.std)
            crop = getattr(tr, "crop_size", [224])
            lm.size = int(crop[0] if isinstance(crop, (list, tuple)) else crop)
    except Exception:
        pass
    return lm


def _timm(name: str, allow_random: bool) -> LoadedModel:
    import timm

    try:
        m = timm.create_model(name, pretrained=True)
    except Exception as e:
        m = timm.create_model(name, pretrained=False)
        if name == "vit_base_patch16_224":
            m.load_state_dict(torch.load(fetch("vit_base_patch16_224_jx.pth"), map_location="cpu"))
        elif not allow_random:
            raise RuntimeError(f"could not download timm weights for '{name}': {e}\n"
                               f"(connect to the internet, pass --weights FILE, or --allow-random-weights)") from e
        else:
            warnings.warn(f"timm weights for {name} unavailable; using random init")
    cfg = getattr(m, "pretrained_cfg", {}) or {}
    lm = LoadedModel(m.eval(), name, mean=tuple(cfg.get("mean", IMAGENET_MEAN)), std=tuple(cfg.get("std", IMAGENET_STD)),
                     size=int(cfg.get("input_size", (3, 224, 224))[-1]))
    if getattr(m, "num_classes", 0) == 1000:
        lm.class_names = imagenet_classes()
    return lm


def _xrv(weights: str) -> LoadedModel:
    import torchxrayvision as xrv

    m = xrv.models.DenseNet(weights=weights or "densenet121-res224-all").eval()
    return LoadedModel(m, f"xrv {weights}", preset="xray", size=224, class_names=list(m.pathologies),
                       output_types={"output": "multilabel_probs"})


def find_bundle(spec: str) -> str:
    if os.path.isdir(spec):
        return spec
    for d in search_dirs():
        for cand in (os.path.join(d, "monai_bundles", spec), os.path.join(d, spec)):
            if os.path.isdir(cand):
                return cand
    raise FileNotFoundError(f"MONAI bundle '{spec}' not found in {search_dirs()} — run `neural-flow fetch unest` "
                            f"or tools/checkout_real_models.py")


def load_bundle(bdir: str) -> LoadedModel:
    """Instantiate a MONAI bundle's network from its own config, load its weights and preprocessing."""
    from monai.bundle import ConfigParser

    cfgp = None
    for n in ("inference.json", "inference.yaml", "inference.yml"):
        if os.path.exists(os.path.join(bdir, "configs", n)):
            cfgp = os.path.join(bdir, "configs", n)
            break
    if cfgp is None:
        raise FileNotFoundError(f"no configs/inference.* in {bdir}")
    sys.path.insert(0, bdir)
    parser = ConfigParser()
    parser.read_config(cfgp)
    parser["bundle_root"] = bdir
    net = parser.get_parsed_content("network_def", instantiate=True)
    wts = [w for w in sorted(glob.glob(os.path.join(bdir, "models", "*.pt")) + glob.glob(os.path.join(bdir, "models", "*.pth")))]
    if wts:
        _load_state(net, wts[0])
    roi = None
    try:
        inf = parser.get("inferer")
        if isinstance(inf, dict) and inf.get("roi_size"):
            roi = tuple(int(v) for v in inf.get("roi_size"))
    except Exception:
        pass
    pre = None
    try:
        pre = parser.get_parsed_content("preprocessing", instantiate=True)
    except Exception:
        pass
    meta = {}
    mp = os.path.join(bdir, "configs", "metadata.json")
    if os.path.exists(mp):
        meta = json.load(open(mp))
    lm = LoadedModel(net.eval(), meta.get("name") or os.path.basename(bdir.rstrip("/")), preset="volume",
                     output_types={"output": "segmentation"}, roi=roi, preprocess=pre)
    return lm


def _python_object(spec: str, model_args: Optional[dict]) -> LoadedModel:
    path, _, attr = spec.rpartition(":")
    if path.endswith(".py"):
        mod_name = "_nf_user_" + os.path.splitext(os.path.basename(path))[0]
        sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
        s = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(s)
        s.loader.exec_module(mod)
    else:
        mod = importlib.import_module(path)
    obj = getattr(mod, attr)
    m = obj(**(model_args or {})) if callable(obj) and not isinstance(obj, nn.Module) else obj
    if not isinstance(m, nn.Module):
        raise TypeError(f"{spec} did not produce a torch.nn.Module")
    return LoadedModel(m.eval(), attr, preset="raw")


def load_model(spec: str, weights: Optional[str] = None, model_args: Optional[dict] = None,
               allow_random: bool = False, device: str = "cpu") -> LoadedModel:
    spec = ALIASES.get(spec, spec)
    kind, _, rest = spec.partition(":")
    if kind == "torchvision":
        name, _, w = rest.partition(":")
        lm = _torchvision(name, None if weights else (w or None), allow_random) if not weights else \
            _torchvision(name, "none", True)
    elif kind == "timm":
        if weights:
            import timm

            lm = LoadedModel(timm.create_model(rest, pretrained=False).eval(), rest)
        else:
            lm = _timm(rest, allow_random)
    elif kind == "xrv":
        lm = _xrv(rest)
    elif kind == "monai":
        lm = load_bundle(find_bundle(rest))
    elif spec.endswith((".pt", ".pth")) and os.path.exists(spec) and ":" not in os.path.basename(spec):
        obj = torch.load(spec, map_location="cpu", weights_only=False)
        if not isinstance(obj, nn.Module):
            raise TypeError(f"{spec} contains a state dict, not a model — use `file.py:ClassName --weights {spec}`")
        lm = LoadedModel(obj.eval(), os.path.splitext(os.path.basename(spec))[0], preset="raw")
    elif ":" in spec:
        lm = _python_object(spec, model_args)
    else:
        raise ValueError(f"unknown model spec '{spec}'. Try one of {sorted(ALIASES)} or torchvision:NAME, timm:NAME, "
                         f"xrv:WEIGHTS, monai:DIR, file.py:Class, module:Class, model.pt")
    if weights:
        _load_state(lm.model, weights)
    lm.model.to(device)
    return lm


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
VOLUME_EXT = (".nii", ".nii.gz", ".mgz", ".nrrd")


def sample_image_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "chelsea_cat.jpg")


def cxr_sample_path() -> str:
    p = find_file("nih_00000001_000.png")
    if p:
        return p
    return fetch("nih_00000001_000.png", CXR_SAMPLE_URL)


def read_image(path: str) -> np.ndarray:
    """HxWx3 float RGB in [0, 1] (grayscale images are replicated)."""
    from PIL import Image

    if path == "sample:cat":
        path = sample_image_path()
    elif path == "sample:cxr":
        path = cxr_sample_path()
    im = Image.open(path)
    if im.mode in ("I;16", "I", "F"):
        a = np.asarray(im, np.float32)
        a = (a - a.min()) / max(a.max() - a.min(), 1e-6)
        return np.repeat(a[..., None], 3, -1)
    return np.asarray(im.convert("RGB"), np.float32) / 255.0


def image_to_tensor(img: np.ndarray, lm: LoadedModel, size: Optional[int] = None, center_crop: bool = True) -> torch.Tensor:
    from PIL import Image

    size = size or lm.size
    H, W = img.shape[:2]
    if center_crop:
        s = min(H, W)
        img = img[(H - s) // 2:(H - s) // 2 + s, (W - s) // 2:(W - s) // 2 + s]
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    pil = pil.resize((size, size), Image.BICUBIC)
    a = np.asarray(pil, np.float32) / 255.0
    if lm.preset == "xray":                                  # TorchXRayVision convention: [-1024, 1024], 1 channel
        g = a.mean(-1)
        return torch.from_numpy(g * 2048.0 - 1024.0)[None, None]
    x = torch.from_numpy(a).permute(2, 0, 1)
    if lm.preset == "raw":
        return x[None]
    mean = torch.tensor(lm.mean)[:, None, None]
    std = torch.tensor(lm.std)[:, None, None]
    return ((x - mean) / std)[None]


def read_volume(path: str, lm: Optional[LoadedModel] = None) -> torch.Tensor:
    """[C, X, Y, Z] tensor; uses the bundle's MONAI preprocessing when available, else foreground z-score."""
    if lm is not None and lm.preprocess is not None:
        try:
            d = lm.preprocess({"image": path})
            img = d["image"] if isinstance(d, dict) else d[0]["image"]
            return torch.as_tensor(np.asarray(img), dtype=torch.float32)
        except Exception as e:
            warnings.warn(f"bundle preprocessing failed ({str(e)[:120]}); using z-score")
    import nibabel as nib

    v = np.asarray(nib.load(path).get_fdata(), np.float32)
    if v.ndim == 4:
        v = np.moveaxis(v, -1, 0)
    else:
        v = v[None]
    m = v > np.percentile(v, 20)
    v = (v - v[m].mean()) / (v[m].std() + 1e-6)
    return torch.from_numpy(v)


def crop_volume(vol: torch.Tensor, roi: Sequence[int], center=None) -> torch.Tensor:
    """ROI crop [1, C, *roi] around ``center`` (default: centre of mass of the foreground)."""
    C, *S = vol.shape
    v = vol[0].numpy()
    if center is None:
        w = np.clip(v - np.percentile(v, 50), 0, None)
        idx = np.indices(v.shape).reshape(3, -1)
        center = (idx * w.reshape(-1)).sum(1) / max(w.sum(), 1e-9)
    sl = []
    for c, r, n in zip(center, roi, S):
        s = int(np.clip(round(c - r / 2), 0, max(n - r, 0)))
        sl.append(slice(s, s + r))
    out = vol[:, sl[0], sl[1], sl[2]]
    pad = [(0, r - o) for r, o in zip(roi, out.shape[1:])]
    if any(p[1] > 0 for p in pad):
        out = torch.nn.functional.pad(out, [x for p in reversed(pad) for x in p])
    return out[None]


def load_input(spec: str, lm: LoadedModel, size: Optional[int] = None, crop: Optional[int] = None,
               center_crop: bool = True) -> torch.Tensor:
    """Turn an input spec into a batched tensor for ``lm.model``.

    ``spec`` is an image file, a NIfTI volume, a ``.npy`` / ``.pt`` tensor, ``random:1,3,224,224``,
    ``sample:cat`` or ``sample:cxr``.
    """
    low = spec.lower()
    if low.startswith("random:"):
        shape = tuple(int(v) for v in spec.split(":", 1)[1].replace("x", ",").split(","))
        g = torch.Generator().manual_seed(0)
        return torch.randn(shape, generator=g)
    if low.endswith(".npy"):
        t = torch.from_numpy(np.load(spec)).float()
        return t if t.dim() >= 4 or t.dim() <= 2 else t[None]
    if low.endswith((".pt", ".pth")):
        t = torch.load(spec, map_location="cpu")
        t = t if torch.is_tensor(t) else torch.as_tensor(t)
        return t.float() if t.is_floating_point() else t
    if low.endswith(VOLUME_EXT):
        vol = read_volume(spec, lm)
        roi = (crop,) * 3 if crop else lm.roi
        return crop_volume(vol, roi) if roi else vol[None]
    if low.startswith("sample:") or low.endswith(IMAGE_EXT):
        return image_to_tensor(read_image(spec), lm, size, center_crop)
    raise ValueError(f"don't know how to read input '{spec}' (image, NIfTI, .npy, .pt, random:SHAPE, sample:cat)")


def parse_class_names(arg: Optional[str], lm: LoadedModel) -> Optional[List[str]]:
    if not arg:
        return lm.class_names
    if arg == "imagenet":
        return imagenet_classes()
    if os.path.exists(arg):
        txt = open(arg).read()
        if arg.endswith(".json"):
            v = json.loads(txt)
            return list(v.values()) if isinstance(v, dict) else list(v)
        return [l.strip() for l in txt.splitlines() if l.strip()]
    return [s.strip() for s in arg.split(",")]
