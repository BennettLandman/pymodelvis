"""Load trained nnU-Net v2 models (results folders) without installing nnU-Net.

A trained nnU-Net model is a *results folder*::

    Dataset123_Name/
      nnUNetTrainer__nnUNetPlans__3d_fullres/     <- the folder to point at
        plans.json   dataset.json   dataset_fingerprint.json
        fold_0/checkpoint_final.pth
        fold_1/…

:func:`load_nnunet` rebuilds the network from ``plans.json`` (old and current plans
formats) with ``dynamic_network_architectures`` — the package nnU-Net itself uses —
loads one fold's checkpoint, switches deep supervision off and attaches nnU-Net's
preprocessing (:class:`NNUNetPreprocessor`) and inference settings:

* reading: NIfTI reoriented to RAS, axes reversed to nnU-Net's ``[z, y, x]`` order
  (the tensor is drawn with ``volume_axes="zyx"``);
* crop to the non-zero bounding box, per-channel normalisation from the plans
  (``CTNormalization``, ``ZScoreNormalization``, ``NoNormalization``,
  ``RescaleTo01Normalization``), resampling to the target spacing;
* sliding window: the plans' patch size, 50 % overlap, Gaussian weighting
  (σ = patch/8), as ``nnUNetPredictor`` does.

Differences from ``nnUNetv2_predict`` (a figure does not need them): no mirroring
test-time augmentation, no fold ensembling, cubic-spline resampling via SciPy
(trilinear when SciPy is missing) without nnU-Net's separate-z handling for very
anisotropic scans, and the prediction stays in the network's resampled space.

TotalSegmentator (Wasserthal et al., Radiology: AI 2023) ships plain nnU-Net v2
results folders; ``totalseg`` / ``totalseg-6mm`` download its openly licensed
(Apache-2.0) fast models on first use, and ``totalseg-organs`` the 1.5 mm organ model
(part 1 of the five full-resolution models), which needs real sliding windows.
"""
from __future__ import annotations

import glob
import json
import os
import pydoc
import sys
import urllib.request
import warnings
import zipfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

TOTALSEG_URL = "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/{name}.zip"
TOTALSEG = {
    # alias: (dataset folder, description)
    "totalseg": ("Dataset297_TotalSegmentator_total_3mm_1559subj", "TotalSegmentator total (fast, 3 mm, 117 structures)"),
    "totalseg-3mm": ("Dataset297_TotalSegmentator_total_3mm_1559subj", "TotalSegmentator total (fast, 3 mm, 117 structures)"),
    "totalseg-6mm": ("Dataset298_TotalSegmentator_total_6mm_1559subj", "TotalSegmentator total (fastest, 6 mm, 117 structures)"),
    "totalseg-organs": ("Dataset291_TotalSegmentator_part1_organs_1559subj",
                        "TotalSegmentator organs (1.5 mm, 24 structures; part 1 of the full-resolution model)"),
}
CT_SAMPLE_URL = "https://raw.githubusercontent.com/wasserth/TotalSegmentator/master/tests/reference_files/example_ct.nii.gz"
CONFIG_PREFERENCE = ("3d_fullres", "3d_cascade_fullres", "3d_lowres")


# ---------------------------------------------------------------------------
# locating a results folder
# ---------------------------------------------------------------------------


def _is_results(d: str) -> bool:
    return os.path.exists(os.path.join(d, "plans.json")) and os.path.exists(os.path.join(d, "dataset.json"))


def find_results(spec: str, download: bool = True, quiet: bool = False) -> str:
    """Resolve ``spec`` to a trainer folder (the one holding ``plans.json`` and ``fold_*``).

    ``spec`` is that folder, a ``DatasetXXX_Name`` folder containing one or more of them
    (``3d_fullres`` preferred), or an alias from :data:`TOTALSEG`.
    """
    from .zoo import cache_dir, search_dirs

    if spec in TOTALSEG:
        name = TOTALSEG[spec][0]
        for d in search_dirs():
            for cand in (os.path.join(d, "nnunet", name), os.path.join(d, name), os.path.join(d, "totalseg", name)):
                if os.path.isdir(cand):
                    return find_results(cand)
        if not download:
            raise FileNotFoundError(f"{spec}: {name} not found in {search_dirs()} — run `neural-flow fetch {spec}`")
        return find_results(fetch_totalseg(spec, quiet=quiet))
    d = os.path.expanduser(spec)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"nnU-Net results folder '{spec}' not found")
    if _is_results(d):
        return d
    subs = sorted(p for p in glob.glob(os.path.join(d, "*")) if os.path.isdir(p) and _is_results(p))
    if not subs:
        raise FileNotFoundError(f"{spec} has no plans.json/dataset.json (point at the "
                                f"Trainer__Plans__configuration folder of a trained nnU-Net model)")
    for c in CONFIG_PREFERENCE:
        for s in subs:
            if s.rstrip("/").endswith("__" + c):
                return s
    return subs[0]


def fetch_totalseg(alias: str = "totalseg", dest: Optional[str] = None, quiet: bool = False) -> str:
    """Download a TotalSegmentator fast model (≈135 MB zip from the project's GitHub releases)."""
    from .zoo import cache_dir

    name = TOTALSEG[alias][0]
    root = dest or os.path.join(cache_dir(), "nnunet")
    out = os.path.join(root, name)
    if os.path.isdir(out):
        return out
    os.makedirs(root, exist_ok=True)
    url = TOTALSEG_URL.format(name=name)
    zp = os.path.join(root, name + ".zip")
    if not quiet:
        print(f"downloading {TOTALSEG[alias][1]}\n  {url}\n  → {out}", file=sys.stderr)
    urllib.request.urlretrieve(url, zp + ".part")
    os.replace(zp + ".part", zp)
    with zipfile.ZipFile(zp) as z:
        z.extractall(root, members=[m for m in z.namelist() if "__MACOSX" not in m and "/._" not in m
                                    and not m.endswith(".DS_Store")])
    os.remove(zp)
    return out


def ct_sample_path(quiet: bool = False) -> str:
    """TotalSegmentator's small example CT (3 mm, thorax–abdomen), downloaded once."""
    from .zoo import fetch, find_file

    for n in ("example_ct.nii.gz", os.path.join("totalseg", "example_ct.nii.gz")):
        p = find_file(n)
        if p:
            return p
    return fetch("example_ct.nii.gz", CT_SAMPLE_URL, quiet=quiet)


# ---------------------------------------------------------------------------
# plans → network
# ---------------------------------------------------------------------------


def architecture_from_plans(conf: Dict[str, Any]) -> Tuple[str, Dict[str, Any], List[str]]:
    """``(class path, kwargs, kwargs needing import)`` for a plans configuration (old or new format)."""
    if "architecture" in conf:
        a = conf["architecture"]
        return a["network_class_name"], dict(a["arch_kwargs"]), list(a.get("_kw_requires_import", []))
    # plans written by nnU-Net ≤ 2.1 (e.g. TotalSegmentator v2): reconstruct, as nnU-Net's ConfigurationManager does
    cls = conf["UNet_class_name"]
    dim = len(conf["patch_size"])
    if cls == "PlainConvUNet":
        path, blocks = "dynamic_network_architectures.architectures.unet.PlainConvUNet", "n_conv_per_stage"
    elif cls == "ResidualEncoderUNet":
        path, blocks = "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet", "n_blocks_per_stage"
    else:
        raise RuntimeError(f"unknown nnU-Net architecture {cls}")
    n = len(conf["n_conv_per_stage_encoder"])
    kw = {
        "n_stages": n,
        "features_per_stage": [min(conf["UNet_base_num_features"] * 2 ** i, conf["unet_max_num_features"]) for i in range(n)],
        "conv_op": f"torch.nn.modules.conv.Conv{dim}d",
        "kernel_sizes": conf["conv_kernel_sizes"],
        "strides": conf["pool_op_kernel_sizes"],
        blocks: conf["n_conv_per_stage_encoder"],
        "n_conv_per_stage_decoder": conf["n_conv_per_stage_decoder"],
        "conv_bias": True,
        "norm_op": f"torch.nn.modules.instancenorm.InstanceNorm{dim}d",
        "norm_op_kwargs": {"eps": 1e-5, "affine": True},
        "dropout_op": None, "dropout_op_kwargs": None,
        "nonlin": "torch.nn.LeakyReLU", "nonlin_kwargs": {"inplace": True},
    }
    return path, kw, ["conv_op", "norm_op", "dropout_op", "nonlin"]


def _locate(path: str):
    obj = pydoc.locate(path)
    if obj is None and path.startswith("dynamic_network_architectures."):
        import importlib

        import dynamic_network_architectures.architectures as A

        name = path.rsplit(".", 1)[-1]
        for sub in ("unet", "residual_unet", "primus"):
            try:
                mod = importlib.import_module(f"{A.__name__}.{sub}")
            except Exception:
                continue
            if hasattr(mod, name):
                return getattr(mod, name)
    if obj is None:
        raise ImportError(f"cannot import {path}")
    return obj


def label_info(dataset: Dict[str, Any]) -> Tuple[List[str], bool]:
    """Output channel names and whether the model is region-based (sigmoid, one channel per region)."""
    labels = dataset["labels"]
    regions = any(isinstance(v, (list, tuple)) for v in labels.values())
    if regions:
        return [k for k, v in labels.items() if k != "background"], True
    inv = {int(v): k for k, v in labels.items()}
    return [inv.get(i, f"label {i}") for i in range(max(inv) + 1)], False


def build_network(plans: Dict[str, Any], configuration: str, dataset: Dict[str, Any],
                  deep_supervision: bool = False) -> nn.Module:
    try:
        import dynamic_network_architectures  # noqa: F401
    except ImportError as e:
        raise ImportError("nnU-Net models need `pip install dynamic-network-architectures` "
                          "(or `pip install pymodelvis[medical]`)") from e
    conf = _conf(plans, configuration)
    path, kw, imp = architecture_from_plans(conf)
    for k in imp:
        if kw.get(k) is not None:
            kw[k] = _locate(kw[k]) if isinstance(kw[k], str) else kw[k]
    names, _ = label_info(dataset)
    n_in = len(dataset.get("channel_names", dataset.get("modality", {"0": "image"})))
    cls = _locate(path)
    kw["deep_supervision"] = deep_supervision
    return cls(input_channels=n_in, num_classes=len(names), **kw)


def _conf(plans: Dict[str, Any], configuration: str) -> Dict[str, Any]:
    conf = dict(plans["configurations"][configuration])
    parent = conf.get("inherits_from")
    while parent:
        base = dict(plans["configurations"][parent])
        parent = base.get("inherits_from")
        base.update(conf)
        conf = base
    return conf


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------


class NNUNetPreprocessor:
    """nnU-Net's default preprocessing for one image (see module docstring).

    Called like a MONAI dict transform: ``pre({"image": path})`` returns
    ``{"image": array [C, z, y, x], "spacing": target spacing (z, y, x), …}``.
    """

    def __init__(self, plans: Dict[str, Any], configuration: str):
        self.conf = _conf(plans, configuration)
        self.props = plans.get("foreground_intensity_properties_per_channel", {})
        self.transpose = list(plans.get("transpose_forward", [0, 1, 2]))
        self.spacing = tuple(float(v) for v in self.conf["spacing"])
        self.schemes = list(self.conf.get("normalization_schemes", ["ZScoreNormalization"]))
        self.use_mask = list(self.conf.get("use_mask_for_norm", [False] * len(self.schemes)))

    # nnU-Net's NibabelIOWithReorient: RAS, then (x, y, z) → (z, y, x)
    @staticmethod
    def read(path: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        import nibabel as nib
        from nibabel.orientations import io_orientation

        img = nib.load(path)
        img = img.as_reoriented(io_orientation(img.affine))
        a = np.asarray(img.dataobj, dtype=np.float32)
        a = a[None] if a.ndim == 3 else np.moveaxis(a, -1, 0)
        sp = tuple(float(v) for v in img.header.get_zooms()[:3])
        return np.ascontiguousarray(a.transpose(0, 3, 2, 1)), sp[::-1]

    def normalize(self, a: np.ndarray) -> np.ndarray:
        out = np.empty_like(a, dtype=np.float32)
        for c in range(a.shape[0]):
            scheme = self.schemes[min(c, len(self.schemes) - 1)]
            x = a[c].astype(np.float32)
            p = self.props.get(str(c), {})
            if scheme == "CTNormalization" and p:
                x = np.clip(x, p["percentile_00_5"], p["percentile_99_5"])
                x = (x - p["mean"]) / max(p["std"], 1e-8)
            elif scheme in ("NoNormalization",):
                pass
            elif scheme == "RescaleTo01Normalization":
                x = (x - x.min()) / max(x.max() - x.min(), 1e-8)
            elif scheme == "RGBTo01Normalization":
                x = x / 255.0
            else:                                            # ZScoreNormalization (default)
                m = x != 0 if self.use_mask[min(c, len(self.use_mask) - 1)] else np.ones_like(x, bool)
                x = (x - x[m].mean()) / max(x[m].std(), 1e-8)
                if self.use_mask[min(c, len(self.use_mask) - 1)]:
                    x[~m] = 0
            out[c] = x
        return out

    def resample(self, a: np.ndarray, spacing: Sequence[float]) -> np.ndarray:
        new = [int(round(n * s / t)) for n, s, t in zip(a.shape[1:], spacing, self.spacing)]
        if tuple(new) == tuple(a.shape[1:]):
            return a
        try:
            from scipy.ndimage import zoom

            f = [n / o for n, o in zip(new, a.shape[1:])]
            return np.stack([zoom(c, f, order=3, mode="nearest", grid_mode=False) for c in a]).astype(np.float32)
        except ImportError:
            t = torch.from_numpy(a)[None]
            return torch.nn.functional.interpolate(t, size=new, mode="trilinear", align_corners=False)[0].numpy()

    def __call__(self, d):
        path = d["image"] if isinstance(d, dict) else d
        a, sp = self.read(path)
        tp = self.transpose
        a = a.transpose([0] + [i + 1 for i in tp])
        sp = tuple(sp[i] for i in tp)
        nz = np.any(a != 0, axis=0)                          # crop to the non-zero bounding box
        bbox = None
        if nz.any() and not nz.all():
            idx = np.where(nz)
            bbox = [(int(i.min()), int(i.max()) + 1) for i in idx]
            a = a[(slice(None),) + tuple(slice(lo, hi) for lo, hi in bbox)]
        a = self.normalize(a)
        a = self.resample(a, sp)
        return {"image": a, "spacing": self.spacing, "original_spacing": sp, "bbox": bbox}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def _load_checkpoint(path: str) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # nnU-Net checkpoints pickle their init arguments (plain Python objects)
        return torch.load(path, map_location="cpu", weights_only=False)


def parse_spec(rest: str) -> Tuple[str, str]:
    """``DIR[:FOLD]`` → (DIR, FOLD); FOLD is a number or ``all`` (default 0)."""
    head, sep, tail = rest.rpartition(":")
    if sep and (tail.isdigit() or tail == "all") and head:
        return head, tail
    return rest, "0"


def load_nnunet(spec: str, fold: Optional[str] = None, checkpoint: str = "checkpoint_final.pth",
                device: str = "cpu", quiet: bool = False):
    """``LoadedModel`` for an nnU-Net results folder (``spec`` = folder[:fold] or a TotalSegmentator alias)."""
    from .zoo import LoadedModel

    d, f = parse_spec(spec)
    fold = fold or f
    rd = find_results(d, quiet=quiet)
    plans = json.load(open(os.path.join(rd, "plans.json")))
    dataset = json.load(open(os.path.join(rd, "dataset.json")))
    configuration = os.path.basename(rd.rstrip("/")).split("__")[-1]
    if configuration not in plans["configurations"]:
        configuration = next(c for c in CONFIG_PREFERENCE + tuple(plans["configurations"]) if c in plans["configurations"])
    conf = _conf(plans, configuration)
    if len(conf["patch_size"]) != 3:
        raise ValueError(f"{rd}: configuration {configuration} is 2-D; neural_flow's nnU-Net loader supports "
                         f"the 3-D configurations (3d_fullres, 3d_lowres)")
    net = build_network(plans, configuration, dataset, deep_supervision=False)
    notes: List[str] = []
    ck = None
    for name in (checkpoint, "checkpoint_final.pth", "checkpoint_best.pth", "checkpoint_latest.pth"):
        p = os.path.join(rd, f"fold_{fold}", name)
        if os.path.exists(p):
            ck = p
            break
    if ck is None:
        avail = sorted(os.path.basename(p) for p in glob.glob(os.path.join(rd, "fold_*")))
        raise FileNotFoundError(f"{rd}: no checkpoint in fold_{fold} (available: {', '.join(avail) or 'none'})")
    obj = _load_checkpoint(ck)
    sd = obj.get("network_weights", obj) if isinstance(obj, dict) else obj
    sd = {k.replace("_orig_mod.", "").removeprefix("module."): v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        warnings.warn(f"nnU-Net checkpoint {ck}: {len(missing)} missing / {len(unexpected)} unexpected keys")
    names, regions = label_info(dataset)
    ds_name = plans.get("dataset_name") or os.path.basename(os.path.dirname(rd.rstrip("/")))
    trainer = obj.get("trainer_name", "") if isinstance(obj, dict) else ""
    lm = LoadedModel(net.eval().to(device), ds_name, preset="volume", class_names=names,
                     output_types={"output": "segmentation"},
                     roi=tuple(int(v) for v in conf["patch_size"]),
                     preprocess=NNUNetPreprocessor(plans, configuration), notes=notes)
    lm.volume_axes = "zyx"
    lm.spacing = tuple(float(v) for v in conf["spacing"])
    lm.sw_overlap = 0.5
    lm.info = {"results": rd, "configuration": configuration, "fold": fold, "checkpoint": os.path.basename(ck),
               "trainer": trainer, "regions": regions, "n_classes": len(names),
               "patch_size": tuple(conf["patch_size"]), "spacing": lm.spacing, "architecture":
               type(net).__name__}
    if regions:
        notes.append("region-based model: outputs are independent sigmoid regions")
    return lm
