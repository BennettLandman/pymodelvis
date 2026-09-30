"""Interpretation of model outputs (heads).

Output semantics are *inferred conservatively*.  A ``[B, K]`` tensor is shown as
``softmax(logits)`` only when it looks like classification (``K > 1`` and the
head / output name or supplied class names suggest it, or the values already
sum to one); otherwise the raw values are shown.  Anything can be overridden
with ``output_types={"name": "sigmoid" | "softmax" | "multilabel" |
"regression" | "segmentation" | "embedding" | "raw"}`` or a custom
``output_interpreter(name, tensor) -> OutputView | dict | str``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import FlowConfig

CLASSIFY_NAME = re.compile(r"(class|cls|logit|label|categor|diagnos|type|fc$|head$|heads|pred|malig|score)", re.I)
REGRESS_NAME = re.compile(r"(age|regress|value|volume|size|score_reg|years|dose|count|mean|estimate)", re.I)
SEG_NAME = re.compile(r"(seg|mask|label_map|parcel)", re.I)
EMB_NAME = re.compile(r"(embed|feature|proj|latent|repr|z$)", re.I)


@dataclass
class OutputView:
    name: str
    kind: str                                 # classification | multilabel | binary | regression | segmentation | embedding | raw | text
    title: str = ""
    headline: str = ""                        # big text, e.g. "tabby cat" / "64.3"
    subline: str = ""                         # e.g. "p = 0.87"
    items: List[Tuple[str, float]] = field(default_factory=list)   # (label, value) for bars
    mask: Optional[np.ndarray] = None         # [H,W] or [D,H,W] int labels / probabilities
    vector: Optional[np.ndarray] = None
    shape: Tuple[int, ...] = ()
    note: str = ""


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _names_for(name: str, cfg: FlowConfig, k: int) -> List[str]:
    cn = cfg.class_names
    if isinstance(cn, dict):
        cn = cn.get(name) or cn.get(name.split(".")[-1])
    if cn is not None and len(cn) == k:
        return list(cn)
    return [f"class {i}" for i in range(k)]


def _fmtp(p: float) -> str:
    return f"{p:.2f}" if p >= 0.01 else f"{p:.1e}"


def _fmt(v: float) -> str:
    a = abs(v)
    if a >= 1000 or (a < 0.01 and a > 0):
        return f"{v:.3g}"
    if a >= 100:
        return f"{v:.1f}"
    return f"{v:.2f}" if a < 10 else f"{v:.1f}"


def interpret_output(name: str, t: Optional[torch.Tensor], summary, cfg: FlowConfig,
                     head_name: str = "", input_spatial: Optional[Tuple[int, ...]] = None) -> OutputView:
    """Turn one output leaf into an :class:`OutputView`."""
    disp = name.split(".")[-1] if name not in ("output", "") else "prediction"
    if cfg.output_interpreter is not None and t is not None:
        res = cfg.output_interpreter(name, t)
        if isinstance(res, OutputView):
            return res
        if isinstance(res, dict):
            return OutputView(name=name, **{"kind": "text", "title": disp, **res})
        if isinstance(res, str):
            return OutputView(name=name, kind="text", title=disp, headline=res)
    shape = tuple(summary.shape) if summary is not None else (tuple(t.shape) if t is not None else ())
    otype = None
    if cfg.output_types:
        otype = cfg.output_types.get(name) or cfg.output_types.get(disp)
    if t is None:
        return OutputView(name, "raw", title=disp, headline="(large output)", shape=shape,
                          subline=f"shape {' × '.join(map(str, shape))}")
    is_labels = not (t.is_floating_point() or t.is_complex()) and t.dim() >= 3
    x = t.detach().float()
    if x.dim() >= 1 and x.shape[0] == 1 and x.dim() > 1:
        x = x[0]
    arr = x.cpu().numpy()
    hint = f"{name} {head_name}"

    # ---- spatial outputs: segmentation (or raw maps)
    if arr.ndim in (3, 4) and (otype in (None, "segmentation")):
        spatial = arr.shape[1:]
        looks_seg = otype == "segmentation" or SEG_NAME.search(hint) is not None or (
            input_spatial is not None and tuple(spatial) == tuple(input_spatial) and arr.shape[0] <= 256)
        if looks_seg or is_labels:
            if is_labels and arr.shape[0] == 1:
                lab = arr[0].astype(np.int32)
                classes = np.unique(lab)
                return OutputView(name, "segmentation", title=disp, headline="segmentation",
                                  subline=f"{len(classes)} labels · {(lab > 0).mean() * 100:.1f}% non-bg",
                                  mask=lab, shape=shape)
            if arr.shape[0] == 1:
                prob = _sigmoid(arr[0]) if (arr.min() < 0 or arr.max() > 1) else arr[0]
                mask = prob
                frac = float((prob > 0.5).mean())
                return OutputView(name, "segmentation", title=disp, headline="segmentation",
                                  subline=f"{frac * 100:.1f}% foreground", mask=mask, shape=shape)
            lab = arr.argmax(0)
            n_fg = int((lab > 0).sum())
            classes = np.unique(lab)
            return OutputView(name, "segmentation", title=disp, headline="segmentation",
                              subline=f"{len(classes)} labels · {n_fg / lab.size * 100:.1f}% non-bg",
                              mask=lab.astype(np.int32), shape=shape)
    # ---- vectors
    if arr.ndim == 0 or arr.size == 1:
        v = float(arr.reshape(-1)[0])
        if otype in ("sigmoid", "binary") or (otype is None and re.search(r"(prob|malig|binary|risk|present)", hint, re.I)):
            p = _sigmoid(v) if (v < 0 or v > 1) else v
            return OutputView(name, "binary", title=disp, headline=f"{p:.2f}", subline="sigmoid probability",
                              items=[(disp, p)], shape=shape)
        return OutputView(name, "regression", title=disp, headline=_fmt(v), subline="predicted value", shape=shape)
    if arr.ndim == 1:
        k = arr.shape[0]
        if otype in ("multilabel", "multilabel_probs"):
            p = _sigmoid(arr) if otype == "multilabel" else np.clip(arr, 0, 1)
            names = _names_for(name, cfg, k)
            order = np.argsort(-p)[: cfg.top_k]
            return OutputView(name, "multilabel", title=disp, headline=names[order[0]],
                              subline=f"p = {p[order[0]]:.2f}  ·  {int((p > 0.5).sum())} labels > 0.5",
                              items=[(names[i], float(p[i])) for i in order], shape=shape, vector=p)
        if otype == "regression":
            return OutputView(name, "regression", title=disp, headline=", ".join(_fmt(v) for v in arr[:4]),
                              subline=f"{k} values", shape=shape, vector=arr)
        if otype == "embedding" or (otype is None and EMB_NAME.search(hint) and not CLASSIFY_NAME.search(disp)):
            return OutputView(name, "embedding", title=disp, headline=f"{k}-d embedding",
                              subline=f"‖z‖ = {np.linalg.norm(arr):.2f}", vector=arr, shape=shape)
        sums_to_one = arr.min() >= 0 and abs(arr.sum() - 1) < 1e-3
        has_names = cfg.class_names is not None and len(_names_for(name, cfg, k)) == k and not _names_for(name, cfg, k)[0].startswith("class ")
        if otype in ("softmax", "classification") or sums_to_one or has_names or (
                otype is None and k >= 2 and CLASSIFY_NAME.search(hint) and not REGRESS_NAME.search(disp)):
            p = arr if sums_to_one else _softmax(arr)
            names = _names_for(name, cfg, k)
            order = np.argsort(-p)[: cfg.top_k]
            top = order[0]
            return OutputView(name, "classification", title=disp, headline=names[top],
                              subline=f"p = {_fmtp(p[top])}" + ("" if sums_to_one else "  (softmax)"),
                              items=[(names[i], float(p[i])) for i in order], shape=shape, vector=p)
        if otype is None and REGRESS_NAME.search(disp) and k <= 8:
            return OutputView(name, "regression", title=disp, headline=", ".join(_fmt(v) for v in arr[:3]),
                              subline=f"{k} values", vector=arr, shape=shape)
        # semantics unknown: show raw values, never pretend they are probabilities
        names = _names_for(name, cfg, k)
        order = np.argsort(-arr)[: cfg.top_k]
        return OutputView(name, "raw", title=disp, headline=f"max: {names[order[0]]}",
                          subline=f"raw values · {k}-d", items=[(names[i], float(arr[i])) for i in order],
                          vector=arr, shape=shape, note="output semantics not inferred")
    return OutputView(name, "raw", title=disp, headline=f"tensor {' × '.join(map(str, shape))}",
                      subline=f"mean {arr.mean():.3g} · std {arr.std():.3g}", shape=shape)


def output_leaves(raw: Any) -> List[Tuple[str, Optional[torch.Tensor]]]:
    from .capture import flatten_tensors

    out = []
    if raw is None:
        return out
    for p, t in flatten_tensors(raw):
        out.append((p or "output", t))
    return out
