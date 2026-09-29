"""Gradient-based explanations used by the cinematic renderer and movies.

One extra forward pass with gradients (model stays in ``eval()``; parameters are
never updated) provides, for every spatial stage:

* **unit field**: take the stage's most active unit (front channel, peak
  position) and backpropagate it to the *input*.  |∂unit/∂input| is its
  empirical receptive field (what that unit "sees"); its extent grows with depth.
* **dependency map**: the same unit backpropagated to the *previous stage*
  shows which positions of that stage feed it (a local window for convolutions,
  the whole map for global pooling, scattered tokens for attention).  The
  renderer draws this as a light beam between stages.

and, for heads:

* **Grad-CAM** for classification / binary outputs on the last spatial stage.
* **weight × activation contributions** for linear heads: which latent units push
  the predicted class up or down.

Everything is optional and fails soft: a stage whose gradient cannot be computed
(e.g. in-place ops, non-differentiable code) is simply skipped.
"""
from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .capture import PreparedInputs, flatten_tensors


@dataclass
class UnitField:
    stage: str
    channel: int
    position: Tuple[int, ...]            # in the stage's spatial grid
    grid: Tuple[int, ...]
    erf: Optional[np.ndarray] = None     # |d unit / d input| (normalised), input spatial shape
    rf_px: float = 0.0                   # effective receptive-field extent (input pixels / voxels)
    rf_box: Optional[Tuple[int, ...]] = None
    dep_stage: Optional[str] = None
    dep_map: Optional[np.ndarray] = None  # normalised dependency over the predecessor's grid
    value: float = 0.0


@dataclass
class Contribution:
    head: str
    source: str                           # stage whose vector feeds the head
    target_index: int
    target_label: str
    indices: List[int]
    values: List[float]
    total: float


@dataclass
class Explanation:
    units: Dict[str, UnitField] = field(default_factory=dict)
    gradcam: Dict[str, np.ndarray] = field(default_factory=dict)       # output key -> input-shaped map
    gradcam_stage: Dict[str, str] = field(default_factory=dict)
    contributions: Dict[str, Contribution] = field(default_factory=dict)  # output key -> contribution
    notes: List[str] = field(default_factory=list)
    strategy: str = "energy"


# ---------------------------------------------------------------------------


def _canon(t: torch.Tensor, summ, b: int) -> Optional[torch.Tensor]:
    """Batch item of a stage tensor in [C, *spatial] layout (tokens -> [F, gh, gw])."""
    if t is None or summ is None:
        return None
    x = t[b] if t.dim() >= 2 else t
    if summ.kind == "image2d":
        if x.dim() != 3:
            return None
        return x.permute(2, 0, 1) if summ.channels_last else x
    if summ.kind == "volume3d":
        if x.dim() != 4:
            return None
        return x.permute(3, 0, 1, 2) if summ.channels_last else x
    if summ.kind == "tokens" and len(summ.spatial) == 2:
        k = summ.n_special
        gh, gw = summ.spatial
        if x.dim() != 2 or x.shape[0] != k + gh * gw:
            return None
        return x[k:].transpose(0, 1).reshape(x.shape[1], gh, gw)
    return None


def _replace(obj, old, new):
    if obj is old:
        return new
    if isinstance(obj, dict):
        return {k: _replace(v, old, new) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        vals = [_replace(v, old, new) for v in obj]
        return type(obj)(*vals) if hasattr(obj, "_fields") else type(obj)(vals)
    return obj


def _extent(m: np.ndarray, frac: float = 0.1) -> Tuple[float, Tuple[int, ...]]:
    """Bounding box and side length of the region where m >= frac * max."""
    if m.max() <= 0:
        return 0.0, tuple([0] * (2 * m.ndim))
    idx = np.argwhere(m >= frac * m.max())
    lo, hi = idx.min(0), idx.max(0) + 1
    side = float(np.max(hi - lo))
    return side, tuple(int(v) for v in np.concatenate([lo, hi]))


def _smooth(m: torch.Tensor) -> torch.Tensor:
    if m.dim() == 2:
        return F.avg_pool2d(m[None, None], 3, 1, 1)[0, 0]
    if m.dim() == 3:
        return F.avg_pool3d(m[None, None], 3, 1, 1)[0, 0]
    return m


def compute_explanations(model: nn.Module, prep: PreparedInputs, res, max_units: int = 16) -> Explanation:
    """Run one gradient-enabled forward pass and derive unit fields, Grad-CAM and contributions."""
    ex = Explanation(strategy=res.config.channel_strategy)
    g = res.graph
    cfg = res.config
    b = cfg.batch_index
    # choose the image-like input to differentiate with respect to
    img_inputs = [(n, t) for n, t in prep.named if t.is_floating_point() and t.dim() in (4, 5)]
    if not img_inputs:
        img_inputs = [(n, t) for n, t in prep.named if t.is_floating_point()]
    if not img_inputs:
        ex.notes.append("no floating-point input to explain")
        return ex
    in_name, x0 = img_inputs[0]
    in_key = f"input:{in_name}"
    x = x0.detach().clone().requires_grad_(True)
    args = _replace(prep.args, x0, x)
    kwargs = _replace(prep.kwargs, x0, x)

    stages = [s for s in g.ordered() if s.call is not None]
    wanted: Dict[str, Dict[int, List[Tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for s in stages:
        wanted[s.call.name][s.call.call_index].append((s.capture, s.key))
    acts: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = defaultdict(int)
    modules = dict(model.named_modules())
    handles = []

    def make_hook(name):
        def hook(module, a, out):
            ci = counts[name]
            counts[name] += 1
            for mode, key in wanted[name].get(ci, []):
                src = out if mode == "output" else a
                ts = flatten_tensors(src)
                if ts:
                    acts[key] = ts[0][1]
        return hook

    was_training = model.training
    model.eval()
    try:
        for name in wanted:
            if name in modules:
                handles.append(modules[name].register_forward_hook(make_hook(name)))
        with torch.enable_grad():
            out = model(*args, **kwargs)
            outs = dict((p or "output", t) for p, t in flatten_tensors(out))
            _unit_fields(ex, g, acts, x, in_key, b, max_units)
            _gradcam(ex, res, g, acts, outs, x, b)
            _contributions(ex, res, g, acts, modules, b)
    except Exception as e:  # never break the visualisation
        warnings.warn(f"neural_flow: explanations skipped ({type(e).__name__}: {e})")
        ex.notes.append(f"explanations failed: {e}")
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    return ex


def _spatial_pred(g, key: str) -> Optional[str]:
    """Nearest predecessor (main path first) that is spatial or the input."""
    seen = set()
    todo = [key]
    while todo:
        k = todo.pop(0)
        preds = sorted(g.preds(k), key=lambda e: (e.kind != "main", -g.stages[e.src].order))
        for e in preds:
            if e.src in seen:
                continue
            seen.add(e.src)
            s = g.stages[e.src]
            if s.kind == "input":
                return e.src
            if s.summary is not None and (s.summary.is_spatial or s.summary.kind in ("image2d", "volume3d")):
                return e.src
            todo.append(e.src)
    return None


def _unit_fields(ex, g, acts, x, in_key, b, max_units):
    spatial = [s for s in g.ordered() if s.key in acts and s.summary is not None and s.summary.is_spatial]
    for s in spatial[:max_units]:
        A = _canon(acts[s.key], s.summary, b)
        if A is None or not A.requires_grad:
            continue
        strategy_sel = s.summary.selections
        c = None
        for strat in ("_forced", ex.strategy, "energy", "variance"):
            if strategy_sel.get(strat):
                c = strategy_sel[strat][0]
                break
        if c is None or c >= A.shape[0]:
            c = int(A.flatten(1).pow(2).mean(1).argmax())
        with torch.no_grad():
            pos = np.unravel_index(int(A[c].argmax()), tuple(A.shape[1:]))
        unit = A[c][pos]
        pred = _spatial_pred(g, s.key)
        targets = [x]
        pred_t = None
        if pred is not None and pred != in_key and pred in acts:
            pred_t = acts[pred]
            if pred_t.requires_grad:
                targets.append(pred_t)
            else:
                pred_t = None
        try:
            grads = torch.autograd.grad(unit, targets, retain_graph=True, allow_unused=True)
        except RuntimeError as e:
            ex.notes.append(f"unit field for {s.key} skipped: {e}")
            continue
        uf = UnitField(s.key, int(c), tuple(int(p) for p in pos), tuple(A.shape[1:]), value=float(unit.detach()))
        gx = grads[0]
        if gx is not None:
            gi = gx[b].detach().abs().float()
            gi = gi.sum(0) if gi.dim() >= 3 else gi
            gi = _smooth(gi)
            m = gi.cpu().numpy()
            m = m / (m.max() + 1e-12)
            uf.erf = m.astype(np.float32)
            uf.rf_px, uf.rf_box = _extent(m, 0.1)
        if pred_t is not None and len(grads) > 1 and grads[1] is not None:
            ps = g.stages[pred].summary
            gp = _canon(grads[1], ps, b)
            if gp is not None:
                dm = gp.detach().abs().float().sum(0).cpu().numpy()
                uf.dep_map = (dm / (dm.max() + 1e-12)).astype(np.float32)
                uf.dep_stage = pred
        elif pred == in_key and uf.erf is not None:
            uf.dep_map, uf.dep_stage = uf.erf, in_key
        ex.units[s.key] = uf


def _gradcam(ex, res, g, acts, outs, x, b):
    for key, view in res.views.items():
        if view.kind not in ("classification", "binary", "multilabel", "regression"):
            continue
        name = key[4:] if key.startswith("out:") else key
        o = outs.get(name)
        if o is None or not o.requires_grad:
            continue
        srcs = res.out_nodes.get(key) or []
        if not srcs:
            continue
        cand = [s for s in g.ordered() if s.key in acts and s.summary is not None and s.summary.is_spatial
                and s.key in (g.ancestors(srcs[0]) | {srcs[0]})]
        ob = o[b].reshape(-1)
        k = int(ob.argmax()) if ob.numel() > 1 else 0
        # last spatial stage whose gradient is non-trivial (a ViT's final patch tokens do not reach
        # the CLS output, so Grad-CAM falls back to an earlier block)
        for last in reversed(cand[-4:]):
            try:
                grad = torch.autograd.grad(ob[k], acts[last.key], retain_graph=True, allow_unused=True)[0]
            except RuntimeError:
                continue
            if grad is None:
                continue
            A = _canon(acts[last.key], last.summary, b)
            G = _canon(grad, last.summary, b)
            if A is None or G is None or float(G.abs().max()) == 0.0:
                continue
            w = G.detach().flatten(1).mean(1)
            cam = torch.relu((w.view(-1, *([1] * (A.dim() - 1))) * A.detach()).sum(0))
            if float(cam.max()) <= 0:
                continue
            size = tuple(x.shape[2:])
            if len(size) == cam.dim():
                mode = "bilinear" if cam.dim() == 2 else "trilinear"
                cam = F.interpolate(cam[None, None].float(), size=size, mode=mode, align_corners=False)[0, 0]
            c = cam.cpu().numpy()
            ex.gradcam[key] = (c / (c.max() + 1e-12)).astype(np.float32)
            ex.gradcam_stage[key] = last.key
            break


def _linear_of(module: nn.Module, in_features: int) -> Optional[nn.Linear]:
    if isinstance(module, nn.Linear) and module.in_features == in_features:
        return module
    lin = [m for m in module.modules() if isinstance(m, nn.Linear)]
    if len(lin) == 1 and lin[0].in_features == in_features:
        return lin[0]
    return None


def _contributions(ex, res, g, acts, modules, b):
    for key, view in res.views.items():
        srcs = res.out_nodes.get(key) or []
        if not srcs:
            continue
        head = g.stages.get(srcs[0])
        if head is None or head.call is None:
            continue
        preds = [e.src for e in g.preds(head.key) if e.kind == "main"]
        if not preds:
            continue
        src = g.stages[preds[0]]
        sv = src.summary
        if sv is None or sv.kind not in ("vector",) and not (sv.spatial and max(sv.spatial) <= 1):
            continue
        z = acts.get(src.key)
        if z is None:
            continue
        z = z[b].detach().reshape(-1).float()
        lin = _linear_of(modules.get(head.call.name), z.numel())
        if lin is None:
            continue
        with torch.no_grad():
            logits = F.linear(z, lin.weight.float(), None if lin.bias is None else lin.bias.float())
            k = int(logits.argmax()) if logits.numel() > 1 else 0
            contrib = lin.weight[k].float() * z
        order = torch.argsort(contrib.abs(), descending=True)[:24].tolist()
        label = view.headline if view.kind == "classification" else view.title
        ex.contributions[key] = Contribution(head.key, src.key, k, label, order,
                                             [float(contrib[i]) for i in order], float(contrib.sum()))
