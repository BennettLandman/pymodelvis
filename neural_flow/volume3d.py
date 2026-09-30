"""Whole-volume inference for 3-D models: sliding windows and flattened 2-D views.

Most 3-D segmentation networks are trained on a fixed region of interest (e.g.
96³) and applied to a whole scan with *sliding-window* inference: overlapping
patches are run one at a time and their outputs are blended with a Gaussian
weight.  A single forward pass therefore shows only one patch.

``sliding_window=True`` keeps both views honest:

* the **stages** show the one patch the network actually processed (centred on
  the foreground, or at ``roi_center``), and
* the **output card** shows the fused prediction over the *whole* volume, with
  the traced patch outlined.

:func:`sliding_window_infer` is self-contained (no MONAI needed) and bounded in
memory: when the fused logits of a many-class model (e.g. 133 brain structures)
would exceed ``sw_max_mb`` they are accumulated on a coarser grid.

:func:`sliding_window_movie` renders the inference itself: one frame per patch,
the stages following the window through the volume while the fused
segmentation assembles in the output card.

:func:`flatten_result` implements ``flat_3d=True``: every 3-D stage is squashed
into a 2-D projection so a volumetric network can be drawn like a 2-D CNN.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .capture import flatten_tensors


# ---------------------------------------------------------------------------
# patch geometry
# ---------------------------------------------------------------------------


def patch_starts(size: int, roi: int, overlap: float) -> List[int]:
    """Start offsets of windows of length ``roi`` covering ``size`` with the given overlap."""
    if size <= roi:
        return [0]
    step = max(1, int(round(roi * (1.0 - overlap))))
    starts = list(range(0, size - roi + 1, step))
    if starts[-1] != size - roi:
        starts.append(size - roi)
    return starts


def patch_grid(spatial: Sequence[int], roi: Sequence[int], overlap: float) -> List[Tuple[int, ...]]:
    """All window starts, in raster order (last axis fastest)."""
    axes = [patch_starts(s, r, overlap) for s, r in zip(spatial, roi)]
    out: List[Tuple[int, ...]] = [()]
    for a in axes:
        out = [o + (v,) for o in out for v in a]
    return out


def foreground_center(x: torch.Tensor) -> Tuple[int, ...]:
    """Centre of mass of the above-median signal of a [1, C, *S] volume (voxel indices)."""
    v = x[0].float().abs().mean(0) if x.shape[1] > 1 else x[0, 0].float()
    v = v - v.median()
    w = v.clamp_min(0)
    tot = float(w.sum())
    if tot <= 0:
        return tuple(s // 2 for s in v.shape)
    idx = [torch.arange(s, dtype=torch.float32) for s in v.shape]
    c = []
    for d in range(v.dim()):
        shape = [1] * v.dim()
        shape[d] = -1
        c.append(float((w * idx[d].view(shape)).sum() / tot))
    return tuple(int(round(ci)) for ci in c)


def patch_at(spatial: Sequence[int], roi: Sequence[int], center: Sequence[int]) -> Tuple[int, ...]:
    """Window start so that the window is centred on ``center`` (clamped to the volume)."""
    return tuple(int(np.clip(round(c - r / 2), 0, max(s - r, 0))) for s, r, c in zip(spatial, roi, center))


def crop(x: torch.Tensor, start: Sequence[int], roi: Sequence[int]) -> torch.Tensor:
    """Crop ``x`` [B, C, *S] at ``start`` with size ``roi`` (zero-padded at the far edge if needed)."""
    sl = [slice(None), slice(None)] + [slice(s, s + r) for s, r in zip(start, roi)]
    out = x[tuple(sl)]
    pad = []
    for r, o in zip(reversed(list(roi)), reversed(list(out.shape[2:]))):
        pad += [0, max(0, r - o)]
    if any(pad):
        out = F.pad(out, pad)
    return out


def _gaussian(roi: Sequence[int], sigma_scale: float = 0.125, device=None) -> torch.Tensor:
    ws = []
    for r in roi:
        c = (r - 1) / 2
        s = max(r * sigma_scale, 1e-3)
        g = torch.exp(-0.5 * ((torch.arange(r, dtype=torch.float32) - c) / s) ** 2)
        ws.append(g)
    w = ws[0]
    for g in ws[1:]:
        w = w[..., None] * g
    w = w / w.max()
    return w.clamp_min(1e-3).to(device) if device is not None else w.clamp_min(1e-3)


# ---------------------------------------------------------------------------
# sliding-window inference
# ---------------------------------------------------------------------------


@dataclass
class SlidingWindowState:
    """Fused output so far (per spatial output leaf) — passed to callbacks and returned at the end."""

    spatial: Tuple[int, ...]
    roi: Tuple[int, ...]
    starts: List[Tuple[int, ...]]
    factor: Dict[str, int] = field(default_factory=dict)          # accumulation grid coarsening per leaf
    acc: Dict[str, torch.Tensor] = field(default_factory=dict)    # [C, *S/f] weighted logits
    weight: Dict[str, torch.Tensor] = field(default_factory=dict)  # [*S/f] weights
    done: int = 0
    frozen: Dict[str, torch.Tensor] = field(default_factory=dict)  # snapshots: fused output, compact dtype

    def names(self) -> List[str]:
        return list(self.acc) + [k for k in self.frozen if k not in self.acc]

    def snapshot(self) -> "SlidingWindowState":
        """A small copy holding only the fused outputs (labels as uint8 / int16), for movies."""
        frozen = {}
        for k in self.acc:
            t = self.fused(k)
            if not t.is_floating_point():
                t = t.to(torch.uint8 if int(t.max()) < 256 else torch.int16)
            frozen[k] = t
        return SlidingWindowState(self.spatial, self.roi, self.starts, dict(self.factor), done=self.done,
                                  frozen=frozen)

    def fused(self, name: str, as_labels: bool = True) -> torch.Tensor:
        """Current fused output at full resolution: labels [1, 1, *S] (int) for multi-class outputs,
        probabilities / values [1, 1, *S] for single-channel outputs.  Unvisited voxels are 0."""
        if name in self.frozen and name not in self.acc:
            t = self.frozen[name]
            return t if t.is_floating_point() else t.to(torch.int32)
        acc, w = self.acc[name], self.weight[name]
        seen = w > 0
        avg = acc / w.clamp_min(1e-8)
        if avg.shape[0] > 1 and as_labels:
            out = avg.argmax(0).to(torch.int32)
            out = torch.where(seen, out, torch.zeros_like(out))
            out = out[None, None].float()
            mode = "nearest"
        else:
            out = torch.where(seen, avg[0], torch.zeros_like(avg[0]))[None, None]
            mode = "trilinear"
        f = self.factor[name]
        if f > 1:
            out = F.interpolate(out, size=self.spatial, mode=mode) if mode == "nearest" else \
                F.interpolate(out, size=self.spatial, mode=mode, align_corners=False)
        out = out[(slice(None), slice(None)) + tuple(slice(0, s) for s in self.spatial)]
        return out.to(torch.int32) if (avg.shape[0] > 1 and as_labels) else out


def _leaves(out: Any) -> List[Tuple[str, torch.Tensor]]:
    return [(p or "output", t) for p, t in flatten_tensors(out)]


@torch.no_grad()
def sliding_window_infer(model, x: torch.Tensor, roi: Sequence[int], overlap: float = 0.25,
                         max_mb: float = 1500.0, callback: Optional[Callable[[int, Tuple[int, ...], SlidingWindowState], None]] = None,
                         starts: Optional[List[Tuple[int, ...]]] = None, progress: bool = False) -> SlidingWindowState:
    """Gaussian-weighted sliding-window inference over ``x`` [1, C, *S].

    Only output leaves whose spatial size equals ``roi`` are fused (segmentation /
    dense maps); vector outputs of a patch are ignored.  Fused logits are kept on
    the CPU; a leaf with ``C`` channels is accumulated on a grid coarsened by an
    integer factor when ``C × voxels × 4 bytes`` would exceed ``max_mb``.
    """
    if x.dim() != 5:
        raise ValueError("sliding_window_infer expects a [1, C, X, Y, Z] tensor")
    roi = tuple(int(r) for r in roi)
    S = tuple(int(s) for s in x.shape[2:])
    Sp = tuple(max(s, r) for s, r in zip(S, roi))
    if Sp != S:  # pad small volumes up to the window size
        x = crop(x, (0, 0, 0), Sp)
    starts = starts if starts is not None else patch_grid(Sp, roi, overlap)
    dev = next((p.device for p in model.parameters()), torch.device("cpu"))
    gw = _gaussian(roi)
    st = SlidingWindowState(S, roi, starts)
    was_training = model.training
    model.eval()
    from .adapters import inference_view

    view = inference_view(model)
    view.__enter__()
    try:
        for i, s0 in enumerate(starts):
            patch = crop(x, s0, roi).to(dev)
            out = model(patch)
            for name, t in _leaves(out):
                if t.dim() != 5 or tuple(t.shape[2:]) != roi:
                    continue
                t = t[0].float().cpu()                              # [C, *roi]
                C = t.shape[0]
                if name not in st.acc:
                    f = 1
                    while C * np.prod([math.ceil(v / f) for v in Sp]) * 4 / 2 ** 20 > max_mb and f < 8:
                        f += 1
                    grid = tuple(math.ceil(v / f) for v in Sp)
                    st.factor[name] = f
                    st.acc[name] = torch.zeros((C,) + grid)
                    st.weight[name] = torch.zeros(grid)
                    if f > 1:
                        warnings.warn(f"neural_flow: fused output '{name}' ({C} channels) accumulated on a "
                                      f"{f}× coarser grid to stay within {max_mb:.0f} MB")
                f = st.factor[name]
                w = gw
                if f > 1:
                    t = F.avg_pool3d(t[None], f, f, ceil_mode=True)[0]
                    w = F.avg_pool3d(gw[None, None], f, f, ceil_mode=True)[0, 0]
                g0 = tuple(s // f for s in s0)
                sl = tuple(slice(a, a + b) for a, b in zip(g0, t.shape[1:]))
                acc, wt = st.acc[name], st.weight[name]
                # clip at the far edge of the (coarse) grid
                sl = tuple(slice(a.start, min(a.stop, n)) for a, n in zip(sl, wt.shape))
                tt = t[(slice(None),) + tuple(slice(0, a.stop - a.start) for a in sl)]
                ww = w[tuple(slice(0, a.stop - a.start) for a in sl)]
                acc[(slice(None),) + sl] += tt * ww
                wt[sl] += ww
            st.done = i + 1
            if callback is not None:
                callback(i, s0, st)
            if progress:
                print(f"\r  sliding window  {i + 1}/{len(starts)}", end="", flush=True)
        if progress:
            print()
    finally:
        view.__exit__(None, None, None)
        model.train(was_training)
    return st


# ---------------------------------------------------------------------------
# integration with trace_model
# ---------------------------------------------------------------------------


def _single_volume(inputs) -> Optional[torch.Tensor]:
    if isinstance(inputs, torch.Tensor) and inputs.dim() == 5:
        return inputs
    return None


def sliding_window_trace(model, inputs, config, kw: Dict[str, Any]):
    """``trace_model`` with ``sliding_window=True`` (see module docstring)."""
    from .api import trace_model
    from .config import make_config
    from .outputs import interpret_output
    from .raster import canonical_fov

    cfg = make_config(config, **kw)
    x = _single_volume(inputs)
    if x is None:
        warnings.warn("neural_flow: sliding_window needs a single [1, C, X, Y, Z] input; ignored")
        return trace_model(model, inputs, config=config, **dict(kw, sliding_window=False))
    S = tuple(x.shape[2:])
    roi = tuple(cfg.roi_size) if cfg.roi_size else None
    if roi is None:
        raise ValueError("sliding_window=True needs roi_size, e.g. roi_size=(96, 96, 96)")
    if len(roi) == 1:
        roi = roi * 3
    center = tuple(cfg.roi_center) if cfg.roi_center else foreground_center(x)
    s0 = patch_at(S, roi, center)
    patch = crop(x, s0, roi)
    kw2 = dict(kw, sliding_window=False, flat_3d=False)
    if cfg.voxel_spacing is not None:
        kw2["physical_fov"] = canonical_fov(roi, cfg.voxel_spacing, cfg.volume_axes)
    res = trace_model(model, patch, config=config, **kw2)
    st = sliding_window_infer(model, x, roi, cfg.sw_overlap, cfg.sw_max_mb)
    res.config = res.config.updated(sliding_window=True, flat_3d=cfg.flat_3d)
    apply_fused_outputs(res, st, x, s0)
    if cfg.flat_3d:
        flatten_result(res, cfg.flat_3d if isinstance(cfg.flat_3d, str) else "max")
    return res


def apply_fused_outputs(res, st: SlidingWindowState, x: torch.Tensor, s0: Tuple[int, ...],
                        final: bool = True) -> None:
    """Replace the patch's dense output views with the fused whole-volume prediction."""
    from .outputs import interpret_output
    from .raster import canonical_fov

    cfg = res.config
    S = tuple(x.shape[2:])
    for key, ov in list(res.views.items()):
        if ov.kind != "segmentation" or ov.name not in st.names():
            continue
        fused = st.fused(ov.name)
        nv = interpret_output(ov.name, fused, None, cfg.updated(output_types={ov.name: "segmentation"}),
                              input_spatial=S)
        nv.title = ov.title
        n = len(st.starts)
        m = nv.mask
        if m is not None and m.dtype.kind in "iu" and m.size and int(m.max()) < 256 and int(m.min()) >= 0:
            nv.mask = m = m.astype(np.uint8)                   # whole-volume labels: keep them small
        n_lab = int(len(np.unique(m))) if m is not None and m.dtype.kind in "iu" else None
        head = f"{n_lab} labels" if n_lab is not None else nv.subline.split(" · ")[0]
        nv.subline = f"{head} · window {st.done}/{n}" if not final else f"{head} · fused from {n} window{'s' if n != 1 else ''}"
        res.views[key] = nv
    lo = tuple(s0)
    hi = tuple(a + r for a, r in zip(s0, st.roi))
    res.context.update({
        "seg_base": x[0].detach().float().cpu().numpy(),
        "seg_fov": canonical_fov(S, cfg.voxel_spacing, cfg.volume_axes) if cfg.voxel_spacing else None,
        "seg_boxes": [(_canon_idx(lo, S, cfg.volume_axes), _canon_idx(hi, S, cfg.volume_axes))],
        "sliding_window": {"volume": S, "roi": st.roi, "start": s0, "windows": len(st.starts), "done": st.done},
    })
    if res.context.get("flat_3d"):
        flatten_views(res)


def _canon_idx(p: Sequence[int], S: Sequence[int], axes: str) -> Tuple[int, int, int]:
    """Tensor-order voxel coordinate -> canonical [x, y, z] coordinate (see raster.canonical_volume)."""
    if axes == "dhw":
        d, h, w = p
        return (w, S[1] - h, d)
    if axes == "zyx":
        return (int(p[2]), int(p[1]), int(p[0]))
    return tuple(int(v) for v in p)


# ---------------------------------------------------------------------------
# movie: watch sliding-window inference
# ---------------------------------------------------------------------------


def sliding_window_movie(model, volume: torch.Tensor, roi: Sequence[int], output: str = "sliding_window.mp4", *,
                         overlap: float = 0.25, max_windows: Optional[int] = None, fps: int = 4,
                         title: Optional[str] = None, subtitle: Optional[str] = None, **kw) -> str:
    """Movie of whole-volume inference: one frame per window.

    The stages follow the window through the volume (fixed stages, channels and
    colour scales, as in :func:`animate_inputs`) while the output card shows the
    segmentation fused so far over the whole scan, with the current window
    outlined.  ``max_windows`` subsamples long window sequences evenly (the fused
    output still uses every window).
    """
    from .movie import animate_inputs

    if volume.dim() != 5:
        raise ValueError("volume must be [1, C, X, Y, Z]")
    roi = tuple(int(r) for r in (roi if len(roi) == 3 else tuple(roi) * 3))
    S = tuple(volume.shape[2:])
    starts = patch_grid(tuple(max(s, r) for s, r in zip(S, roi)), roi, overlap)
    shown = list(range(len(starts)))
    if max_windows is not None and len(starts) > max_windows:
        shown = sorted({int(round(v)) for v in np.linspace(0, len(starts) - 1, max_windows)})
    # fused state after each shown window (fusing every window, in order)
    snaps: Dict[int, SlidingWindowState] = {}

    def cb(i, s0, st):
        if i in shown:
            snaps[i] = st.snapshot()                    # fused labels only: small even for many classes

    sliding_window_infer(model, volume, roi, overlap, kw.pop("sw_max_mb", 1500.0), callback=cb)
    frames = [crop(volume, starts[i], roi) for i in shown]
    spacing = kw.get("voxel_spacing")
    axes = kw.get("volume_axes", "xyz")
    if spacing is not None:
        from .raster import canonical_fov

        kw["physical_fov"] = canonical_fov(roi, spacing, axes)

    def hook(t, res):
        i = shown[t]
        apply_fused_outputs(res, snaps[i], volume, starts[i], final=False)

    labels = [f"window {i + 1}/{len(starts)}  ·  at {starts[i]}" for i in shown]
    sub = subtitle or (f"sliding-window inference  ·  volume {'×'.join(map(str, S))}  ·  window "
                       f"{'×'.join(map(str, roi))}  ·  {len(starts)} windows, {int(overlap * 100)}% overlap")
    return animate_inputs(model, frames, output=output, fps=fps, frame_labels=labels, title=title, subtitle=sub,
                          frame_hook=hook, thumb_fn=_window_thumb(volume, starts, shown, axes), **kw)


def _window_thumb(volume: torch.Tensor, starts, shown, axes: str):
    """Film-strip thumbnails: a coronal slice of the whole volume with the window outlined."""
    from .raster import canonical_volume, robust_norm

    V = canonical_volume(volume[0, 0].float().cpu().numpy(), axes)
    S = tuple(volume.shape[2:])

    def thumb(t, res):
        i = shown[t]
        sw = res.context.get("sliding_window", {})
        roi = sw.get("roi")
        lo = _canon_idx(starts[i], S, axes)
        hi = _canon_idx(tuple(a + r for a, r in zip(starts[i], roi)), S, axes)
        yc = int(np.clip((lo[1] + hi[1]) // 2, 0, V.shape[1] - 1))
        sl = np.flipud(V[:, yc, :].T)                         # rows = z (top = superior), cols = x
        g = robust_norm(sl, (1, 99.5))
        img = np.concatenate([np.repeat(g[..., None], 3, -1), np.ones(g.shape + (1,))], -1).astype(np.float32)
        Z = V.shape[2]
        r0, r1 = Z - min(hi[2], Z), Z - lo[2]
        c0, c1 = lo[0], min(hi[0], V.shape[0])
        col = np.array([0.98, 0.75, 0.25, 1.0], np.float32)
        for r in (r0, max(r1 - 1, r0)):
            img[r, c0:c1] = col
        for c in (c0, max(c1 - 1, c0)):
            img[r0:r1, c] = col
        return img

    return thumb


# ---------------------------------------------------------------------------
# flat_3d: draw a volumetric network as if it were 2-D
# ---------------------------------------------------------------------------


def _project(a: np.ndarray, how: str, axes: str) -> np.ndarray:
    """[..., *S3] (tensor order) -> [..., rows, cols] axial projection in radiological display order."""
    from .raster import canonical_volume

    lead = a.shape[:-3]
    flat = a.reshape((-1,) + a.shape[-3:])
    outs = []
    for v in flat:
        V = canonical_volume(v, axes)
        P = V.max(2) if how == "max" else V.mean(2)
        outs.append(np.flipud(P.T))
    return np.stack(outs).reshape(lead + outs[0].shape)


def _axial_slice(a: np.ndarray, axes: str) -> Tuple[int, np.ndarray]:
    """[C, *S3] volume -> (z, [C, rows, cols] axial slice through the signal's centre of mass)."""
    from .raster import canonical_volume, center_of_mass

    V0 = canonical_volume(a[0], axes)
    z = center_of_mass(V0)[2]
    sl = np.stack([np.flipud(canonical_volume(v, axes)[:, :, z].T) for v in a])
    return int(z), np.ascontiguousarray(sl).astype(np.float32)


def flatten_result(res, how: str = "max") -> None:
    """``flat_3d``: squash every 3-D stage of ``res`` into a 2-D axial view (in place).

    Feature maps (and their PCA / energy maps, receptive fields and Grad-CAM) are
    projected along the superior–inferior axis (``how`` = ``"max"`` or ``"mean"``);
    the input volume and a 3-D segmentation are shown as the axial slice through
    the input's centre of mass, so the anatomy stays legible.  Stages then render
    with the 2-D feature-map decks.
    """
    cfg = res.config
    axes = cfg.volume_axes
    how = "mean" if how == "mean" else "max"
    for st in res.graph.stages.values():
        sm = st.summary
        if sm is None or sm.kind != "volume3d":
            continue
        if sm.image is not None and st.kind == "input":
            z, sm.image = _axial_slice(sm.image, axes)
            res.context.setdefault("flat_z", z)
        for attr in ("maps", "pca_maps", "mean_abs_map", "energy_map", "token_norm_map"):
            v = getattr(sm, attr)
            if v is not None and v.ndim >= 3:
                setattr(sm, attr, _project(v, how, axes).astype(np.float32))
        sp = tuple(sm.spatial)
        if len(sp) == 3:
            c = _canon_shape(sp, axes)
            sm.spatial = (c[1], c[0])
        rs = tuple(sm.reduced_spatial)
        if len(rs) == 3:
            c = _canon_shape(rs, axes)
            sm.reduced_spatial = (c[1], c[0])
        sm.kind = "image2d"
        sm.notes.append(f"3-D stage flattened to a 2-D {how} projection (flat_3d)")
    ex = getattr(res, "explanation", None)
    if ex is not None:
        for uf in ex.units.values():
            if uf.erf is not None and uf.erf.ndim == 3:
                uf.erf = _project(uf.erf, "max", axes)
                uf.rf_box = _box(uf.erf)
            if uf.dep_map is not None and uf.dep_map.ndim == 3:
                uf.dep_map = _project(uf.dep_map, "max", axes)
            if len(uf.grid) == 3:
                g = _canon_shape(uf.grid, axes)
                p = _canon_idx(uf.position, uf.grid, axes)
                uf.grid = (g[1], g[0])
                uf.position = (g[1] - 1 - p[1], p[0])
        for k, cam in list(ex.gradcam.items()):
            if cam is not None and np.ndim(cam) == 3:
                ex.gradcam[k] = _project(cam, "max", axes)
    res.context["flat_3d"] = how
    flatten_views(res)


def flatten_views(res) -> None:
    """Flat mode for dense 3-D outputs: the axial slice matching the input slice."""
    from .raster import canonical_volume, center_of_mass

    axes = res.config.volume_axes
    ctx = res.context
    base3 = ctx.get("seg_base")
    z_full = None
    if base3 is not None and np.ndim(base3) == 4:
        z_full, ctx["seg_base"] = _axial_slice(base3, axes)
        ctx["seg_boxes"] = None
        ctx["seg_fov"] = None
    for key, ov in res.views.items():
        if ov.kind != "segmentation" or ov.mask is None or ov.mask.ndim != 3:
            continue
        M = canonical_volume(ov.mask, axes)
        z = z_full if z_full is not None else ctx.get("flat_z")
        if z is None:
            fg = (M > (0.5 if M.dtype.kind == "f" else 0)).astype(np.float32)
            z = center_of_mass(fg)[2] if fg.sum() > 0 else M.shape[2] // 2
        z = int(np.clip(z, 0, M.shape[2] - 1))
        ov.mask = np.ascontiguousarray(np.flipud(M[:, :, z].T))


def _canon_shape(sp: Sequence[int], axes: str) -> Tuple[int, int, int]:
    return (sp[2], sp[1], sp[0]) if axes in ("dhw", "zyx") else tuple(sp)


def _box(m: np.ndarray):
    if m.max() <= 0:
        return tuple([0] * (2 * m.ndim))
    idx = np.argwhere(m >= 0.1 * m.max())
    lo, hi = idx.min(0), idx.max(0) + 1
    return tuple(int(v) for v in np.concatenate([lo, hi]))
