"""Movies over changing inputs.

``animate_inputs(model, frames, output="sweep.mp4")`` renders one figure per
input with everything that should stay fixed held fixed:

* the same stages (selected on the first frame),
* the same channels on every page (ranked by activity summed over *all* frames),
* the same normalisation per channel (percentiles over all frames), so
  brightness changes mean activation changes,
* the same PCA colour basis for the front pages (fitted on the middle frame),

while the things that should move do move: activations, the strongest unit and
its beam / receptive field, Grad-CAM evidence, head contributions and outputs.
A timeline under the flow tracks every output over the sequence (probabilities,
regressed values, segmented volume) with a film strip of the inputs.
"""
from __future__ import annotations

import io
import math
import os
import shutil
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .api import FlowResult, draw, trace_model
from .cinematic import FrameState, n_sheets

SERIES_COLORS = ["#fbbf24", "#67e8f9", "#f472b6", "#a3e635", "#c4b5fd", "#fb923c"]


def _split_frames(inputs) -> List[Any]:
    if isinstance(inputs, torch.Tensor):
        return [inputs[i:i + 1] for i in range(inputs.shape[0])]
    return list(inputs)


def _stage_selectors(res: FlowResult) -> List[str]:
    return [s.call.key for s in res.graph.ordered() if s.kind == "module" and s.call is not None]


def _thumb(res: FlowResult) -> Optional[np.ndarray]:
    from .cinematic import _input_rgb, _input_volume
    from .raster import robust_norm

    rgb = _input_rgb(res)
    if rgb is not None:
        return rgb
    V = _input_volume(res)
    if V is not None:
        sl = np.flipud(V[:, V.shape[1] // 2, :].T)       # coronal slice
        g = robust_norm(sl, (1, 99.5))
        return np.concatenate([np.repeat(g[..., None], 3, -1), np.ones(g.shape + (1,))], -1).astype(np.float32)
    return None


def _series(results: List[FlowResult], max_series: int = 4) -> List[Dict[str, Any]]:
    """Per-output time series for the timeline."""
    out = []
    keys = list(results[0].views)
    for key in keys:
        v0 = results[0].views[key]
        if v0.kind in ("classification", "multilabel") and v0.vector is not None:
            P = np.stack([r.views[key].vector for r in results])        # [T, K]
            top = np.argsort(-P.max(0))[:max_series]
            names = None
            cfg = results[0].config
            from .outputs import _names_for

            names = _names_for(v0.name, cfg, P.shape[1])
            for j, c in enumerate(top):
                out.append({"key": key, "label": names[c], "values": P[:, c], "unit": "p", "ylim": (0, 1)})
        elif v0.kind in ("binary",):
            vals = np.array([r.views[key].items[0][1] if r.views[key].items else np.nan for r in results])
            out.append({"key": key, "label": v0.title, "values": vals, "unit": "p", "ylim": (0, 1)})
        elif v0.kind == "regression":
            vals = []
            for r in results:
                try:
                    vals.append(float(r.views[key].headline.split(",")[0]))
                except ValueError:
                    vals.append(np.nan)
            out.append({"key": key, "label": v0.title, "values": np.array(vals), "unit": "", "ylim": None})
        elif v0.kind == "segmentation":
            vals = []
            for r in results:
                m = r.views[key].mask
                vals.append(float((m > 0.5).mean() if m.dtype.kind == "f" else (m > 0).mean()) * 100 if m is not None else np.nan)
            out.append({"key": key, "label": f"{v0.title} (% fg)", "values": np.array(vals), "unit": "%", "ylim": None})
    return out


def _make_footer(series, thumbs, T: int, t: int, frame_labels: Optional[Sequence[str]]):
    """Returns a draw_footer(fig, ax, rect, th, fs, u) callback for the cinematic renderer."""

    def draw_footer(fig, ax, rect, th, fs, u):
        X0, Y0, W, H = rect
        pad = 0.45
        x0, x1 = X0 + pad + 3.4, X0 + W - pad - 1.6
        # film strip
        strip_h = float(np.clip(H * 0.42, 0.7, 2.4))
        n = min(len(thumbs), max(6, int((x1 - x0) / (strip_h * 1.08))))
        idxs = np.unique(np.linspace(0, T - 1, n).round().astype(int))
        cell = (x1 - x0) / max(len(idxs), 1)
        sy1 = Y0 + H - 0.15
        for j, i in enumerate(idxs):
            im = thumbs[i]
            if im is None:
                continue
            w = min(cell * 0.92, strip_h * im.shape[1] / im.shape[0])
            cx = x0 + (j + 0.5) * cell
            ax.imshow(im, extent=(cx - w / 2, cx + w / 2, sy1 - strip_h, sy1), zorder=3,
                      alpha=1.0 if abs(i - t) <= (T / n) / 2 else 0.35)
        # current-frame marker on the strip
        cxm = x0 + (t + 0.5) / T * (x1 - x0)
        ax.plot([cxm, cxm], [Y0 + 0.25, sy1 + 0.05], color=th["accent"], lw=1.4, zorder=6, alpha=0.9)
        # timeline plot
        py0, py1 = Y0 + 0.3, sy1 - strip_h - 0.15
        ax.plot([x0, x1], [py0, py0], color=th["faint"], lw=0.6, zorder=2)
        groups: Dict[Any, List[Dict[str, Any]]] = {}
        for sser in series:
            groups.setdefault(sser["unit"] + sser["key"], []).append(sser)
        ci = 0
        ly = py1
        for gk, ss in groups.items():
            allv = np.concatenate([s_["values"] for s_ in ss])
            ylim = ss[0]["ylim"] or (np.nanmin(allv), np.nanmax(allv))
            lo, hi = ylim
            if hi - lo < 1e-9:
                hi = lo + 1
            for s_ in ss:
                col = SERIES_COLORS[ci % len(SERIES_COLORS)]
                ci += 1
                v = s_["values"]
                xs = x0 + (np.arange(T) + 0.5) / T * (x1 - x0)
                ys = py0 + (np.clip(v, lo, hi) - lo) / (hi - lo) * (py1 - py0)
                ax.plot(xs[: t + 1], ys[: t + 1], color=col, lw=1.8, zorder=4, solid_capstyle="round")
                ax.plot(xs, ys, color=col, lw=0.8, alpha=0.25, zorder=3)
                ax.scatter([xs[t]], [ys[t]], s=22, color=col, zorder=5, linewidths=0)
                val = v[t]
                txt = f"{s_['label'][:22]}  {val:.2f}" if s_["unit"] == "p" else f"{s_['label'][:22]}  {val:.1f}{s_['unit']}"
                ax.text(X0 + pad, ly, txt, fontsize=fs * 0.8, color=col, ha="left", va="top", zorder=6)
                ly -= fs * 0.8 / 72 / u * 1.6
        if frame_labels is not None and t < len(frame_labels) and frame_labels[t]:
            ax.text(x1, py1 + 0.05, frame_labels[t], fontsize=fs * 0.85, color=th["text"], ha="right", va="bottom",
                    zorder=6)
        ax.text(X0 + pad, Y0 + 0.28, f"frame {t + 1}/{T}", fontsize=fs * 0.66, color=th["faint"], ha="left",
                va="bottom", zorder=6)

    return draw_footer


def animate_inputs(model, inputs, output: str = "input_sweep.mp4", *, fps: int = 8, frame_labels=None,
                   title: Optional[str] = None, subtitle: Optional[str] = None, style: str = "cinematic",
                   figsize=(16, 9), dpi: int = 120, hold_last: int = 0, footer_height: float = 2.2,
                   progress: bool = True, **kw) -> str:
    """Render a movie of the activation flow over a sequence of inputs.

    ``inputs``: a tensor ``[T, ...]`` (one frame per leading index), or any
    iterable of per-frame inputs (tensors or dicts for multi-input models).
    """
    frames = _split_frames(inputs)
    T = len(frames)
    if T == 0:
        raise ValueError("no frames")
    kw = dict(kw)
    kw.update(style=style, figsize=figsize, dpi=dpi)
    # ---- pass 0: choose stages on the first frame
    r0 = trace_model(model, frames[0], explain=False, **kw)
    sel = _stage_selectors(r0)
    strategy = r0.config.channel_strategy
    kw_fixed = dict(kw, layers=sel)
    for k in ("channel_strategy", "volume_mode", "max_channels"):
        kw_fixed.setdefault(k, getattr(r0.config, k))
    if "theme" not in kw and style == "cinematic":
        kw_fixed["theme"] = "black"
    # ---- pass A: activity ranking over all frames (+ PCA basis from the middle frame)
    scores: Dict[str, np.ndarray] = {}
    basis: Dict[str, Any] = {}
    mid = T // 2
    for i, f in enumerate(frames):
        r = trace_model(model, f, explain=False, **kw_fixed)
        for st in r.graph.stages.values():
            sm = st.summary
            if sm is None or st.kind == "input" or not sm.channel_scores:
                continue
            sc = sm.channel_scores.get(strategy if strategy in sm.channel_scores else "energy")
            if sc is None:
                continue
            sc = np.nan_to_num(sc / (np.nanmax(sc) + 1e-12))
            scores[st.key] = scores.get(st.key, 0) + sc
            if i == mid and sm.pca_loadings is not None and sm.pca_mean is not None:
                basis[st.key] = (sm.pca_loadings, sm.pca_mean)
        if progress:
            print(f"\r  ranking channels  {i + 1}/{T}", end="", flush=True)
    force = {}
    for key, sc in scores.items():
        st0 = r0.graph.stages.get(key)
        C = sc.shape[0]
        n = n_sheets(C) if st0 is None or st0.summary is None or st0.summary.kind != "volume3d" else 5
        force[key] = [int(i) for i in np.argsort(-sc)[:n]]
    # ---- pass B: capture with fixed channels & colour basis (+ explanations)
    results: List[FlowResult] = []
    for i, f in enumerate(frames):
        r = trace_model(model, f, force_channels=force, force_pca=basis, **kw_fixed)
        results.append(r)
        if progress:
            print(f"\r  capturing frames  {i + 1}/{T}   ", end="", flush=True)
    # ---- fixed normalisation ranges over all frames
    ranges: Dict[str, List[Tuple[float, float]]] = {}
    pca_ranges: Dict[str, List[Tuple[float, float]]] = {}
    vec_ranges: Dict[str, Tuple[float, float]] = {}
    for key in results[0].graph.stages:
        sms = [r.graph.stages[key].summary for r in results if key in r.graph.stages]
        sms = [s for s in sms if s is not None]
        if not sms:
            continue
        s0 = sms[0]
        if s0.maps is not None and s0.selections.get("_forced"):
            ids = [i for i in s0.selections["_forced"] if i in s0.channel_ids]
            rng = []
            for cid in ids:
                vals = np.concatenate([s.maps[s.channel_ids.index(cid)].reshape(-1) for s in sms if cid in s.channel_ids])
                lo, hi = np.percentile(vals, (1, 99.5))
                rng.append((float(lo), float(hi)))
            ranges[key] = rng
        if s0.pca_maps is not None and s0.pca_maps.shape[0] >= 3:
            pr = []
            for c in range(3):
                vals = np.concatenate([s.pca_maps[c].reshape(-1) for s in sms if s.pca_maps is not None])
                pr.append(tuple(np.percentile(vals, (1, 99))))
            pca_ranges[key] = pr
        if s0.vector is not None:
            vals = np.concatenate([s.vector.reshape(-1) for s in sms if s.vector is not None])
            vec_ranges[key] = tuple(np.percentile(vals, (1, 99.5)))
        elif s0.spatial and max(s0.spatial) <= 1 and "mean" in s0.channel_scores:
            vals = np.concatenate([s.channel_scores["mean"] for s in sms])
            vec_ranges[key] = tuple(np.percentile(vals, (1, 99.5)))
    series = _series(results)
    thumbs = [_thumb(r) for r in results]
    # ---- render
    ext = os.path.splitext(output)[1].lower()
    writer = _Writer(output, fps)
    try:
        for t, r in enumerate(results):
            fs_ = FrameState(ranges=ranges, vec_ranges=vec_ranges, pca_ranges=pca_ranges,
                             footer_height=footer_height,
                             draw_footer=_make_footer(series, thumbs, T, t, frame_labels),
                             title=title, subtitle=subtitle)
            ff = draw(r, frame=fs_)
            writer.add(_fig_to_array(ff.fig, dpi))
            import matplotlib.pyplot as plt

            plt.close(ff.fig)
            if progress:
                print(f"\r  rendering frames  {t + 1}/{T}   ", end="", flush=True)
        for _ in range(hold_last):
            writer.repeat_last()
    finally:
        path = writer.close()
    if progress:
        print(f"\r  wrote {path}                ")
    return path


def _fig_to_array(fig, dpi) -> np.ndarray:
    fig.set_dpi(dpi)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[..., :3].copy()


class _Writer:
    """MP4 via imageio-ffmpeg / system ffmpeg when available, otherwise animated GIF (Pillow)."""

    def __init__(self, path: str, fps: int):
        self.path, self.fps = path, fps
        self.frames: List[np.ndarray] = []
        self._w = None
        self.mode = "gif"
        if path.lower().endswith(".mp4"):
            try:
                import imageio.v2 as imageio

                self._w = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p",
                                             macro_block_size=16)
                self.mode = "mp4"
            except Exception as e:
                warnings.warn(f"neural_flow: MP4 writer unavailable ({e}); writing GIF instead")
                self.path = os.path.splitext(path)[0] + ".gif"

    def add(self, arr: np.ndarray):
        if self.mode == "mp4":
            h, w = arr.shape[:2]
            arr = arr[: h - h % 16, : w - w % 16]
            self._w.append_data(arr)
            self._last = arr
        else:
            self.frames.append(arr)

    def repeat_last(self):
        if self.mode == "mp4":
            self._w.append_data(self._last)
        elif self.frames:
            self.frames.append(self.frames[-1])

    def close(self) -> str:
        if self.mode == "mp4":
            self._w.close()
        else:
            from PIL import Image

            ims = [Image.fromarray(f) for f in self.frames]
            if ims:
                w = 1200
                ims = [im.resize((w, int(im.height * w / im.width)), Image.LANCZOS) if im.width > w else im for im in ims]
                ims = [im.convert("P", palette=Image.ADAPTIVE, colors=255) for im in ims]
                ims[0].save(self.path, save_all=True, append_images=ims[1:], duration=int(1000 / self.fps), loop=0,
                            optimize=False)
        return self.path
