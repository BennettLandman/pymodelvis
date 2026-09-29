"""Optional animation: stages light up in data-flow order (GIF via Pillow, MP4 via ffmpeg).

The static renderer is reused; each frame only changes artist alphas, so the
captured activations appear progressively:  input → stage 1 → … → head → prediction.
"""
from __future__ import annotations

import os
import shutil
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
from matplotlib.patches import FancyBboxPatch

from .api import draw, trace_model

GHOST = 0.07


def _alpha(a) -> float:
    v = a.get_alpha()
    return 1.0 if v is None else float(v)


def animate_model(model, inputs, output: str = "network_flow.gif", *, fps: int = 12, frames_per_stage: int = 6,
                  hold_frames: int = 18, config=None, **kw):
    """Render an animation of activations flowing through the network.

    ``output`` ending in ``.mp4`` uses ffmpeg when available (falls back to GIF);
    anything else is written as an animated GIF with Pillow.  Returns the path written.
    """
    from matplotlib import animation

    res = trace_model(model, inputs, config=config, **kw)
    ff = draw(res)
    lay = ff.layout
    # reveal column by column (parallel branches light up together)
    cols: Dict[int, List[str]] = {}
    for k in ff.stage_artists:
        b = lay.boxes.get(k)
        cols.setdefault(b.col if b is not None else 0, []).append(k)
    steps = [cols[c] for c in sorted(cols)]
    base = {id(a): _alpha(a) for arts in ff.stage_artists.values() for a in arts}
    for _, _, a in ff.edge_artists:
        base[id(a)] = _alpha(a)
    th_accent = "#c2410c" if res.config.theme == "light" else "#fb923c"
    glows = []
    for keys in steps:
        g = []
        for k in keys:
            b = lay.boxes[k]
            p = FancyBboxPatch((b.left - 0.05, b.bottom - 0.05), b.w + 0.1, b.h + 0.1,
                               boxstyle="round,pad=0.02,rounding_size=0.05", fill=False, lw=2.2,
                               edgecolor=th_accent, alpha=0.0, zorder=7)
            ff.ax.add_patch(p)
            g.append(p)
        glows.append(g)
    step_of = {k: i for i, keys in enumerate(steps) for k in keys}
    n_frames = len(steps) * frames_per_stage + hold_frames

    def update(f):
        cur = f / frames_per_stage           # fractional step being revealed
        for k, arts in ff.stage_artists.items():
            i = step_of[k]
            t = float(np.clip(cur - i + 1, 0, 1))   # 0 = hidden, 1 = fully lit
            for a in arts:
                a.set_alpha(base[id(a)] * (GHOST + (1 - GHOST) * t))
        for src, dst, a in ff.edge_artists:
            t = float(np.clip(cur - step_of.get(dst, 0) + 1, 0, 1))
            a.set_alpha(base[id(a)] * (GHOST + (1 - GHOST) * t))
        for i, g in enumerate(glows):
            d = cur - i
            v = float(np.clip(1 - abs(d - 0.5) * 1.6, 0, 1)) if f < len(steps) * frames_per_stage else 0.0
            for p in g:
                p.set_alpha(v)
        return []

    ext = os.path.splitext(output)[1].lower()
    if ext == ".mp4" and (shutil.which("ffmpeg") or _imageio_ffmpeg()):
        if not shutil.which("ffmpeg"):
            import matplotlib

            matplotlib.rcParams["animation.ffmpeg_path"] = _imageio_ffmpeg()
        writer = animation.FFMpegWriter(fps=fps, bitrate=6000, extra_args=["-pix_fmt", "yuv420p"])
    else:
        if ext == ".mp4":
            warnings.warn("neural_flow: ffmpeg not found; writing GIF instead")
            output = os.path.splitext(output)[0] + ".gif"
        writer = animation.PillowWriter(fps=fps)
    anim = animation.FuncAnimation(ff.fig, update, frames=n_frames, blit=False)
    dpi = min(res.config.dpi, 110)
    anim.save(output, writer=writer, dpi=dpi, savefig_kwargs={"facecolor": ff.fig.get_facecolor()})
    return output


def _imageio_ffmpeg() -> Optional[str]:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
