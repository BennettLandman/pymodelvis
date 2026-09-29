"""Tensor interpretation and memory-aware reduction.

``summarize_tensor`` turns an arbitrary activation into a compact
:class:`TensorSummary`.  Everything expensive is computed *on the tensor's own
device*, channel-chunk by channel-chunk, so a ``[1, 1024, 128, 128, 128]``
activation is never copied wholesale to the CPU: only per-channel statistics,
a handful of representative channel maps at reduced resolution, PCA maps and
spatial energy maps leave the device.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import FlowConfig

STRATEGIES = ("energy", "variance", "spread", "mean_abs", "even", "pca")
KINDS = ("image2d", "volume3d", "tokens", "seq1d", "vector", "attention", "scalar", "generic")

_CHUNK_ELEMS = 16_000_000


@dataclass
class TensorSummary:
    shape: Tuple[int, ...]                  # original full shape (incl. batch)
    dtype: str
    kind: str
    role: str                               # input | activation | output
    stats: Dict[str, float] = field(default_factory=dict)
    channels: int = 0                       # channel / feature count
    spatial: Tuple[int, ...] = ()           # original spatial dims (H,W) / (D,H,W) / patch grid
    reduced_spatial: Tuple[int, ...] = ()
    maps: Optional[np.ndarray] = None       # [K, *reduced_spatial] retained channel maps
    channel_ids: List[int] = field(default_factory=list)
    channel_scores: Dict[str, np.ndarray] = field(default_factory=dict)
    selections: Dict[str, List[int]] = field(default_factory=dict)  # strategy -> original channel ids
    pca_maps: Optional[np.ndarray] = None   # [3, *reduced_spatial]
    pca_var: Optional[np.ndarray] = None
    pca_loadings: Optional[np.ndarray] = None   # [3, C] (to reuse a fixed colour basis across frames)
    pca_mean: Optional[np.ndarray] = None
    mean_abs_map: Optional[np.ndarray] = None
    energy_map: Optional[np.ndarray] = None
    image: Optional[np.ndarray] = None      # raw input image / volume [C, *spatial] (inputs only)
    vector: Optional[np.ndarray] = None     # [F] for vectors
    token_matrix: Optional[np.ndarray] = None  # [N', F'] reduced token x feature matrix
    cls: Optional[np.ndarray] = None        # [k, F] special tokens
    n_special: int = 0
    token_norm_map: Optional[np.ndarray] = None
    attention: Optional[np.ndarray] = None  # [h', N, N]
    channels_last: bool = False
    reduced: bool = False
    notes: List[str] = field(default_factory=list)

    # convenience ------------------------------------------------------------
    @property
    def is_spatial(self) -> bool:
        if self.spatial and max(self.spatial) <= 1:
            return False
        return self.kind in ("image2d", "volume3d") or (self.kind == "tokens" and self.maps is not None)

    @property
    def resolution(self) -> float:
        """Geometric-mean spatial extent (0 for non-spatial tensors)."""
        if not self.spatial:
            return 0.0
        return float(np.prod(self.spatial)) ** (1.0 / len(self.spatial))

    def channel_maps(self, strategy: str, n: int) -> Tuple[np.ndarray, List[int]]:
        """Return up to ``n`` retained maps chosen by ``strategy`` (+ their channel ids)."""
        if self.maps is None:
            return np.zeros((0,)), []
        if strategy == "pca" and self.pca_maps is not None and n <= 3 and strategy not in self.selections:
            return self.pca_maps[:n], [-1] * min(n, 3)
        ids = self.selections.get(strategy) or self.selections.get("energy") or self.channel_ids
        ids = [i for i in ids if i in self.channel_ids][:n]
        pos = [self.channel_ids.index(i) for i in ids]
        return self.maps[pos], ids

    def shape_label(self) -> str:
        s = self.shape[1:] if len(self.shape) > 1 else self.shape
        if self.kind == "tokens":
            n, f = self.shape[-2], self.shape[-1]
            return f"{n} tokens × {f}"
        if self.kind in ("image2d", "volume3d"):
            if self.channels_last:
                s = (s[-1],) + tuple(s[:-1])
            return " × ".join(str(v) for v in s)
        if self.kind == "vector":
            return f"{self.shape[-1]}-d"
        return " × ".join(str(v) for v in s) if s else "scalar"


# ---------------------------------------------------------------------------
# kind inference
# ---------------------------------------------------------------------------

_SEQ1D_TYPES = ("Conv1d", "MaxPool1d", "AvgPool1d", "BatchNorm1d", "AdaptiveAvgPool1d", "ConvTranspose1d")


def looks_channels_last(r: Sequence[int], ctx: Dict[str, Any], cfg: FlowConfig) -> bool:
    if cfg.channels_last is not None:
        return bool(cfg.channels_last)
    if len(r) not in (3, 4):
        return False
    tn = ctx.get("type_name", "")
    if any(s in tn for s in ("Permute", "PatchMerging", "Swin", "LayerNorm")) and r[-1] > r[0]:
        return True
    sp, c = r[:-1], r[-1]
    return len(set(sp)) == 1 and c != sp[0] and c >= 3 and r[0] != c and (c > sp[0] or c <= 4)


def infer_kind(t: torch.Tensor, role: str, ctx: Dict[str, Any], cfg: FlowConfig) -> Tuple[str, torch.Tensor, bool]:
    """Return (kind, batch-item tensor in canonical layout, channels_last flag)."""
    x = t.detach()
    b = cfg.batch_index
    bsz = ctx.get("batch_size")
    if x.dim() == 0:
        return "scalar", x, False
    if x.dim() == 1:
        # [B] (one value per sample) vs. an un-batched feature vector [F]
        if bsz is not None and x.shape[0] == bsz and role != "input":
            return "scalar", x[min(b, x.shape[0] - 1)], False
        return "vector", x, False
    # sequence-first transformer outputs [N, B, F]
    if x.dim() == 3 and bsz is not None and x.shape[0] != bsz and x.shape[1] == bsz:
        x = x.transpose(0, 1)
    if x.shape[0] == 0:
        return "generic", x.reshape(-1), False
    x = x[min(b, x.shape[0] - 1)]
    r = tuple(x.shape)
    tn = ctx.get("type_name", "")
    name = (ctx.get("name") or "").lower()
    if len(r) == 0:
        return "scalar", x, False
    if len(r) == 1:
        return "vector", x, False
    if len(r) == 2:
        if any(tn.startswith(s) for s in _SEQ1D_TYPES):
            return "seq1d", x, False
        return "tokens", x, False
    if len(r) == 3:
        if r[1] == r[2] and ("attn" in name or "attention" in name) and r[0] <= 64 and r[1] >= 4:
            return "attention", x, False
        cl = looks_channels_last(r, ctx, cfg)
        if cl:
            x = x.permute(2, 0, 1)
        return "image2d", x, cl
    if len(r) == 4:
        cl = cfg.channels_last is True or (cfg.channels_last is None and looks_channels_last(r, ctx, cfg) and r[-1] > 8)
        if cl:
            x = x.permute(3, 0, 1, 2)
        return "volume3d", x, cl
    return "generic", x, False


# ---------------------------------------------------------------------------
# reductions (all on-device, chunked)
# ---------------------------------------------------------------------------


def _robust_stats(x: torch.Tensor, pct: Tuple[float, float]) -> Dict[str, float]:
    xf = x.reshape(-1)
    n = xf.numel()
    if n == 0:
        return {}
    step = max(1, n // 2_000_000)
    s = xf[::step].float()
    s = s[torch.isfinite(s)]
    if s.numel() == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "p_lo": 0.0, "p_hi": 0.0, "frac_neg": 0.0}
    q = torch.quantile(s[: 16_000_000], torch.tensor([pct[0] / 100, pct[1] / 100, 0.5], device=s.device))
    mn = float(xf.min().float()) if xf.is_floating_point() else float(xf.min())
    mx = float(xf.max().float()) if xf.is_floating_point() else float(xf.max())
    return {
        "min": mn, "max": mx,
        "mean": float(s.mean()), "std": float(s.std()) if s.numel() > 1 else 0.0,
        "p_lo": float(q[0]), "p_hi": float(q[1]), "median": float(q[2]),
        "frac_neg": float((s < 0).float().mean()),
        "frac_zero": float((s == 0).float().mean()),
    }


def _chunks(C: int, per_channel: int):
    step = max(1, _CHUNK_ELEMS // max(per_channel, 1))
    for c0 in range(0, C, step):
        yield c0, min(C, c0 + step)


def _pool(x: torch.Tensor, size: Tuple[int, ...]) -> torch.Tensor:
    """Adaptive average pool of [C, *S] to size (no-op when already that size)."""
    if tuple(x.shape[1:]) == tuple(size):
        return x
    if len(size) == 1:
        return F.adaptive_avg_pool1d(x[None], size)[0]
    if len(size) == 2:
        return F.adaptive_avg_pool2d(x[None], size)[0]
    return F.adaptive_avg_pool3d(x[None], size)[0]


def _target_size(spatial: Sequence[int], cap: int) -> Tuple[int, ...]:
    m = max(spatial)
    if m <= cap:
        return tuple(int(s) for s in spatial)
    f = cap / m
    return tuple(max(1, int(round(s * f))) for s in spatial)


def _channel_scores(x: torch.Tensor) -> Dict[str, torch.Tensor]:
    C = x.shape[0]
    P = int(np.prod(x.shape[1:]))
    mean = torch.empty(C, device=x.device)
    var = torch.empty(C, device=x.device)
    mabs = torch.empty(C, device=x.device)
    energy = torch.empty(C, device=x.device)
    spread = torch.empty(C, device=x.device)
    step = max(1, P // 4096)
    for c0, c1 in _chunks(C, P):
        xc = x[c0:c1].reshape(c1 - c0, -1).float()
        mean[c0:c1] = xc.mean(1)
        var[c0:c1] = xc.var(1, unbiased=False)
        mabs[c0:c1] = xc.abs().mean(1)
        energy[c0:c1] = (xc * xc).mean(1)
        # robust spread (p90 - p10): structure across positions, insensitive to a few outlier positions
        q = torch.quantile(xc[:, ::step], torch.tensor([0.1, 0.9], device=xc.device), dim=1)
        spread[c0:c1] = q[1] - q[0]
    return {"mean": mean, "variance": var, "mean_abs": mabs, "energy": energy, "spread": spread}


def _topk(score: torch.Tensor, k: int) -> List[int]:
    k = min(k, score.numel())
    # stable deterministic ordering: sort by (-score, index)
    s = score.detach().float().cpu().numpy()
    order = np.lexsort((np.arange(len(s)), -np.nan_to_num(s, nan=-np.inf)))
    return [int(i) for i in order[:k]]


def _spatial_summary(x: torch.Tensor, cfg: FlowConfig, budget: int, cap: int, summ: TensorSummary,
                     keep_all_small: bool = False, force: Optional[Sequence[int]] = None,
                     basis: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> None:
    """Fill channel statistics, selections, retained maps, PCA and energy maps. x: [C, *S]."""
    C = x.shape[0]
    S = tuple(x.shape[1:])
    summ.channels = C
    summ.spatial = S
    scores = _channel_scores(x)
    summ.channel_scores = {k: v.float().cpu().numpy() for k, v in scores.items()}
    n = max(1, cfg.max_channels)
    sel = {
        "energy": _topk(scores["energy"], n),
        "variance": _topk(scores["variance"], n),
        "mean_abs": _topk(scores["mean_abs"], n),
        "spread": _topk(scores["spread"], n),
        "even": [int(round(v)) for v in np.linspace(0, C - 1, min(n, C))],
    }
    # --- PCA over channels at a small resolution (spatial positions are observations)
    pca_size = _target_size(S, 32 if len(S) <= 2 else 16)
    pooled_small = torch.empty((C,) + pca_size, device=x.device)
    P = int(np.prod(S))
    for c0, c1 in _chunks(C, P):
        pooled_small[c0:c1] = _pool(x[c0:c1].float(), pca_size)
    Z = pooled_small.reshape(C, -1)
    mu = Z.mean(1, keepdim=True)
    Zc = (Z - mu)
    loadings = None
    if basis is not None and basis[0].shape[1] == C:
        # fixed PCA basis (movies): identical colours for identical features across frames
        loadings = torch.as_tensor(basis[0], dtype=torch.float32, device=x.device)
        mu = torch.as_tensor(basis[1], dtype=torch.float32, device=x.device).reshape(C, 1)
        sel["pca"] = sel["variance"]
    elif C >= 2 and Zc.shape[1] >= 2:
        try:
            U, Sv, Vh = torch.linalg.svd(Zc.T.double().cpu(), full_matrices=False)
            k = min(3, Vh.shape[0])
            loadings = Vh[:k].float()                      # [k, C]
            signs = torch.sign(loadings.gather(1, loadings.abs().argmax(1, keepdim=True)))
            loadings = loadings * signs
            ev = (Sv[:k] ** 2) / max(float((Sv ** 2).sum()), 1e-12)
            summ.pca_var = ev.float().numpy()
            lev = (loadings ** 2 * ev[:, None].float()).sum(0)
            sel["pca"] = _topk(lev, n)
            loadings = loadings.to(x.device)
        except Exception:  # pragma: no cover
            loadings = None
    if "pca" not in sel:
        sel["pca"] = sel["variance"]
    if loadings is not None:
        summ.pca_loadings = loadings.detach().cpu().numpy()
        summ.pca_mean = mu.reshape(-1).detach().cpu().numpy()
    if force:
        sel["_forced"] = [int(i) for i in force if 0 <= int(i) < C]
    summ.selections = sel

    # --- reduced resolution under budget
    retained = sorted(set(i for v in sel.values() for i in v)) if not sel.get("_forced") else sorted(set(sel["_forced"]))
    if keep_all_small and C <= 4:
        retained = list(range(C))
    size = _target_size(S, cap)
    while len(retained) * int(np.prod(size)) * 4 + 6 * int(np.prod(size)) * 4 > budget and max(size) > 4:
        size = tuple(max(1, int(s * 0.8)) for s in size)
    summ.reduced_spatial = size
    if size != S:
        summ.reduced = True
        summ.notes.append(f"spatial {'×'.join(map(str, S))} → {'×'.join(map(str, size))}")
    if len(retained) < C:
        summ.notes.append(f"{len(retained)}/{C} channels retained")

    idx = torch.tensor(retained, device=x.device, dtype=torch.long)
    summ.maps = _pool(x.index_select(0, idx).float(), size).cpu().numpy().astype(np.float32)
    summ.channel_ids = retained

    # --- all-channel spatial summaries (chunked accumulation)
    mabs = torch.zeros(size, device=x.device)
    energy = torch.zeros(size, device=x.device)
    pmaps = torch.zeros((loadings.shape[0],) + size, device=x.device) if loadings is not None else None
    for c0, c1 in _chunks(C, P):
        xc = _pool(x[c0:c1].float(), size)
        mabs += xc.abs().sum(0)
        energy += (xc * xc).sum(0)
        if pmaps is not None:
            xcc = xc - mu[c0:c1].reshape((-1,) + (1,) * len(size))
            pmaps += torch.einsum("kc,c...->k...", loadings[:, c0:c1], xcc)
    summ.mean_abs_map = (mabs / C).cpu().numpy()
    summ.energy_map = (energy / C).cpu().numpy()
    if pmaps is not None:
        summ.pca_maps = pmaps[: loadings.shape[0]].cpu().numpy()


def infer_patch_grid(n_tokens: int, cfg: FlowConfig, aspect: Optional[float] = None) -> Tuple[int, Optional[Tuple[int, int]]]:
    """Guess (#special tokens, (rows, cols)) for a token sequence."""
    if cfg.patch_grid is not None:
        gh, gw = cfg.patch_grid
        k = cfg.cls_tokens if cfg.cls_tokens is not None else max(0, n_tokens - gh * gw)
        if gh * gw + k == n_tokens:
            return k, (gh, gw)
    primary = [cfg.cls_tokens] if cfg.cls_tokens is not None else [0, 1]
    secondary = [] if cfg.cls_tokens is not None else [2, 4, 5]   # distillation / register tokens
    best = None
    for ks, allow_rect in ((primary, False), (primary, True), (secondary, False)):
        for k in ks:
            m = n_tokens - k
            if m < 4:
                continue
            r = int(round(math.sqrt(m)))
            if r * r == m and (aspect is None or abs(math.log(max(aspect, 1e-6))) < 0.08):
                return k, (r, r)
            if allow_rect and aspect:
                for gh in range(2, m + 1):
                    if m % gh == 0:
                        gw = m // gh
                        err = abs(math.log((gh / gw) / aspect))
                        if err < 0.08 and (best is None or err < best[0]):
                            best = (err, k, (gh, gw))
        if best:
            return best[1], best[2]
    if best:
        return best[1], best[2]
    return (cfg.cls_tokens or 0), None


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def summarize_tensor(t: torch.Tensor, role: str, budget: int, cfg: FlowConfig,
                     ctx: Optional[Dict[str, Any]] = None) -> TensorSummary:
    """Reduce a tensor to a :class:`TensorSummary` within ``budget`` bytes (approx.)."""
    ctx = dict(ctx or {})
    with torch.no_grad():
        return _summarize(t, role, budget, cfg, ctx)


def _summarize(t, role, budget, cfg, ctx):
    shape = tuple(int(s) for s in t.shape)
    summ = TensorSummary(shape=shape, dtype=str(t.dtype).replace("torch.", ""), kind="generic", role=role)
    if t.numel() == 0:
        return summ
    if t.is_complex():
        t = t.abs()
    kind, x, cl = infer_kind(t, role, ctx, cfg)
    x = x.float() if not x.is_floating_point() else x
    summ.kind, summ.channels_last = kind, cl
    summ.stats = _robust_stats(x, cfg.percentiles)
    cap2, cap3 = (cfg.max_input_2d, cfg.max_input_3d) if role == "input" else (cfg.max_spatial_2d, cfg.max_spatial_3d)
    if role == "output":
        cap2 = max(cap2, 160)

    if kind in ("image2d", "volume3d"):
        cap = cap2 if kind == "image2d" else cap3
        if role == "input" and x.shape[0] <= 4:
            size = _target_size(tuple(x.shape[1:]), cap)
            summ.image = _pool(x.float(), size).cpu().numpy()
            summ.channels = x.shape[0]
            summ.spatial = tuple(x.shape[1:])
            summ.reduced_spatial = size
            if size != tuple(x.shape[1:]):
                summ.reduced = True
                summ.notes.append(f"input {'×'.join(map(str, x.shape[1:]))} → {'×'.join(map(str, size))}")
            # still compute energy/mean maps for slice selection
            summ.maps = summ.image.astype(np.float32)
            summ.channel_ids = list(range(x.shape[0]))
            summ.selections = {s: list(range(x.shape[0])) for s in STRATEGIES}
            summ.mean_abs_map = np.abs(summ.image).mean(0)
            summ.energy_map = (summ.image ** 2).mean(0)
            return summ
        _spatial_summary(x, cfg, budget, cap, summ, keep_all_small=(role == "output"),
                         force=ctx.get("force"), basis=ctx.get("pca_basis"))
        return summ

    if kind == "tokens":
        N, Fdim = x.shape
        if ctx.get("input_ndim") == 5 and cfg.patch_grid is None:
            # 3-D token grids (volumetric transformers): N - k = m³ → treat as a feature volume
            for k in ([cfg.cls_tokens] if cfg.cls_tokens is not None else [0, 1]):
                m = int(round((N - k) ** (1 / 3))) if N - k > 7 else 0
                if m >= 2 and m ** 3 == N - k:
                    vol = x[k:].float().T.reshape(Fdim, m, m, m)
                    summ.kind = "volume3d"
                    summ.n_special = k
                    summ.notes.append(f"{N} tokens → {m}³ token grid")
                    _spatial_summary(vol, cfg, budget, cap3, summ, force=ctx.get("force"), basis=ctx.get("pca_basis"))
                    return summ
        aspect = ctx.get("input_aspect")
        k, grid = infer_patch_grid(N, cfg, aspect)
        summ.n_special = k
        summ.channels = Fdim
        if k:
            summ.cls = x[:k].float().cpu().numpy()
        # token x feature heatmap (reduced)
        tm = x.float()
        tsize = (min(N, 256), min(Fdim, 256))
        summ.token_matrix = F.adaptive_avg_pool2d(tm[None, None], tsize)[0, 0].cpu().numpy()
        if tsize != (N, Fdim):
            summ.notes.append(f"token matrix {N}×{Fdim} → {tsize[0]}×{tsize[1]}")
        if grid is not None:
            gh, gw = grid
            patches = x[k:].float().T.reshape(Fdim, gh, gw)       # features as channels
            _spatial_summary(patches, cfg, budget, cap2, summ, force=ctx.get("force"), basis=ctx.get("pca_basis"))
            summ.spatial = (gh, gw)
            summ.token_norm_map = patches.norm(dim=0).cpu().numpy()
        else:
            scores = _channel_scores(x.float().T.contiguous())
            summ.channel_scores = {kk: v.cpu().numpy() for kk, v in scores.items()}
        return summ

    if kind == "seq1d":
        C, L = x.shape
        summ.channels = C
        summ.spatial = (L,)
        tsize = (min(C, 256), min(L, 512))
        summ.token_matrix = F.adaptive_avg_pool2d(x.float()[None, None], tsize)[0, 0].cpu().numpy()
        return summ

    if kind == "vector":
        v = x.float().reshape(-1)
        summ.channels = v.numel()
        if v.numel() > 8192:
            summ.vector = F.adaptive_avg_pool1d(v[None, None], 8192)[0, 0].cpu().numpy()
            summ.reduced = True
            summ.notes.append(f"vector {v.numel()} → 8192 bins")
        else:
            summ.vector = v.cpu().numpy()
        return summ

    if kind == "attention":
        h, n, m = x.shape
        size = (min(n, 256), min(m, 256))
        a = F.adaptive_avg_pool2d(x.float()[None], size)[0]
        summ.attention = a[:16].cpu().numpy()
        summ.channels = h
        return summ

    if kind == "scalar":
        summ.vector = x.float().reshape(1).cpu().numpy()
        return summ

    # generic: flatten + bin
    v = x.float().reshape(-1)
    summ.vector = F.adaptive_avg_pool1d(v[None, None], min(v.numel(), 4096))[0, 0].cpu().numpy() if v.numel() else v.cpu().numpy()
    summ.channels = v.numel()
    return summ
