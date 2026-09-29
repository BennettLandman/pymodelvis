"""Automatic (and explicit) stage selection.

The goal is a *representation-flow* view: ~5–12 module calls whose outputs
summarise how the input is transformed, instead of every ReLU / norm / reshape.

Algorithm (``layer_selection="auto"``) – a greedy *module-tree cut*:

1. Start from the meaningful children of the root module (trivial modules such
   as activations, dropout, normalisation, identity/flatten are ignored; thin
   wrappers with a single meaningful child are looked through).
2. Repeatedly *expand* the frontier unit whose children reveal the most new
   tensor shapes (new resolution / channel / rank = new information).  Units are
   also expanded, even without new shapes, while the frontier is smaller than
   ``target_stages``.
3. When an expansion would overflow the budget, the children are *sampled*:
   children that change the tensor shape are always kept, the first and last are
   kept, the rest are sampled evenly across depth (this is how 12 identical
   transformer blocks become 4–6 representative ones).
4. If the frontier is still larger than the target, parameter-free leaves
   (pooling, upsampling, padding …) that are immediately followed by a block at
   the same resolution are *absorbed* into that block; remaining overflow is
   removed by an importance score (heads, first stage and shape changes score
   high).
"""
from __future__ import annotations

import math
import re
import warnings
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from .capture import ModuleCall, TraceResult
from .config import FlowConfig

TRIVIAL_PATTERNS = [
    r"^(ReLU|ReLU6|LeakyReLU|PReLU|RReLU|ELU|SELU|CELU|GELU|SiLU|Swish|Mish|Hardswish|Hardsigmoid|Hardtanh|"
    r"Sigmoid|Tanh|Softplus|Softsign|Tanhshrink|Threshold|GLU|LogSigmoid|Softmax|Softmax2d|LogSoftmax|QuickGELU|NewGELUActivation|GELUActivation)$",
    r"Dropout", r"DropPath", r"StochasticDepth",
    r"Norm(\dd)?$", r"^(BatchNorm|LayerNorm|GroupNorm|InstanceNorm|LocalResponseNorm|RMSNorm|SyncBatchNorm|LazyBatchNorm)",
    r"^(Identity|Flatten|Unflatten|Permute|Reshape|View|Squeeze|Unsqueeze|Transpose|Contiguous|ZeroPad\dd|ConstantPad\dd|ReflectionPad\dd|ReplicationPad\dd|ChannelShuffle|PixelShuffle|PixelUnshuffle)$",
]
_TRIVIAL_RE = [re.compile(p) for p in TRIVIAL_PATTERNS]

BLOCK_HINTS = re.compile(r"(Block|Layer|Stage|Encoder|Decoder|Down|Up|Bottleneck|Residual|Res|Attention|Attn|"
                         r"Embed|Head|Fusion|Fuse|Merge|Transformer|Mixer|Conv|Pool|Proj|Stem)", re.I)
HEAD_NAME = re.compile(r"(head|classifier|fc$|logits|output|out$|pred|regress|seg)", re.I)
PARAM_FREE_LEAF = re.compile(r"(Pool|Upsample|Pad|Interpolate)", re.I)
CONNECTOR = re.compile(r"(PatchMerging|Downsample|DownSample|Upsample|Transition|ConvTranspose)", re.I)


def is_trivial(call: ModuleCall) -> bool:
    return any(r.search(call.type_name) for r in _TRIVIAL_RE)


def _rank_res(shape: Optional[Tuple[int, ...]]) -> Tuple:
    """Shape signature used for 'new information' detection (batch dropped, size-1 dims squeezed)."""
    if not shape:
        return ()
    s = tuple(v for v in shape[1:] if v != 1)
    return s


def _spatial_res(shape: Optional[Tuple[int, ...]]) -> Optional[Tuple[int, ...]]:
    if not shape or len(shape) < 4:
        return None
    return tuple(shape[2:])


class _Tree:
    def __init__(self, trace: TraceResult):
        self.t = trace
        self.calls = trace.calls

    def has_tensor_out(self, c: ModuleCall) -> bool:
        return bool(c.out_shapes) and any(len(s) >= 1 for s in c.out_shapes)

    def meaningful_children(self, c: ModuleCall, _depth: int = 0) -> List[ModuleCall]:
        kids = [self.calls[i] for i in c.children]
        kids = [k for k in kids if self.has_tensor_out(k) and not is_trivial(k)]
        kids.sort(key=lambda k: k.start)
        if len(kids) == 1 and _depth < 8:
            deeper = self.meaningful_children(kids[0], _depth + 1)
            if len(deeper) >= 2:
                return deeper
        return kids


def _homogeneous(kids: Sequence[ModuleCall]) -> bool:
    return len(kids) >= 3 and len({k.type_name for k in kids}) == 1


def _sample(kids: List[ModuleCall], allowed: int) -> List[ModuleCall]:
    """Keep shape-changing children + first/last, fill the rest evenly across depth."""
    if len(kids) <= allowed:
        return kids
    keep: Set[int] = {0, len(kids) - 1}
    for i in range(1, len(kids)):
        if _rank_res(kids[i].main_shape) != _rank_res(kids[i - 1].main_shape):
            keep.add(i)
    if len(keep) > allowed:
        # too many shape changes: keep evenly spaced ones among them
        ks = sorted(keep)
        pick = [ks[int(round(j))] for j in _linspace(0, len(ks) - 1, allowed)]
        return [kids[i] for i in sorted(set(pick))]
    remaining = allowed - len(keep)
    if remaining > 0:
        for j in _linspace(0, len(kids) - 1, remaining + 2)[1:-1]:
            i = int(round(j))
            while i in keep and i < len(kids) - 1:
                i += 1
            keep.add(i)
    return [kids[i] for i in sorted(keep)][:allowed] if len(keep) > allowed else [kids[i] for i in sorted(keep)]


def _linspace(a: float, b: float, n: int) -> List[float]:
    if n <= 1:
        return [a]
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def _auto_select(trace: TraceResult, cfg: FlowConfig) -> List[ModuleCall]:
    tree = _Tree(trace)
    root = trace.calls[trace.root]
    max_s = max(1, cfg.max_stages)
    target = cfg.resolved_target()
    frontier = tree.meaningful_children(root)
    if not frontier:
        return [root] if tree.has_tensor_out(root) else []
    if len(frontier) > target:
        frontier = _compress_runs(frontier, target)
    if len(frontier) > max_s:
        frontier = _prune(frontier, max_s, trace, target)
    blocked: Set[int] = set()

    for _ in range(200):
        shapes = {_rank_res(c.main_shape) for c in frontier}
        best = None
        for i, u in enumerate(frontier):
            if u.id in blocked:
                continue
            kids = tree.meaningful_children(u)
            if len(kids) < 2:
                continue
            gain = len({_rank_res(k.main_shape) for k in kids} - shapes)
            need = len(frontier) < target
            # residual-style blocks (same in/out shape) and members of repeated runs are natural
            # units: their internals (e.g. a transformer MLP's hidden width) are not new information
            same_io = bool(u.in_shapes) and u.in_shapes[0] == u.main_shape and u.type_name not in ("Sequential", "ModuleList", "ModuleDict")
            repeated = sum(1 for o in frontier if o.type_name == u.type_name and o.main_shape == u.main_shape) >= 2
            if (same_io or repeated) and not need:
                gain = 0
            if gain == 0 and not need:
                continue
            score = gain * 10 + (5 if need else 0) + math.log1p(u.n_params) * 0.1 - 0.01 * i
            if best is None or score > best[0]:
                best = (score, i, u, kids, gain)
        if best is None:
            break
        _, i, u, kids, gain = best
        limit = max_s if gain > 0 else target
        allowed = limit - (len(frontier) - 1)
        if len(kids) > allowed:
            if allowed < 2:
                blocked.add(u.id)
                continue
            kids = _sample(kids, allowed)
        frontier = frontier[:i] + kids + frontier[i + 1:]
    if len(frontier) > target:
        frontier = _compress_runs(frontier, target)
    if len(frontier) > target:
        frontier = _absorb(frontier, trace, target)
    if len(frontier) > max_s:
        frontier = _prune(frontier, max_s, trace, target)
    return frontier


def _compress_runs(frontier: List[ModuleCall], target: int) -> List[ModuleCall]:
    """Sample long runs of homogeneous blocks (same type & output shape) down toward ``target``."""
    fr = list(frontier)

    def runs(f):
        out, i = [], 0
        while i < len(f):
            j = i
            while j + 1 < len(f) and f[j + 1].type_name == f[i].type_name and \
                    _rank_res(f[j + 1].main_shape) == _rank_res(f[i].main_shape):
                j += 1
            if j - i + 1 >= 3:
                out.append((i, j))
            i = j + 1
        return out

    full = {id(c): c for c in fr}
    groups = {}
    for i, j in runs(fr):
        groups[(fr[i].id)] = fr[i:j + 1]
    while len(fr) > target:
        rs = runs(fr)
        if not rs:
            break
        i, j = max(rs, key=lambda r: (r[1] - r[0], -r[0]))
        run = fr[i:j + 1]
        orig = next((g for g in groups.values() if run[0] in g and run[-1] in g), run)
        n = len(run) - 1
        if n < 2:
            break
        idx = sorted({int(round(v)) for v in _linspace(0, len(orig) - 1, n)})
        fr = fr[:i] + [orig[k] for k in idx] + fr[j + 1:]
    return fr


def _absorb(frontier: List[ModuleCall], trace: TraceResult, target: int) -> List[ModuleCall]:
    """Drop parameter-free / connector leaves that precede a block at the same resolution."""
    out: List[ModuleCall] = []
    for i, c in enumerate(frontier):
        nxt = frontier[i + 1] if i + 1 < len(frontier) else None
        absorbable = (
            nxt is not None and (c.is_leaf_module or CONNECTOR.search(c.type_name)) and not nxt.is_leaf_module
            and _spatial_res(c.main_shape) is not None
            and _spatial_res(c.main_shape) == _spatial_res(nxt.main_shape)
            and (c.n_params == 0 or PARAM_FREE_LEAF.search(c.type_name) or CONNECTOR.search(c.type_name))
        )
        if absorbable and i != 0:
            continue
        out.append(c)
    return out if len(out) >= min(3, len(frontier)) else frontier


def _importance(frontier: List[ModuleCall], i: int, trace: TraceResult) -> float:
    c = frontier[i]
    s = 3.0 if not c.is_leaf_module else (2.0 if c.n_params else 1.0)
    if i == 0:
        s += 2
    if i == len(frontier) - 1 or HEAD_NAME.search(c.name.split(".")[-1] or ""):
        s += 3
    prev = frontier[i - 1] if i > 0 else None
    if prev is not None and _rank_res(prev.main_shape) != _rank_res(c.main_shape):
        s += 2
    if prev is not None and _spatial_res(prev.main_shape) != _spatial_res(c.main_shape):
        s += 1
    nxt = frontier[i + 1] if i + 1 < len(frontier) else None
    if nxt is not None and c.is_leaf_module and _spatial_res(c.main_shape) == _spatial_res(nxt.main_shape):
        s -= 1.5
    return s


def _prune(frontier: List[ModuleCall], n: int, trace: TraceResult, target: int) -> List[ModuleCall]:
    fr = list(frontier)
    fr = _absorb(fr, trace, target) if len(fr) > n else fr
    while len(fr) > n:
        scores = [(_importance(fr, i, trace), -i) for i in range(len(fr))]
        j = min(range(len(fr)), key=lambda i: scores[i])
        fr.pop(j)
    return fr


# ---------------------------------------------------------------------------
# explicit selectors
# ---------------------------------------------------------------------------


def _matches(sel: str, c: ModuleCall) -> bool:
    if sel.startswith("re:"):
        return re.search(sel[3:], c.name) is not None
    if sel.startswith("type:"):
        return re.fullmatch(sel[5:], c.type_name) is not None or c.type_name == sel[5:]
    if "#" in sel:
        return c.key == sel
    return c.name == sel


def select_by(selectors: Sequence[str], trace: TraceResult) -> List[ModuleCall]:
    calls = [c for c in trace.calls_in_order() if c.out_shapes and c.id != trace.root]
    chosen: List[ModuleCall] = []
    for sel in selectors:
        hit = [c for c in calls if _matches(sel, c)]
        if not hit:
            warnings.warn(f"neural_flow: layer selector {sel!r} matched no executed module")
        chosen += [c for c in hit if c not in chosen]
    chosen.sort(key=lambda c: c.start)
    return chosen


def select_stages(trace: TraceResult, cfg: FlowConfig) -> List[ModuleCall]:
    """Return the module calls to display, in execution order."""
    if cfg.layers:
        chosen = select_by(list(cfg.layers), trace)
    elif callable(cfg.layer_selection):
        chosen = [c for c in trace.calls_in_order()
                  if c.id != trace.root and c.out_shapes and cfg.layer_selection(c)]
    elif cfg.layer_selection == "all":
        chosen = [c for c in trace.calls_in_order()
                  if c.id != trace.root and c.out_shapes and c.is_leaf_module and not is_trivial(c)]
    elif cfg.layer_selection == "auto":
        chosen = _auto_select(trace, cfg)
    else:
        raise ValueError(f"layer_selection must be 'auto', 'all' or a callable, got {cfg.layer_selection!r}")
    if cfg.exclude:
        chosen = [c for c in chosen if not any(_matches(s, c) for s in cfg.exclude)]
    return sorted(chosen, key=lambda c: c.start)


def short_label(call: ModuleCall, cfg: FlowConfig) -> str:
    if cfg.labels:
        if call.key in cfg.labels:
            return cfg.labels[call.key]
        if call.name in cfg.labels:
            return cfg.labels[call.name]
    parts = call.name.split(".") if call.name else [call.type_name]
    if len(parts) >= 2 and parts[-1].isdigit():
        lab = f"{parts[-2]}.{parts[-1]}"
    else:
        lab = parts[-1]
    if call.call_index:
        lab += f" (×{call.call_index + 1})"
    return lab
