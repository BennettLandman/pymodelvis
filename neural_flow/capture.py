"""Model instrumentation.

Two forward passes are used, both under ``model.eval()`` and ``torch.no_grad()``:

1. **Metadata pass** – lightweight hooks on *every* module record names, types,
   shapes, execution order and the module call tree.  A ``TorchFunctionMode``
   (``DataflowTracer``) simultaneously records a runtime dataflow graph of every
   tensor operation, tagging each produced tensor with an integer node id.  No
   activations are retained.
2. **Capture pass** – hooks only on the *selected* stage modules compute compact,
   memory-bounded visualisation summaries directly on the tensor's device
   (channel statistics, top channels, downsampled maps, PCA maps …) and move only
   those summaries to ``capture_device``.

Every hook is removed in a ``finally`` block and the model's train/eval state is
restored, so the model is never permanently modified.
"""
from __future__ import annotations

import inspect
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

from .config import FlowConfig

TAG = "_nf_node"

# ---------------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------------


def flatten_tensors(obj: Any, prefix: str = "", _depth: int = 0) -> List[Tuple[str, torch.Tensor]]:
    """Return ``[(path, tensor), ...]`` for tensors nested in tuples/lists/dicts/dataclasses."""
    out: List[Tuple[str, torch.Tensor]] = []
    if _depth > 6:
        return out
    if isinstance(obj, torch.Tensor):
        out.append((prefix, obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out += flatten_tensors(v, f"{prefix}.{k}" if prefix else str(k), _depth + 1)
    elif isinstance(obj, (list, tuple)):
        fields_ = getattr(obj, "_fields", None)  # namedtuple
        for i, v in enumerate(obj):
            key = fields_[i] if fields_ else str(i)
            out += flatten_tensors(v, f"{prefix}.{key}" if prefix else key, _depth + 1)
    elif hasattr(obj, "__dataclass_fields__"):
        for k in obj.__dataclass_fields__:
            out += flatten_tensors(getattr(obj, k), f"{prefix}.{k}" if prefix else k, _depth + 1)
    elif hasattr(obj, "to_tuple") and callable(obj.to_tuple):  # HF ModelOutput
        try:
            for k, v in obj.items():
                out += flatten_tensors(v, f"{prefix}.{k}" if prefix else k, _depth + 1)
        except Exception:
            pass
    return out


def _node(t: torch.Tensor) -> Optional[int]:
    return getattr(t, TAG, None)


def _set_node(t: torch.Tensor, n: int) -> None:
    try:
        setattr(t, TAG, n)
    except Exception:
        pass


def model_device(model: nn.Module) -> torch.device:
    for p in model.parameters():
        return p.device
    for b in model.buffers():
        return b.device
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------


@dataclass
class PreparedInputs:
    args: tuple
    kwargs: dict
    named: List[Tuple[str, torch.Tensor]]  # (input name, tensor) in a stable order


def prepare_inputs(model: nn.Module, inputs: Any, cfg: FlowConfig) -> PreparedInputs:
    """Normalise user inputs into (args, kwargs) plus a named list of input tensors.

    Dict inputs are passed as keyword arguments when the forward signature accepts
    those names, otherwise as a single positional dict.
    """
    dev = model_device(model)

    def mv(t):
        if isinstance(t, torch.Tensor):
            return t.to(dev) if t.device != dev else t
        if isinstance(t, dict):
            return {k: mv(v) for k, v in t.items()}
        if isinstance(t, (list, tuple)):
            return type(t)(mv(v) for v in t) if not hasattr(t, "_fields") else type(t)(*[mv(v) for v in t])
        return t

    inputs = mv(inputs)
    if isinstance(inputs, torch.Tensor):
        args, kwargs = (inputs,), {}
    elif isinstance(inputs, dict):
        try:
            params = inspect.signature(model.forward).parameters
            accepts_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())
            if accepts_kw or all(k in params for k in inputs):
                args, kwargs = (), dict(inputs)
            else:
                args, kwargs = (inputs,), {}
        except (TypeError, ValueError):
            args, kwargs = (inputs,), {}
    elif isinstance(inputs, (list, tuple)):
        args, kwargs = tuple(inputs), {}
    else:
        raise TypeError(f"Unsupported input type {type(inputs)!r}")

    named = flatten_tensors((args, kwargs))
    # prettify names: ((x,), {}) -> "0.0" ; use kwargs keys / positional index
    pretty = []
    for path, t in named:
        parts = path.split(".")
        if parts[0] == "1":  # kwargs
            nm = ".".join(parts[1:])
        else:
            nm = ".".join(parts[1:]) if len(parts) > 1 else parts[0]
            if isinstance(inputs, dict) and args and isinstance(args[0], dict):
                nm = ".".join(parts[2:]) or nm
        pretty.append((nm, t))
    if cfg.input_names:
        pretty = [(cfg.input_names[i] if i < len(cfg.input_names) else n, t) for i, (n, t) in enumerate(pretty)]
    elif len(pretty) == 1:
        pretty = [("input", pretty[0][1])]
    else:
        pretty = [(("input" + n) if n.isdigit() else n, t) for n, t in pretty]
    return PreparedInputs(args=args, kwargs=kwargs, named=pretty)


def _alias_inputs(prep: PreparedInputs, tag_base: int) -> Tuple[tuple, dict, Dict[int, str]]:
    """Create aliases (views) of the input tensors so tags never touch user objects."""
    mapping: Dict[int, str] = {}
    names = iter(prep.named)
    counter = [tag_base]

    def alias(obj):
        if isinstance(obj, torch.Tensor):
            nm, _ = next(names)
            try:
                a = obj.detach().view_as(obj)
            except Exception:
                a = obj
            counter[0] -= 1
            _set_node(a, counter[0])
            mapping[counter[0]] = nm
            return a
        if isinstance(obj, dict):
            return {k: alias(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            vals = [alias(v) for v in obj]
            return type(obj)(*vals) if hasattr(obj, "_fields") else type(obj)(vals)
        return obj

    args = alias(prep.args)
    kwargs = alias(prep.kwargs)
    return args, kwargs, mapping


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------


@dataclass
class ModuleCall:
    id: int
    name: str
    type_name: str
    call_index: int
    start: int
    end: int = -1
    depth: int = 0
    parent: Optional[int] = None
    children: List[int] = field(default_factory=list)
    in_shapes: List[Tuple[int, ...]] = field(default_factory=list)
    out_shapes: List[Tuple[int, ...]] = field(default_factory=list)
    out_paths: List[str] = field(default_factory=list)
    in_nodes: List[int] = field(default_factory=list)
    out_nodes: List[int] = field(default_factory=list)
    n_params: int = 0
    is_leaf_module: bool = True

    @property
    def key(self) -> str:
        return self.name if self.call_index == 0 else f"{self.name}#{self.call_index}"

    @property
    def main_shape(self) -> Optional[Tuple[int, ...]]:
        return self.out_shapes[0] if self.out_shapes else None


@dataclass
class OpNode:
    id: int
    op: str
    parents: List[int]
    owner: Optional[int]  # ModuleCall id active when the op ran


@dataclass
class TraceResult:
    calls: Dict[int, ModuleCall]
    root: int
    ops: Dict[int, OpNode]
    input_nodes: Dict[int, str]                 # node id -> input name
    input_shapes: Dict[str, Tuple[int, ...]]
    outputs: List[Tuple[str, Tuple[int, ...], Optional[int]]]  # (path, shape, node)
    dataflow_ok: bool
    notes: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)      # set by adapters during stage selection

    def calls_in_order(self) -> List[ModuleCall]:
        return sorted(self.calls.values(), key=lambda c: c.start)


# ---------------------------------------------------------------------------
# dataflow tracer
# ---------------------------------------------------------------------------

_SKIP_FUNCS = {"__get__", "__set__", "size", "dim", "numel", "__len__", "__bool__", "item", "tolist",
               "data_ptr", "is_contiguous", "stride", "element_size", "__format__", "__repr__"}


class DataflowTracer(TorchFunctionMode):
    """Records every tensor op as a node whose parents are the producing ops of its inputs."""

    def __init__(self, stack: List[int]):
        super().__init__()
        self.stack = stack
        self.ops: Dict[int, OpNode] = {}
        self.counter = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        name = getattr(func, "__name__", "")
        if name in _SKIP_FUNCS:
            return out
        outs = flatten_tensors(out)
        if not outs:
            return out
        parents = []
        for _, t in flatten_tensors((args, kwargs)):
            n = _node(t)
            if n is not None and n not in parents:
                parents.append(n)
        self.counter += 1
        nid = self.counter
        self.ops[nid] = OpNode(nid, name, parents, self.stack[-1] if self.stack else None)
        for _, t in outs:
            _set_node(t, nid)
        return out


# ---------------------------------------------------------------------------
# pass 1: metadata
# ---------------------------------------------------------------------------


def _shape(t: torch.Tensor) -> Tuple[int, ...]:
    return tuple(int(s) for s in t.shape)


@contextmanager
def eval_no_grad(model: nn.Module):
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        model.train(was_training)


def _call_model(model, args, kwargs):
    return model(*args, **kwargs)


def trace_metadata(model: nn.Module, prep: PreparedInputs, cfg: FlowConfig) -> TraceResult:
    """Run the metadata pass (shapes, call tree, execution order, dataflow)."""
    names = {id(m): n for n, m in model.named_modules()}
    counts: Dict[str, int] = defaultdict(int)
    calls: Dict[int, ModuleCall] = {}
    stack: List[int] = []
    seq = [0]
    handles = []
    own_params = {}
    for n, m in model.named_modules():
        own_params[id(m)] = sum(p.numel() for p in m.parameters())

    def pre(module, args, kwargs):
        name = names.get(id(module), "?")
        ci = counts[name]
        counts[name] += 1
        seq[0] += 1
        cid = len(calls)
        c = ModuleCall(
            id=cid, name=name, type_name=type(module).__name__, call_index=ci, start=seq[0],
            depth=len(stack), parent=stack[-1] if stack else None,
            n_params=own_params.get(id(module), 0),
            is_leaf_module=len(module._modules) == 0,
        )
        ts = flatten_tensors((args, kwargs))
        c.in_shapes = [_shape(t) for _, t in ts]
        c.in_nodes = [n for n in (_node(t) for _, t in ts) if n is not None]
        if stack:
            calls[stack[-1]].children.append(cid)
        calls[cid] = c
        stack.append(cid)

    def post(module, args, kwargs, output):
        name = names.get(id(module), "?")
        # pop the matching call (robust to exceptions in nested modules)
        while stack:
            cid = stack.pop()
            if calls[cid].name == name:
                break
        else:
            return
        seq[0] += 1
        c = calls[cid]
        c.end = seq[0]
        ts = flatten_tensors(output)
        c.out_shapes = [_shape(t) for _, t in ts]
        c.out_paths = [p for p, _ in ts]
        c.out_nodes = [n for n in (_node(t) for _, t in ts) if n is not None]

    use_tracer = cfg.topology in ("auto", "runtime")
    notes: List[str] = []

    def run(with_tracer: bool):
        calls.clear(); counts.clear(); stack.clear(); seq[0] = 0
        args, kwargs, in_map = _alias_inputs(prep, 0) if with_tracer else (prep.args, prep.kwargs, {})
        tracer = DataflowTracer(stack) if with_tracer else None
        with eval_no_grad(model):
            with (tracer if tracer is not None else nullcontext()):
                out = _call_model(model, args, kwargs)
        return out, tracer, in_map

    try:
        for m in model.modules():
            handles.append(m.register_forward_pre_hook(pre, with_kwargs=True))
            handles.append(m.register_forward_hook(post, with_kwargs=True))
        tracer = None
        in_map: Dict[int, str] = {}
        if use_tracer:
            try:
                out, tracer, in_map = run(True)
            except Exception as e:  # pragma: no cover - depends on exotic models
                tracer = None
                out, _, _ = run(False)   # re-raises if the model itself fails
                warnings.warn(f"neural_flow: runtime dataflow tracing failed ({e!s}); using execution order only.")
                notes.append("runtime dataflow tracing failed; execution order used")
        else:
            out, _, _ = run(False)
    finally:
        for h in handles:
            h.remove()

    root = next((c.id for c in calls.values() if c.parent is None and c.name == ""), 0)
    outputs = [(p or "output", _shape(t), _node(t)) for p, t in flatten_tensors(out)]
    return TraceResult(
        calls=dict(calls), root=root,
        ops=tracer.ops if tracer else {},
        input_nodes=in_map,
        input_shapes={n: _shape(t) for n, t in prep.named},
        outputs=outputs,
        dataflow_ok=tracer is not None,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# pass 2: capture summaries for selected calls
# ---------------------------------------------------------------------------


@dataclass
class AttentionRecord:
    order: int
    stage_key: Optional[str]          # enclosing selected stage (if any)
    weights: torch.Tensor             # [heads, Nq, Nk] (batch item), float32 on capture device
    head_avg: torch.Tensor            # [Nq, Nk]


class AttentionTracer(TorchFunctionMode):
    """Intercepts attention primitives and records attention weights.

    * ``F.multi_head_attention_forward`` is re-invoked with ``need_weights=True``
      and per-head weights; the caller still receives what it asked for.
    * ``F.scaled_dot_product_attention`` weights are recomputed explicitly from
      q and k (only when the matrix is reasonably small).
    """

    MAX_TOKENS = 1024

    def __init__(self, stage_stack: List[str], cfg: FlowConfig, batch_index: int):
        super().__init__()
        self.stage_stack = stage_stack
        self.records: List[AttentionRecord] = []
        self.cfg = cfg
        self.b = batch_index
        self._mha_sig = inspect.signature(F.multi_head_attention_forward)

    def _record(self, w: torch.Tensor):
        # w: [B, h, Nq, Nk] or [B, Nq, Nk]
        if w.dim() == 3:
            w = w[:, None]
        b = min(self.b, w.shape[0] - 1)
        w = w[b].detach().float()
        if w.shape[-1] > self.MAX_TOKENS:
            return
        w = w.to(self.cfg.capture_device)
        self.records.append(AttentionRecord(len(self.records), self.stage_stack[-1] if self.stage_stack else None,
                                            w, w.mean(0)))

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is F.multi_head_attention_forward:
            try:
                ba = self._mha_sig.bind(*args, **kwargs)
                ba.apply_defaults()
                want, avg = ba.arguments["need_weights"], ba.arguments["average_attn_weights"]
                ba.arguments["need_weights"] = True
                ba.arguments["average_attn_weights"] = False
                out, w = func(*ba.args, **ba.kwargs)
                if w is not None:
                    self._record(w)
                if not want:
                    return out, None
                return out, (w.mean(1) if (avg and w is not None) else w)
            except Exception:
                return func(*args, **kwargs)
        if func is F.scaled_dot_product_attention:
            out = func(*args, **kwargs)
            try:
                q, k = args[0], args[1]
                if q.dim() == 4 and q.shape[-2] <= self.MAX_TOKENS and k.shape[-2] <= self.MAX_TOKENS:
                    b = min(self.b, q.shape[0] - 1)
                    qb, kb = q[b:b + 1].float(), k[b:b + 1].float()
                    scale = kwargs.get("scale") or qb.shape[-1] ** -0.5
                    logits = qb @ kb.transpose(-2, -1) * scale
                    mask = kwargs.get("attn_mask", args[3] if len(args) > 3 else None)
                    if mask is not None:
                        mask = mask[b:b + 1] if mask.dim() == 4 and mask.shape[0] > 1 else mask
                        logits = logits.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else logits + mask
                    if kwargs.get("is_causal", False):
                        n, m = logits.shape[-2:]
                        causal = torch.ones(n, m, dtype=torch.bool, device=logits.device).tril()
                        logits = logits.masked_fill(~causal, float("-inf"))
                    self._record(logits.softmax(-1))
            except Exception:
                pass
            return out
        return func(*args, **kwargs)


@dataclass
class CaptureResult:
    summaries: Dict[str, Any]                   # stage key -> TensorSummary (first/main output)
    extra_outputs: Dict[str, List[Any]]         # stage key -> summaries of additional outputs
    inputs: Dict[str, Any]                      # input name -> TensorSummary (hi-res)
    outputs: List[Tuple[str, Any]]              # model outputs (path, summary)
    raw_outputs: Any                            # small model outputs moved to CPU (for interpreters)
    attention: List[AttentionRecord]
    notes: List[str]


def capture_stages(model: nn.Module, prep: PreparedInputs, requests: Sequence[Tuple[ModuleCall, str, str]],
                   cfg: FlowConfig, summarize, budget_bytes: int) -> CaptureResult:
    """Second pass: capture compact summaries for the selected module calls.

    ``requests`` holds ``(call, "output" | "input", stage_key)`` tuples.
    ``summarize(tensor, role, budget_bytes, cfg, context)`` is supplied by
    :mod:`neural_flow.tensors` so reduction policy lives with tensor interpretation.
    """
    wanted: Dict[str, Dict[int, List[Tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for c, mode, key in requests:
        wanted[c.name][c.call_index].append((mode, key))
    modules = dict(model.named_modules())
    counts: Dict[str, int] = defaultdict(int)
    summaries: Dict[str, Any] = {}
    extras: Dict[str, List[Any]] = {}
    stage_stack: List[str] = []
    notes: List[str] = []
    handles = []
    per_stage = max(budget_bytes // max(len(requests) + 2, 1), 256 * 1024)
    base_ctx = {"batch_size": None, "input_aspect": None,
                "input_ndim": max((t.dim() for _, t in prep.named), default=0)}
    for _, t in prep.named:
        if t.dim() >= 1 and base_ctx["batch_size"] is None:
            base_ctx["batch_size"] = int(t.shape[0])
        if t.dim() == 4 and base_ctx["input_aspect"] is None:
            base_ctx["input_aspect"] = t.shape[2] / max(t.shape[3], 1)

    def make_pre(name):
        def pre(module, args):
            ci = counts[name]
            for mode, key in wanted[name].get(ci, []):
                if mode == "output":
                    stage_stack.append(key)
        return pre

    def make_post(name):
        def post(module, args, output):
            ci = counts[name]
            counts[name] += 1
            reqs = wanted[name].get(ci)
            if not reqs:
                return
            for mode, key in reqs:
                if mode == "output" and stage_stack and stage_stack[-1] == key:
                    stage_stack.pop()
                src = output if mode == "output" else args
                ts = flatten_tensors(src)
                ts = [(p, t) for p, t in ts if t.is_floating_point() or t.dtype in (torch.int64, torch.int32, torch.uint8, torch.bool)]
                if not ts:
                    continue
                ctx = dict(base_ctx, module=module, name=name, type_name=type(module).__name__, key=key)
                if cfg.force_channels and key in cfg.force_channels:
                    ctx["force"] = cfg.force_channels[key]
                if cfg.force_pca and key in cfg.force_pca:
                    ctx["pca_basis"] = cfg.force_pca[key]
                try:
                    summaries[key] = summarize(ts[0][1], "activation", per_stage, cfg, ctx)
                    if mode == "output" and len(ts) > 1:
                        extras[key] = [summarize(t, "activation", per_stage // 4, cfg, dict(ctx, path=p))
                                       for p, t in ts[1:4]]
                except Exception as e:  # never let a visualisation failure break the forward pass
                    notes.append(f"summary failed for {key}: {e!s}")
        return post

    try:
        for name in wanted:
            m = modules.get(name)
            if m is None:
                continue
            handles.append(m.register_forward_pre_hook(make_pre(name)))
            handles.append(m.register_forward_hook(make_post(name)))
        attn = AttentionTracer(stage_stack, cfg, cfg.batch_index) if cfg.capture_attention else None
        with eval_no_grad(model):
            with (attn if attn is not None else nullcontext()):
                out = _call_model(model, prep.args, prep.kwargs)
    finally:
        for h in handles:
            h.remove()

    def safe(t, role, ctx):
        try:
            return summarize(t, role, budget_bytes // 8, cfg, ctx)
        except Exception as e:  # pragma: no cover - defensive
            notes.append(f"summary failed for {role} {ctx.get('name')}: {e!s}")
            return None

    base_ctx["input_ndim"] = max((t.dim() for _, t in prep.named), default=0)
    with torch.no_grad():
        inputs = {nm: safe(t, "input", dict(base_ctx, name=nm)) for nm, t in prep.named}
        outs = [(p or "output", safe(t, "output", dict(base_ctx, name=p or "output"))) for p, t in flatten_tensors(out)]
    raw = _small_copy(out, cfg)
    return CaptureResult(summaries, extras, inputs, outs, raw, attn.records if attn else [], notes)


def _small_copy(obj, cfg: FlowConfig, limit: int = 4_000_000):
    """Batch item of the model output moved to CPU (skip anything very large)."""
    if isinstance(obj, torch.Tensor):
        t = obj.detach()
        if t.dim() > 0 and t.shape[0] > cfg.batch_index:
            t = t[cfg.batch_index:cfg.batch_index + 1]
        if t.numel() <= limit:
            return t.to("cpu")
        if t.dim() in (4, 5) and t.shape[1] > 1 and t.is_floating_point():
            # very large dense prediction (e.g. 133-class 3-D segmentation): keep argmax labels only
            return t.argmax(1, keepdim=True).to(torch.int32).to("cpu")
        return None
    if isinstance(obj, dict):
        return {k: _small_copy(v, cfg, limit) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        vals = [_small_copy(v, cfg, limit) for v in obj]
        return type(obj)(*vals) if hasattr(obj, "_fields") else type(obj)(vals)
    return obj


@contextmanager
def instrument(model: nn.Module, hook) -> Iterator[None]:
    """Temporarily attach ``hook(name, module, args, output)`` to every module."""
    handles = []
    try:
        for n, m in model.named_modules():
            handles.append(m.register_forward_hook(lambda mod, a, o, _n=n: hook(_n, mod, a, o)))
        yield
    finally:
        for h in handles:
            h.remove()
