"""Stage-level topology inference.

Topology sources, in order of preference (``topology="auto"``):

1. **runtime dataflow** – the metadata pass records every tensor op executed
   (via ``TorchFunctionMode``) together with the module call that owned it.
   For each selected stage we walk backwards from its input tensors until we
   hit a tensor produced *inside another selected stage* (or a model input).
   This recovers skip connections, U-Net concatenations, residual paths,
   branching heads and multimodal fusion, and works with dynamic control flow.
2. **torch.fx** – ``symbolic_trace`` gives a static graph; used when requested
   (``topology="fx"``) or when runtime tracing was unavailable.  FX fails on
   data-dependent control flow, so failure falls back gracefully.
3. **execution order** – a simple chain.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch.nn as nn

from .capture import ModuleCall, TraceResult
from .config import FlowConfig


@dataclass
class Stage:
    key: str
    label: str
    kind: str                          # input | module | representation
    order: float
    call: Optional[ModuleCall] = None
    capture: str = "output"            # output | input (representation stages capture a head's input)
    shape: Optional[Tuple[int, ...]] = None
    summary: Any = None
    is_head: bool = False
    outputs: List[str] = field(default_factory=list)   # model outputs this stage produces
    concept: str = ""
    type_name: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    src: str
    dst: str
    kind: str = "main"                 # main | skip | merge
    features: int = 0


@dataclass
class OutputSpec:
    name: str
    shape: Tuple[int, ...]
    sources: List[str]
    view: Any = None


@dataclass
class StageGraph:
    stages: Dict[str, Stage]
    edges: List[Edge]
    outputs: List[OutputSpec]
    topology_source: str
    notes: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)     # adapter annotations (e.g. backbone roles)

    def preds(self, key: str) -> List[Edge]:
        return [e for e in self.edges if e.dst == key]

    def succs(self, key: str) -> List[Edge]:
        return [e for e in self.edges if e.src == key]

    def ordered(self) -> List[Stage]:
        return sorted(self.stages.values(), key=lambda s: s.order)

    def ancestors(self, key: str) -> Set[str]:
        seen: Set[str] = set()
        todo = [key]
        while todo:
            k = todo.pop()
            for e in self.preds(k):
                if e.src not in seen:
                    seen.add(e.src)
                    todo.append(e.src)
        return seen


# ---------------------------------------------------------------------------


def _stage_of_calls(trace: TraceResult, selected: Dict[int, str]) -> Dict[int, Optional[str]]:
    out: Dict[int, Optional[str]] = {}
    for cid in sorted(trace.calls):
        c = trace.calls[cid]
        if cid in selected:
            out[cid] = selected[cid]
        else:
            out[cid] = out.get(c.parent) if c.parent is not None else None
    return out


def _backtrack(start: Sequence[int], trace: TraceResult, node_stage: Dict[int, Optional[str]],
               exclude: Optional[str], input_key: Dict[int, str]) -> List[str]:
    found: List[str] = []
    seen: Set[int] = set()
    todo = list(start)
    while todo:
        n = todo.pop()
        if n in seen:
            continue
        seen.add(n)
        if n in input_key:
            if input_key[n] not in found:
                found.append(input_key[n])
            continue
        st = node_stage.get(n)
        if st is not None and st != exclude:
            if st not in found:
                found.append(st)
            continue
        op = trace.ops.get(n)
        if op is not None:
            todo.extend(op.parents)
    return found


def build_stage_graph(trace: TraceResult, calls: Sequence[ModuleCall], cfg: FlowConfig,
                      model: Optional[nn.Module] = None, label_fn=None) -> StageGraph:
    stages: Dict[str, Stage] = {}
    notes: List[str] = []
    # input pseudo-stages
    input_key: Dict[int, str] = {}
    for i, (nm, shp) in enumerate(trace.input_shapes.items()):
        k = f"input:{nm}"
        stages[k] = Stage(key=k, label=nm, kind="input", order=-100 + i, shape=shp)
    for nid, nm in trace.input_nodes.items():
        input_key[nid] = f"input:{nm}"

    selected: Dict[int, str] = {}
    for c in calls:
        k = c.key
        selected[c.id] = k
        stages[k] = Stage(key=k, label=label_fn(c) if label_fn else c.name, kind="module", order=c.start,
                          call=c, shape=c.main_shape, type_name=c.type_name)

    edges: List[Edge] = []
    out_specs: List[OutputSpec] = []
    source = "sequential"
    mode = cfg.topology

    if trace.dataflow_ok and trace.ops and mode in ("auto", "runtime"):
        source = "runtime"
        call_stage = _stage_of_calls(trace, selected)
        node_stage = {n: call_stage.get(op.owner) for n, op in trace.ops.items()}
        for c in calls:
            srcs = _backtrack(c.in_nodes, trace, node_stage, c.key, input_key)
            for s in srcs:
                edges.append(Edge(s, c.key))
        for path, shp, node in trace.outputs:
            srcs = _backtrack([node] if node is not None else [], trace, node_stage, None, input_key)
            out_specs.append(OutputSpec(path, shp, srcs))
    else:
        fx_edges = None
        if mode in ("fx", "auto") and model is not None:
            fx_edges = fx_stage_edges(model, calls, trace)
            if fx_edges is not None:
                source = "fx"
                edges = fx_edges[0]
                out_specs = [OutputSpec(p, s, fx_edges[1]) for p, s, _ in trace.outputs]
        if fx_edges is None:
            if mode == "fx":
                notes.append("fx tracing failed; runtime execution order used")
            source = "sequential"

    # sequential fallback / repair: any stage without predecessors hangs off the previous stage
    ordered = sorted([s for s in stages.values() if s.kind == "module"], key=lambda s: s.order)
    first_input = next((k for k in stages if k.startswith("input:")), None)
    has_pred = {e.dst for e in edges}
    prev = first_input
    for s in ordered:
        if s.key not in has_pred and prev is not None:
            edges.append(Edge(prev, s.key))
        prev = s.key
    if not out_specs:
        out_specs = [OutputSpec(p, s, [ordered[-1].key] if ordered else []) for p, s, _ in trace.outputs]
    for o in out_specs:
        if not o.sources and ordered:
            o.sources = [ordered[-1].key]

    # de-duplicate edges and drop self loops
    uniq: Dict[Tuple[str, str], Edge] = {}
    for e in edges:
        if e.src != e.dst and e.src in stages and e.dst in stages:
            uniq[(e.src, e.dst)] = e
    g = StageGraph(stages, list(uniq.values()), out_specs, source, notes)
    _mark_heads(g)
    _insert_representations(g, trace)
    _classify_edges(g)
    for e in g.edges:
        s = g.stages[e.src]
        e.features = _features(s.shape)
    return g


def _features(shape: Optional[Tuple[int, ...]]) -> int:
    if not shape:
        return 0
    if len(shape) >= 3 and shape[1] != shape[2]:
        return shape[1]
    return shape[-1]


def _mark_heads(g: StageGraph) -> None:
    for o in g.outputs:
        for s in o.sources:
            if s in g.stages:
                g.stages[s].is_head = True
                g.stages[s].outputs.append(o.name)


def _insert_representations(g: StageGraph, trace: TraceResult) -> None:
    """Show the latent vector a head consumes when the preceding stage is still spatial / token-shaped."""
    created: Dict[Any, str] = {}
    for st in list(g.ordered()):
        if not st.is_head or st.call is None or not st.call.in_shapes:
            continue
        preds = [e for e in g.preds(st.key)]
        if len(preds) != 1:
            continue
        p = g.stages[preds[0].src]
        in_shape = st.call.in_shapes[0]
        in_rank = len([v for v in in_shape[1:]])
        p_rank = len([v for v in (p.shape or ())[1:] if v != 1])
        if in_rank == 1 and p_rank >= 2:
            node = st.call.in_nodes[0] if st.call.in_nodes else None
            key = created.get((p.key, node, in_shape))
            if key is None:
                key = f"repr:{st.key}"
                created[(p.key, node, in_shape)] = key
                g.stages[key] = Stage(key=key, label="representation", kind="representation",
                                      order=st.order - 0.5, call=st.call, capture="input", shape=in_shape,
                                      type_name="latent")
                g.edges.append(Edge(p.key, key))
            g.edges = [e for e in g.edges if not (e.src == p.key and e.dst == st.key)]
            g.edges.append(Edge(key, st.key))


def _classify_edges(g: StageGraph) -> None:
    for st in g.stages.values():
        ins = g.preds(st.key)
        if len(ins) <= 1:
            for e in ins:
                e.kind = "main"
            continue
        # primary = the deepest incoming pathway (most ancestors), ties -> most recent
        primary = max(ins, key=lambda e: (len(g.ancestors(e.src)), g.stages[e.src].order))
        anc = g.ancestors(primary.src)
        for e in ins:
            if e is primary:
                e.kind = "main"
            elif e.src in anc:
                e.kind = "skip"
            else:
                e.kind = "merge"


# ---------------------------------------------------------------------------
# torch.fx (optional static topology)
# ---------------------------------------------------------------------------


def fx_stage_edges(model: nn.Module, calls: Sequence[ModuleCall], trace: TraceResult):
    """Derive stage edges from ``torch.fx.symbolic_trace``; returns None on failure."""
    try:
        import torch.fx as fx

        names = {c.name for c in calls}

        class _Tracer(fx.Tracer):
            def is_leaf_module(self, m, qualname):
                return qualname in names or super().is_leaf_module(m, qualname)

        graph = _Tracer().trace(model)
    except Exception as e:
        warnings.warn(f"neural_flow: model could not be symbolically traced ({type(e).__name__}). "
                      f"Using runtime execution order instead.")
        return None
    key_of = {c.name: c.key for c in calls if c.call_index == 0}
    inputs = list(trace.input_shapes)
    producer: Dict[Any, List[str]] = {}
    edges: List = []
    ph = 0
    out_src: List[str] = []
    for node in graph.nodes:
        if node.op == "placeholder":
            nm = inputs[ph] if ph < len(inputs) else node.name
            ph += 1
            producer[node] = [f"input:{nm}"]
            continue
        srcs: List[str] = []
        for a in node.all_input_nodes:
            for s in producer.get(a, []):
                if s not in srcs:
                    srcs.append(s)
        if node.op == "call_module" and node.target in key_of:
            k = key_of[node.target]
            for s in srcs:
                edges.append(Edge(s, k))
            producer[node] = [k]
        elif node.op == "output":
            out_src = srcs
        else:
            producer[node] = srcs
    return edges, out_src
