"""Public API: ``trace_model``, ``visualize_model``."""
from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from . import adapters as _adapters
from .capture import CaptureResult, TraceResult, capture_stages, prepare_inputs, trace_metadata
from .config import FlowConfig, make_config
from .graph import StageGraph, build_stage_graph
from .outputs import OutputView, interpret_output, output_leaves
from .raster import (Visual, attention_rollout, render_attention, render_segmentation, render_vector,
                     visualize_tensor, upsample, colorize, robust_norm)
from .stages import select_stages, short_label
from .tensors import summarize_tensor


@dataclass
class FlowResult:
    """Everything computed for one model/input pair (renderable many times)."""

    config: FlowConfig
    trace: TraceResult
    graph: StageGraph
    capture: CaptureResult
    views: Dict[str, OutputView]              # output node key -> view
    out_nodes: Dict[str, List[str]]           # output node key -> source stage keys
    adapters: List[str]
    notes: List[str] = field(default_factory=list)
    rollout: Optional[np.ndarray] = None
    model_name: str = ""
    explanation: Any = None
    context: Dict[str, Any] = field(default_factory=dict)   # e.g. whole-volume base for sliding-window outputs

    @property
    def stages(self):
        return self.graph.ordered()

    def summary_table(self) -> str:
        rows = [f"{'stage':28s} {'type':22s} {'shape':24s} kind"]
        for s in self.stages:
            sm = s.summary
            rows.append(f"{s.label[:28]:28s} {s.type_name[:22]:22s} {(sm.shape_label() if sm else '-')[:24]:24s} {sm.kind if sm else '-'}")
        for k, v in self.views.items():
            rows.append(f"{'→ ' + v.title:28s} {v.kind:22s} {v.headline[:24]:24s} {v.subline}")
        return "\n".join(rows)


def _tensor_leaves(obj) -> List[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, dict):
        return [t for v in obj.values() for t in _tensor_leaves(v)]
    if isinstance(obj, (list, tuple)):
        return [t for v in obj for t in _tensor_leaves(v)]
    return []


def _user_set(kw: Dict[str, Any]) -> set:
    return {k for k, v in kw.items() if v is not None}


def trace_model(model: nn.Module, inputs: Any, config: Optional[FlowConfig] = None, **kw) -> FlowResult:
    """Instrument ``model`` on ``inputs`` and compute stage summaries (no drawing).

    Deep supervision (nnU-Net's extra low-resolution training outputs) is switched
    off for the duration unless ``aux_outputs=True``; the model is restored afterwards.
    """
    cfg0 = make_config(config, **kw)
    with _adapters.inference_view(model, keep_aux=cfg0.aux_outputs) as switched:
        res = _trace_model(model, inputs, config, **kw)
    if switched:
        res.notes.append("deep supervision switched off (inference view): only the full-resolution output is shown")
    return res


def _trace_model(model: nn.Module, inputs: Any, config: Optional[FlowConfig] = None, **kw) -> FlowResult:
    user = _user_set(kw)
    cfg = make_config(config, **kw)
    if cfg.sliding_window:
        from .volume3d import sliding_window_trace

        return sliding_window_trace(model, inputs, config, kw)
    if cfg.style == "cinematic" and "theme" not in user and not (config is not None and config.theme != "light"):
        cfg = cfg.updated(theme="black")
    notes: List[str] = []
    prep = prepare_inputs(model, inputs, cfg)
    if cfg.voxel_spacing is not None and cfg.physical_fov is None:
        from .raster import canonical_fov

        for t in [t for _, t in prep.named]:
            if t.dim() == 5:
                fov = canonical_fov(tuple(t.shape[2:]), cfg.voxel_spacing, cfg.volume_axes)
                if fov is not None:
                    cfg = cfg.updated(physical_fov=fov)
                break
    trace = trace_metadata(model, prep, cfg)
    notes += trace.notes

    ads = _adapters.matching_adapters(model, trace)
    for a in ads:
        cfg = _adapters.apply_defaults(cfg, a.defaults(model, trace), user | (set(config.__dict__) if config else set()))

    calls = select_stages(trace, cfg, ads)
    if not calls:
        warnings.warn("neural_flow: no module stages selected; showing inputs and outputs only")
    graph = build_stage_graph(trace, calls, cfg, model, label_fn=lambda c: _pretty(short_label(c, cfg)))
    graph.meta.update(getattr(trace, "meta", {}) or {})
    for a in ads:
        if hasattr(a, "annotate"):
            try:
                a.annotate(trace, graph)
            except Exception:
                pass
    notes += graph.notes

    requests = []
    for st in graph.stages.values():
        if st.call is not None:
            requests.append((st.call, st.capture, st.key))
    budget = int(cfg.max_capture_mb * 1024 * 1024)
    cap = capture_stages(model, prep, requests, cfg, summarize_tensor, budget)
    notes += cap.notes
    for st in graph.stages.values():
        if st.kind == "input":
            st.summary = cap.inputs.get(st.label)
        else:
            st.summary = cap.summaries.get(st.key)
        if st.summary is None and st.kind != "input":
            notes.append(f"stage {st.key} produced no tensor output; skipped")
    # drop stages that produced nothing (no tensor output)
    for k in [k for k, s in graph.stages.items() if s.summary is None]:
        _remove_stage(graph, k)

    # attention (optional)
    rollout = None
    for rec in cap.attention:
        if rec.stage_key in graph.stages:
            graph.stages[rec.stage_key].extra.setdefault("attention", []).append(rec)
    if cap.attention:
        mats = [r.head_avg.numpy() for r in cap.attention]
        n0 = mats[-1].shape[-1]
        mats = [m for m in mats if m.shape[-1] == n0 and m.shape[-2] == n0]
        if mats:
            try:
                rollout = attention_rollout(mats)
            except Exception:
                rollout = None

    # outputs
    raw_leaves = dict(output_leaves(cap.raw_outputs))
    summ_by_name = dict(cap.outputs)
    views: Dict[str, OutputView] = {}
    out_nodes: Dict[str, List[str]] = {}
    in_spatial = None
    for st in graph.stages.values():
        if st.kind == "input" and st.summary is not None and st.summary.kind in ("image2d", "volume3d"):
            in_spatial = tuple(st.summary.spatial)
            break
    for o in graph.outputs:
        key = f"out:{o.name}"
        srcs = [s for s in o.sources if s in graph.stages]
        head = graph.stages[srcs[0]] if srcs else None
        head_name = head.call.name if head is not None and head.call is not None else ""
        views[key] = interpret_output(o.name, raw_leaves.get(o.name), summ_by_name.get(o.name), cfg,
                                      head_name=head_name, input_spatial=in_spatial)
        out_nodes[key] = srcs
        if len(graph.outputs) > 1 and views[key].title in ("output", "0", "1", "2", "3"):
            views[key].title = f"output {o.name}"

    _adapters.assign_concepts(graph, ads)
    ads_final = _adapters.matching_adapters(model, trace, graph)
    _adapters.assign_concepts(graph, ads_final)

    reduced = [s.key for s in graph.stages.values() if s.summary is not None and s.summary.reduced and s.kind != "input"]
    if reduced:
        notes.append("activations summarised (reduced resolution) for: " + ", ".join(reduced))
    res = FlowResult(cfg, trace, graph, cap, views, out_nodes, [a.name for a in ads_final], notes, rollout,
                     type(model).__name__)
    if cfg.explain or (cfg.explain is None and cfg.style == "cinematic"):
        from .explain import compute_explanations

        res.explanation = compute_explanations(model, prep, res)
        res.notes += res.explanation.notes
    if cfg.flat_3d:
        from .volume3d import flatten_result

        flatten_result(res, cfg.flat_3d if isinstance(cfg.flat_3d, str) else "max")
    return res


def _pretty(label: str) -> str:
    m = re.match(r"^(?:encoder_)?layers?_(\d+)$", label)
    if m:
        return f"block {m.group(1)}"
    m = re.match(r"^(blocks|layers|layer|features|stages|stage)\.(\d+)$", label)
    if m:
        return f"{m.group(1).rstrip('s')} {m.group(2)}"
    return label


def _remove_stage(g: StageGraph, key: str) -> None:
    ins = [e for e in g.edges if e.dst == key]
    outs = [e for e in g.edges if e.src == key]
    g.edges = [e for e in g.edges if e.src != key and e.dst != key]
    from .graph import Edge

    for a in ins:
        for b in outs:
            if not any(e.src == a.src and e.dst == b.dst for e in g.edges):
                g.edges.append(Edge(a.src, b.dst, b.kind, a.features))
    for o in g.outputs:
        if key in o.sources:
            o.sources = [s for s in o.sources if s != key] + [a.src for a in ins]
    del g.stages[key]


# ---------------------------------------------------------------------------
# visuals
# ---------------------------------------------------------------------------


def build_visuals(res: FlowResult, strategy: Optional[str] = None):
    cfg = res.config
    visuals: Dict[str, Visual] = {}
    input_image = None
    for st in res.graph.ordered():
        sm = st.summary
        if sm is None:
            continue
        if st.kind == "input" and sm.image is not None and input_image is None:
            input_image = sm.image
        if st.kind == "representation" or (sm.kind in ("vector", "scalar") and st.kind != "input"):
            vis = render_vector(sm.vector if sm.vector is not None else np.zeros(1), cfg)
        else:
            vis = visualize_tensor(sm, cfg, strategy=strategy, volume_axes=cfg.volume_axes, style=cfg.style)
        # attention inset (CLS -> patches, averaged over heads)
        att = st.extra.get("attention")
        if att and cfg.capture_attention and sm.kind == "tokens":
            grid = sm.spatial if len(sm.spatial) == 2 else None
            a = render_attention(att[-1].weights.numpy(), cfg, sm.n_special, grid)
            if "cls" in a:
                vis.extras["attn"] = a["cls"]
        if st.kind == "representation" and res.rollout is not None:
            r = _rollout_map(res, st)
            if r is not None:
                vis.extras["attn"] = r
        visuals[st.key] = vis
    out_visuals: Dict[str, Visual] = {}
    ctx = res.context
    for key, ov in res.views.items():
        if ov.kind == "segmentation" and ov.mask is not None:
            base = ctx.get("seg_base", input_image)
            out_visuals[key] = render_segmentation(ov.mask, base, cfg, cfg.volume_axes, fov=ctx.get("seg_fov"),
                                                   boxes=ctx.get("seg_boxes"))
        elif ov.kind == "embedding" and ov.vector is not None:
            out_visuals[key] = render_vector(ov.vector, cfg, max_len=256)
    return visuals, out_visuals


def _rollout_map(res: FlowResult, st) -> Optional[np.ndarray]:
    toks = [s.summary for s in res.graph.ordered() if s.summary is not None and s.summary.kind == "tokens"
            and len(s.summary.spatial) == 2]
    if not toks:
        return None
    t = toks[-1]
    R = res.rollout
    gh, gw = t.spatial
    k = t.n_special
    if k < 1 or R.shape[0] != k + gh * gw:
        return None
    m = R[0, k:].reshape(gh, gw)
    return upsample(colorize(robust_norm(m, (1, 99.5)), "viridis"), 96)


def _title(res: FlowResult) -> str:
    if res.config.style == "cinematic":
        return res.config.title or res.model_name
    return res.config.title or f"{res.model_name} · activation flow"


def _subtitle(res: FlowResult) -> str:
    if res.config.subtitle is not None:
        return res.config.subtitle
    if res.config.style == "story":
        return ""
    if res.config.style == "cinematic":
        ins = ", ".join(f"{s.label} {'×'.join(map(str, s.shape[1:] if s.shape else ()))}"
                        for s in res.graph.ordered() if s.kind == "input")
        n = len([s for s in res.graph.stages.values() if s.kind != "input"])
        return f"{ins}{_sw_note(res)}   ·   {n} stages   ·   pages ranked by {res.config.channel_strategy}"
    ins = ", ".join(f"{s.label} {'×'.join(map(str, s.shape[1:] if s.shape else ()))}"
                    for s in res.graph.ordered() if s.kind == "input")
    n = len([s for s in res.graph.stages.values() if s.kind != "input"])
    return f"input: {ins}{_sw_note(res)}   ·   {n} stages of {len(res.trace.calls)} module calls   ·   channels by {res.config.channel_strategy}   ·   topology: {res.graph.topology_source}"


def _sw_note(res: FlowResult) -> str:
    sw = res.context.get("sliding_window")
    if not sw:
        return ""
    vol = "×".join(map(str, sw["volume"]))
    at = ",".join(map(str, sw["start"]))
    return f" (window at {at} of a {vol} volume; output fused from {sw['windows']} windows)"


def draw(res: FlowResult, strategy: Optional[str] = None, frame=None):
    from .render import render_flow

    if res.config.style == "cinematic":
        from .cinematic import render_cinematic

        return render_cinematic(res, frame=frame, title=_title(res), subtitle=_subtitle(res))

    visuals, out_visuals = build_visuals(res, strategy)
    foot = None
    if any(s.summary is not None and s.summary.reduced and s.kind != "input" for s in res.graph.stages.values()):
        foot = "≈ activation summarised (downsampled) to respect max_capture_mb / display limits"
    return render_flow(res.graph, visuals, res.views, out_visuals, res.out_nodes, res.config,
                       title=_title(res), subtitle=_subtitle(res), footnote=foot)


def visualize_model(model: nn.Module, inputs: Any = None, output: Optional[str] = None, *,
                    config: Optional[FlowConfig] = None, interactive: bool = False, return_result: bool = False,
                    **kw):
    """Visualise how ``inputs`` flow through ``model``.

    Returns a matplotlib ``Figure`` (with the computed :class:`FlowResult` attached as
    ``fig.flow``).  ``output`` may end in ``.png``, ``.svg``, ``.pdf`` or ``.html``
    (``.html`` / ``interactive=True`` writes the interactive explorer).
    """
    if inputs is None:
        inputs = kw.pop("input_tensor", None)
    if inputs is None:
        raise TypeError("visualize_model() needs `inputs`")
    res = trace_model(model, inputs, config=config, **kw)
    ff = draw(res)
    fig = ff.fig
    fig.flow = res
    fig.flow_figure = ff
    if output:
        ext = os.path.splitext(output)[1].lower()
        if ext in (".html", ".htm") or interactive:
            from .interactive import write_html

            html_path = output if ext in (".html", ".htm") else os.path.splitext(output)[0] + ".html"
            write_html(res, html_path, figure=ff)
            if ext not in (".html", ".htm"):
                fig.savefig(output, dpi=res.config.dpi, facecolor=fig.get_facecolor())
        else:
            fig.savefig(output, dpi=res.config.dpi, facecolor=fig.get_facecolor())
    for n in res.notes:
        if n.startswith("activations summarised"):
            warnings.warn("neural_flow: " + n, stacklevel=2)
    return (fig, res) if return_result else fig
