"""Optional interactive HTML explorer (single self-contained file, no server, no CDN).

Features: click a stage to inspect module name/type, tensor shape, min/max/mean/std,
switch the channel-selection strategy, expand individual channels, and browse
individual attention heads when attention was captured.  The static figure is
embedded as a second tab.
"""
from __future__ import annotations

import base64
import html
import io
import json
from typing import Any, Dict, List, Optional

import numpy as np

from .raster import (STRATEGY_NAMES, Visual, colorize, render_attention, robust_norm, to_data_uri, upsample,
                     visualize_tensor, act_cmap, canonical_volume, ortho_slices, _slice_tiles, mosaic, activation_peak)
from .tensors import STRATEGIES


def _fmt(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    a = abs(v)
    if a != 0 and (a < 1e-3 or a >= 1e4):
        return f"{v:.3e}"
    return f"{v:.4g}"


def _channel_tiles(summ, cfg, limit: int = 32) -> List[Dict[str, Any]]:
    out = []
    if summ.maps is None:
        return out
    cm = act_cmap(cfg)
    ids = summ.channel_ids[:limit]
    for i, cid in enumerate(ids):
        m = summ.maps[i]
        if m.ndim == 3:  # volume channel -> orthogonal slices through its peak
            V = canonical_volume(m, cfg.volume_axes)
            img = mosaic(_slice_tiles(ortho_slices(V, activation_peak(np.abs(V))), cm, 72), cols=3, gap=3)
        else:
            img = upsample(colorize(robust_norm(m, cfg.percentiles), cm), 96)
        stats = {k: _fmt(v[cid]) for k, v in summ.channel_scores.items() if v is not None and cid < len(v)}
        out.append({"id": int(cid), "img": to_data_uri(img), "stats": stats})
    return out


def _stage_payload(st, vis: Visual, cfg, res) -> Dict[str, Any]:
    summ = st.summary
    d: Dict[str, Any] = {
        "key": st.key, "label": st.label, "concept": st.concept, "kind": st.kind,
        "module": st.call.name if st.call is not None else st.label,
        "type": st.type_name or ("input" if st.kind == "input" else ""),
        "shape": list(summ.shape) if summ else [],
        "shape_label": summ.shape_label() if summ else "",
        "tensor_kind": summ.kind if summ else "",
        "stats": {k: _fmt(v) for k, v in (summ.stats or {}).items()} if summ else {},
        "notes": list(summ.notes) if summ else [],
        "reduced": bool(summ.reduced) if summ else False,
        "thumb": to_data_uri(vis.image),
        "strategies": {}, "channels": [], "heads": [],
    }
    if summ is not None and summ.kind in ("image2d", "volume3d", "tokens") and summ.maps is not None and st.kind != "input":
        # 3-D stages: strategy views use orthogonal slices (fast); the voxel render stays the thumbnail
        scfg = cfg.updated(volume_mode="ortho") if summ.kind == "volume3d" else cfg
        for s in STRATEGIES:
            try:
                v = visualize_tensor(summ, scfg, strategy=s, volume_axes=cfg.volume_axes)
                d["strategies"][s] = to_data_uri(v.image)
            except Exception:
                pass
        d["channels"] = _channel_tiles(summ, cfg)
        if summ.pca_var is not None:
            d["pca_var"] = [_fmt(v) for v in summ.pca_var]
    att = st.extra.get("attention")
    if att:
        w = att[-1].weights.numpy()
        grid = summ.spatial if summ is not None and len(summ.spatial) == 2 else None
        k = summ.n_special if summ is not None else 0
        for h in range(min(w.shape[0], 16)):
            a = render_attention(w[h:h + 1], cfg, k, grid)
            d["heads"].append({"head": h, "matrix": to_data_uri(a["matrix"]),
                               "cls": to_data_uri(a["cls"]) if "cls" in a else None,
                               "entropy": _fmt(float(np.mean(a["entropy_vector"])))})
        avg = render_attention(w, cfg, k, grid)
        d["heads_avg"] = {"matrix": to_data_uri(avg["matrix"]), "cls": to_data_uri(avg["cls"]) if "cls" in avg else None}
    return d


def build_payload(res, figure=None) -> Dict[str, Any]:
    from .api import build_visuals, _title, _subtitle

    cfg = res.config
    visuals, out_visuals = build_visuals(res)
    stages = []
    for st in res.graph.ordered():
        if st.key in visuals:
            stages.append(_stage_payload(st, visuals[st.key], cfg, res))
    outputs = []
    for key, ov in res.views.items():
        o = {"key": key, "title": ov.title, "kind": ov.kind, "headline": ov.headline, "subline": ov.subline,
             "items": [[a, _fmt(b)] for a, b in ov.items], "sources": res.out_nodes.get(key, []), "note": ov.note}
        if key in out_visuals:
            o["img"] = to_data_uri(out_visuals[key].image)
        outputs.append(o)
    edges = [{"src": e.src, "dst": e.dst, "kind": e.kind} for e in res.graph.edges]
    fig_uri = None
    if figure is not None:
        buf = io.BytesIO()
        figure.fig.savefig(buf, format="png", dpi=min(cfg.dpi, 150), facecolor=figure.fig.get_facecolor())
        fig_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    return {"title": _title(res), "subtitle": _subtitle(res), "stages": stages, "outputs": outputs, "edges": edges,
            "figure": fig_uri, "notes": res.notes, "topology": res.graph.topology_source,
            "strategy": cfg.channel_strategy, "strategy_names": STRATEGY_NAMES}


def write_html(res, path: str, figure=None) -> str:
    data = build_payload(res, figure)
    doc = _TEMPLATE.replace("__TITLE__", html.escape(data["title"])).replace(
        "__DATA__", json.dumps(data).replace("</", "<\\/"))
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    return path


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#fff;--fg:#16181d;--muted:#6b7280;--card:#f3f4f6;--edge:#dfe2e7;--accent:#c2410c;--skip:#2f7fc1;--sel:#2563eb}
@media (prefers-color-scheme:dark){:root{--bg:#0e1016;--fg:#e8eaef;--muted:#a3aab6;--card:#191c24;--edge:#2a2f3a;--accent:#fb923c;--skip:#5aa9e6;--sel:#60a5fa}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:18px 20px 6px}h1{font-size:20px;margin:0}header p{margin:4px 0 0;color:var(--muted);font-size:12.5px}
.tabs{display:flex;gap:4px;padding:8px 20px 0}.tabs button{border:1px solid var(--edge);background:var(--card);color:var(--fg);padding:6px 12px;border-radius:6px 6px 0 0;cursor:pointer}
.tabs button.on{background:var(--bg);border-bottom-color:var(--bg);font-weight:600}
.pane{display:none;border-top:1px solid var(--edge);padding:14px 20px}.pane.on{display:block}
.flow{display:flex;align-items:center;gap:6px;overflow-x:auto;padding:6px 2px 14px}
.card{flex:0 0 auto;background:var(--card);border:2px solid transparent;border-radius:10px;padding:8px;cursor:pointer;text-align:center;min-width:96px}
.card:hover{border-color:var(--edge)}.card.sel{border-color:var(--sel)}
.card img{display:block;height:120px;max-width:220px;object-fit:contain;image-rendering:pixelated;margin:0 auto}
.card .t{font-weight:600;font-size:12.5px;margin-bottom:4px;white-space:nowrap}.card .s{font-size:11px;color:var(--muted);margin-top:4px;white-space:nowrap}
.card.out{border-color:var(--accent);background:transparent}.card.out .h{font-size:20px;font-weight:700}
.arrow{flex:0 0 auto;color:var(--muted);font-size:22px}
.branch{flex:0 0 auto;display:flex;flex-direction:column;gap:8px}
.detail{display:grid;grid-template-columns:minmax(260px,1.2fr) minmax(240px,1fr);gap:18px;margin-top:8px}
@media (max-width:760px){.detail{grid-template-columns:1fr}}
.big{background:var(--card);border-radius:10px;padding:10px;text-align:center}.big img{max-width:100%;height:340px;object-fit:contain;image-rendering:pixelated}
table{border-collapse:collapse;width:100%;font-size:12.5px}td{padding:3px 6px;border-bottom:1px solid var(--edge)}td:first-child{color:var(--muted);width:40%}
.seg{display:flex;flex-wrap:wrap;gap:4px;margin:8px 0}.seg button{border:1px solid var(--edge);background:var(--card);color:var(--fg);border-radius:14px;padding:3px 10px;cursor:pointer;font-size:12px}
.seg button.on{background:var(--sel);color:#fff;border-color:var(--sel)}
.chans{display:grid;grid-template-columns:repeat(auto-fill,minmax(72px,1fr));gap:6px}
.chans div{background:var(--card);border-radius:6px;padding:3px;cursor:pointer;text-align:center;font-size:10.5px;color:var(--muted);border:2px solid transparent}
.chans div.on{border-color:var(--sel)}.chans img{width:100%;image-rendering:pixelated;display:block}
h3{font-size:13px;margin:14px 0 4px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
.note{font-size:12px;color:var(--muted)}.fig img{max-width:100%}
</style></head><body>
<header><h1 id="title"></h1><p id="sub"></p></header>
<div class="tabs"><button class="on" data-p="explore">Explore</button><button data-p="figure">Static figure</button></div>
<div class="pane on" id="explore"><div class="flow" id="flow"></div><div class="detail" id="detail"></div></div>
<div class="pane fig" id="figure"></div>
<script>
const D = __DATA__;
const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
$('#title').textContent = D.title; $('#sub').textContent = D.subtitle;
document.querySelectorAll('.tabs button').forEach(b => b.onclick = () => {
  document.querySelectorAll('.tabs button').forEach(x => x.classList.toggle('on', x === b));
  document.querySelectorAll('.pane').forEach(p => p.classList.toggle('on', p.id === b.dataset.p));
});
$('#figure').innerHTML = D.figure ? `<img alt="static activation-flow figure" src="${D.figure}">` : '<p class="note">No static figure embedded.</p>';
const byKey = Object.fromEntries(D.stages.map(s => [s.key, s]));
let sel = null, strat = D.strategy;
function card(s){
  const d = document.createElement('div'); d.className = 'card'; d.dataset.key = s.key;
  d.innerHTML = `<div class="t">${esc(s.label)}</div><img alt="${esc(s.label)} activations" src="${s.thumb}"><div class="s">${s.reduced?'≈ ':''}${esc(s.shape_label)}</div>`;
  d.onclick = () => select(s.key); return d;
}
function outCard(o){
  const d = document.createElement('div'); d.className = 'card out';
  d.innerHTML = `<div class="t" style="color:var(--accent)">${esc(o.title)}</div>` + (o.img ? `<img alt="${esc(o.title)}" src="${o.img}">` : `<div class="h">${esc(o.headline)}</div>`) + `<div class="s">${esc(o.subline)}</div>`;
  d.onclick = () => showOutput(o); return d;
}
(function build(){
  const f = $('#flow');
  D.stages.forEach((s, i) => { if (i) { const a = document.createElement('div'); a.className='arrow'; a.textContent='→'; f.appendChild(a); } f.appendChild(card(s)); });
  const a = document.createElement('div'); a.className='arrow'; a.textContent = D.outputs.length > 1 ? '⇉' : '→'; f.appendChild(a);
  const br = document.createElement('div'); br.className = 'branch'; D.outputs.forEach(o => br.appendChild(outCard(o))); f.appendChild(br);
  const skips = D.edges.filter(e => e.kind !== 'main');
  if (skips.length) { const n = document.createElement('p'); n.className='note';
    n.textContent = 'Non-sequential connections: ' + skips.map(e => `${(byKey[e.src]||{label:e.src}).label} ⇢ ${(byKey[e.dst]||{label:e.dst}).label} (${e.kind})`).join(' · ');
    f.after(n); }
  select(D.stages[Math.min(1, D.stages.length - 1)].key);
})();
function table(obj){ return '<table>' + Object.entries(obj).map(([k,v]) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join('') + '</table>'; }
function select(key){
  sel = byKey[key];
  document.querySelectorAll('.card').forEach(c => c.classList.toggle('sel', c.dataset.key === key));
  render(sel.strategies[strat] || sel.thumb);
}
function render(mainImg, chanNote){
  const s = sel, st = s.stats;
  let right = `<h3>Module</h3>` + table({name: s.module, type: s.type || '—', stage: s.kind, concept: s.concept, tensor: s.tensor_kind, shape: '[' + s.shape.join(', ') + ']'});
  right += `<h3>Statistics</h3>` + table({min: st.min, max: st.max, mean: st.mean, std: st.std, 'p1 / p99': (st.p_lo||'') + ' / ' + (st.p_hi||''), 'fraction < 0': st.frac_neg, 'fraction = 0': st.frac_zero});
  if (s.notes.length) right += `<p class="note">≈ ${esc(s.notes.join('; '))}</p>`;
  if (s.pca_var) right += `<p class="note">PCA explained variance (3 comps): ${s.pca_var.join(', ')}</p>`;
  let left = `<div class="big"><img alt="${esc(s.label)} detail" src="${mainImg}">${chanNote ? `<p class="note">${esc(chanNote)}</p>` : ''}</div>`;
  const names = Object.keys(s.strategies);
  if (names.length) left += `<h3>Channel selection</h3><div class="seg" id="strats">` + names.map(n => `<button data-s="${n}" class="${n===strat?'on':''}">${esc(D.strategy_names[n]||n)}</button>`).join('') + `</div>`;
  if (s.channels.length) left += `<h3>Retained channels (${s.channels.length}) — click to expand</h3><div class="chans" id="chans">` + s.channels.map((c,i) => `<div data-i="${i}"><img alt="channel ${c.id}" src="${c.img}">#${c.id}</div>`).join('') + `</div>`;
  if (s.heads.length) left += `<h3>Attention heads</h3><div class="seg" id="heads"><button data-h="avg" class="on">mean</button>` + s.heads.map(h => `<button data-h="${h.head}">h${h.head}</button>`).join('') + `</div><div class="big" id="attn"></div>`;
  $('#detail').innerHTML = `<div>${left}</div><div>${right}</div>`;
  document.querySelectorAll('#strats button').forEach(b => b.onclick = () => { strat = b.dataset.s; render(s.strategies[strat]); });
  document.querySelectorAll('#chans div').forEach(d => d.onclick = () => { const c = s.channels[+d.dataset.i];
    render(c.img, 'channel #' + c.id + ' · ' + Object.entries(c.stats).map(([k,v]) => k + ' ' + v).join(' · ')); });
  if (s.heads.length) { const show = h => { const a = h === 'avg' ? s.heads_avg : s.heads[+h];
      $('#attn').innerHTML = (a.cls ? `<img alt="CLS to patch attention" src="${a.cls}" style="max-height:200px">` : '') + `<img alt="attention matrix" src="${a.matrix}" style="max-height:200px">` + (a.entropy ? `<p class="note">mean entropy ${a.entropy}</p>` : '<p class="note">left: CLS→patch attention · right: attention matrix</p>'); };
    document.querySelectorAll('#heads button').forEach(b => b.onclick = () => { document.querySelectorAll('#heads button').forEach(x => x.classList.toggle('on', x===b)); show(b.dataset.h); });
    show('avg'); }
}
function showOutput(o){
  document.querySelectorAll('.card').forEach(c => c.classList.remove('sel'));
  const items = o.items.length ? '<h3>Top values</h3>' + table(Object.fromEntries(o.items)) : '';
  $('#detail').innerHTML = `<div><div class="big">${o.img ? `<img alt="${esc(o.title)}" src="${o.img}">` : `<div style="font-size:28px;font-weight:700">${esc(o.headline)}</div>`}<p class="note">${esc(o.subline)}</p></div></div>` +
    `<div><h3>Output</h3>${table({name: o.title, interpretation: o.kind, 'produced by': o.sources.map(k => (byKey[k]||{label:k}).label).join(', ')})}${items}${o.note ? `<p class="note">${esc(o.note)}</p>` : ''}</div>`;
}
</script></body></html>
"""
