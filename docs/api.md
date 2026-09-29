# API reference

```python
from neural_flow import (visualize_model, trace_model, draw, animate_inputs, animate_model,
                         FlowConfig, FlowResult, summarize_tensor, visualize_tensor)
from neural_flow import sequences          # pan, crossfade, zoom, occlusion_sweep, slices_to_frames
```

## Functions

### `visualize_model(model, inputs, output=None, *, config=None, interactive=False, return_result=False, **options)`

Traces `model` on `inputs` and draws the representation flow.

| argument | meaning |
|---|---|
| `model` | `torch.nn.Module` (switched to `eval()` during the call, then restored) |
| `inputs` | tensor, tuple/list of tensors (positional arguments), or dict (keyword arguments) |
| `output` | path; `.png`, `.svg`, `.pdf` or `.html` |
| `config` | optional `FlowConfig`; keyword `options` override its fields |
| `interactive` | also write the HTML explorer next to a static `output` |
| `return_result` | return `(fig, FlowResult)` instead of `fig` |

Returns a matplotlib `Figure` with `fig.flow` (`FlowResult`) and `fig.flow_figure`
(`FlowFigure`) attached.

### `trace_model(model, inputs, config=None, **options) -> FlowResult`

Runs the metadata pass, stage selection, topology, capture and, if enabled, explanations. Does not
draw.

### `draw(result, strategy=None, frame=None) -> FlowFigure`

Renders a `FlowResult` with its configured style. `strategy` overrides the channel ranking for this
drawing. `FlowFigure` has `.fig`, `.ax`, `.stage_artists` (stage key → artists), `.edge_artists`
and `.layout`.

### `animate_inputs(model, inputs, output="input_sweep.mp4", *, fps=8, frame_labels=None, title=None, subtitle=None, style="cinematic", figsize=(16, 9), dpi=120, hold_last=0, footer_height=2.2, progress=True, **options) -> str`

Renders one frame per input with stages, channels, per-channel scales and the PCA colour basis held
fixed. `inputs` is a tensor `[T, …]` (one frame per leading index) or any iterable of per-frame
inputs (tensors or dicts). Writes MP4 (needs `imageio-ffmpeg`) or GIF, and returns the path
written. See [movies.md](movies.md).

### `animate_model(model, inputs, output="network_flow.gif", *, fps=12, frames_per_stage=6, hold_frames=18, config=None, **options) -> str`

A single input; stages light up one after another in data-flow order.

### `neural_flow.sequences`

| function | returns |
|---|---|
| `pan(image, window=224, steps=48, out_size=None, vertical=False)` | `[T, C, window, window]` windows sliding across a wide (or tall) image `[C, H, W]` |
| `crossfade(a, b, steps=32, ease=True)` | `[T, …]` morph between two inputs |
| `zoom(image, start=1.0, end=0.35, steps=40, center=None, out_size=None)` | `[T, C, S, S]` zoom into `center` (fractions of H, W) |
| `occlusion_sweep(image, patch=48, stride=24, value=0.0)` | `[T, C, H, W]` sliding occluding patch |
| `slices_to_frames(volume, axis=-1, step=1)` | `[T, C, H, W]` 2-D frames through a 3-D volume |

### `summarize_tensor(tensor, role, budget_bytes, cfg, ctx=None) -> TensorSummary` and `visualize_tensor(summary, cfg, strategy=None, volume_axes="xyz") -> Visual`

The tensor → summary → RGBA-raster dispatcher, usable on its own.
`role ∈ {"input", "activation", "output"}`. `Visual.image` is an `[H, W, 4]` float array.

## `FlowConfig` options

Every option can be passed as a keyword to `visualize_model`, `trace_model`, `animate_inputs` and
`animate_model`, or collected in a `FlowConfig(...)`. Adapters (CNN, U-Net, transformer, 3-D)
may adjust defaults you have not set yourself; for example, 3-D models rank channels by
`variance`.

### Stage selection

| option | default | meaning |
|---|---|---|
| `layer_selection` | `"auto"` | `"auto"`, `"all"` (every non-trivial leaf module) or a predicate `f(ModuleCall) -> bool` |
| `layers` | `None` | explicit selectors: `"name"`, `"name#k"` (k-th call of a reused module), `"re:<regex>"`, `"type:<ClassRegex>"` |
| `exclude` | `None` | selectors removed from the selection |
| `max_stages` | `10` | hard cap on module stages |
| `target_stages` | `None` | preferred count (8 technical, 6 story, 7 cinematic) |
| `labels` | `None` | `{module name: display label}` |

### Tensor → picture

| option | default | meaning |
|---|---|---|
| `max_channels` | `16` | channels retained per stage |
| `channel_strategy` | `"energy"` | `energy` (mean x²), `variance`, `spread` (p90 − p10), `mean_abs`, `even`, `pca` |
| `normalize` | `"channel"` | `channel` or `stage` (shared) percentile normalization |
| `percentiles` | `(1, 99)` | display range |
| `diverging` | `False` | `True` / `"auto"`: zero-centred colour map for signed activations |
| `cmap` | theme default | activation colour map (`magma` light, `inferno` dark/black) |
| `volume_mode` | `"auto"` | 3-D: `volume`, `ortho`, `montage`, `projection` |
| `volume_style` | `"voxels"` | 3-D volume rendering: `voxels`, `cutaway`, `glow` |
| `projection` | `"max"` | `max`, `mean`, `meanabs` for `volume_mode="projection"` |
| `volume_axes` | `"xyz"` | `xyz` (nibabel/MONAI `[X, Y, Z]`) or `dhw` (slice stack `[D, H, W]`) |
| `token_mode` | `"auto"` | `grid`, `heatmap`, `pca`, `l2`, `mean` |
| `cls_tokens`, `patch_grid` | inferred | number of leading special tokens; `(rows, cols)` of patch tokens |
| `channels_last` | inferred | force `[B, H, W, C]` interpretation |
| `capture_attention` | `True` | record attention weights (MHA / SDPA) |

### Outputs

| option | default | meaning |
|---|---|---|
| `class_names` | `None` | list, or `{output name: list}` |
| `output_types` | `None` | `{output name: softmax \| sigmoid \| multilabel \| multilabel_probs \| regression \| segmentation \| embedding \| raw}` |
| `output_interpreter` | `None` | `f(name, tensor) -> OutputView \| dict \| str` |
| `top_k` | `3` | entries listed per classification card |
| `input_names` | `None` | display names for the inputs |
| `batch_index` | `0` | which sample of a batch to show |

### Explanations (cinematic)

| option | default | meaning |
|---|---|---|
| `explain` | `None` | gradient explanations; `None` means on for cinematic, off otherwise |
| `front_page` | `"pca"` | front page of each stack: `pca` (all channels as RGB) or `channel` (strongest channel) |

### Capture and memory

| option | default | meaning |
|---|---|---|
| `capture_device` | `"cpu"` | where summaries are stored |
| `max_capture_mb` | `500` | total budget for retained summaries |
| `max_spatial_2d`, `max_spatial_3d` | `128`, `48` | display resolution caps for activations |
| `max_input_2d`, `max_input_3d` | `256`, `128` | resolution caps for the input image / volume |
| `topology` | `"auto"` | `auto` / `runtime` (dataflow tracing), `fx`, `sequential` |
| `random_state` | `42` | reserved; all reductions are deterministic |

### Layout and appearance

| option | default | meaning |
|---|---|---|
| `style` | `"technical"` | `technical`, `story`, `cinematic` |
| `theme` | `"light"` | `light`, `dark`, `black` (`cinematic` defaults to `black`) |
| `layout` | `"horizontal"` | `horizontal`, `vertical`, `wrap` |
| `figsize` | auto | e.g. `(16, 9)`; long chains wrap into rows to fill it |
| `dpi` | `200` | raster resolution |
| `title`, `subtitle` | auto | figure text |
| `font_scale` | `1.0` | scale all text |
| `show_skips` | `True` | draw skip connections |
| `edge_width` | `"features"` | arrow width ∝ log(features), or `constant` |
| `unet_layout` | `"auto"` | U-shaped layout for encoder/decoder networks |
| `overlay_segmentation` | `True` | overlay 2-D segmentations on the input |

### Movie internals (set automatically by `animate_inputs`)

| option | meaning |
|---|---|
| `force_channels` | `{stage key: [channel ids]}` shown in every frame |
| `force_pca` | `{stage key: (loadings [3, C], mean [C])}` fixed PCA colour basis |

## `FlowResult`

| attribute | content |
|---|---|
| `config` | the resolved `FlowConfig` (after adapter defaults) |
| `stages` | stages in execution order (`Stage`: `key`, `label`, `kind`, `concept`, `summary`, `call`, `is_head`) |
| `graph` | `StageGraph`: `stages`, `edges` (`src`, `dst`, `kind ∈ main/skip/merge`), `outputs`, `topology_source` |
| `views` | output key → `OutputView` (`kind`, `headline`, `subline`, `items`, `mask`, `vector`) |
| `explanation` | `Explanation`: `units` (receptive fields, dependency maps), `gradcam`, `contributions` |
| `rollout` | attention-rollout matrix (transformers), if captured |
| `trace` | raw metadata pass: every module call, shapes, dataflow ops |
| `notes` | warnings and summarization notes |
| `summary_table()` | text table of stages and outputs |

## Extending

* **Adapters**: subclass `neural_flow.adapters.Adapter` (`match`, `defaults`, `concepts`) and
  register it with `neural_flow.adapters.register_adapter(...)` to add defaults and conceptual
  stage names for an architecture family.
* **Custom output semantics**: `output_interpreter`.
* **Custom stage choice**: `layer_selection=callable` receives each `ModuleCall` (`name`,
  `type_name`, `call_index`, `in_shapes`, `out_shapes`, `n_params`, `depth`, …).
