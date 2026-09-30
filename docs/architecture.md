# Architecture

## Package layout

```
neural_flow/
  __init__.py      public API
  api.py           visualize_model / trace_model / draw: the orchestration
  config.py        FlowConfig: every option in one dataclass
  capture.py       instrumentation: metadata pass (hooks + TorchFunctionMode dataflow tracer),
                   capture pass (hooks on selected stages, on-device reduction), attention tracer
  stages.py        automatic / explicit stage selection
  graph.py         stage-level topology (runtime dataflow → torch.fx → execution order),
                   heads, representation stages, edge kinds (main / skip / merge)
  tensors.py       tensor interpretation and memory-aware reduction → TensorSummary
  explain.py       gradient pass: unit receptive fields, inter-stage dependency, Grad-CAM,
                   weight × activation contributions
  outputs.py       conservative interpretation of output heads → OutputView
  raster.py        tensor → RGBA rasters: mosaics, strips, token grids, 3-D ray caster,
                   orthogonal slices, segmentation overlays
  artistic.py      cinematic primitives: feature-map stacks, dot matrices, glow, vignettes, fonts
  layout.py        columns / lanes / U-dip / band wrapping
  render.py        technical & story renderer
  cinematic.py     cinematic renderer
  interactive.py   self-contained HTML explorer
  animate.py       "light-up" animation of a single input
  movie.py         movies over changing inputs
  sequences.py     input-sequence generators
  volume3d.py      3-D helpers: sliding-window inference and its movie, flat_3d projections
  nnunet.py        trained nnU-Net v2 results folders: plans → network, checkpoint, nnU-Net preprocessing,
                   TotalSegmentator downloads
  zoo.py           model / input loading for the command line (aliases, bundles, nnU-Net, NIfTI + spacing)
  cli.py           the `neural-flow` command
  adapters/        architecture families: nnunet (dynamic_network_architectures U-Nets; also switches
                   deep supervision off), transformer-unet (UNETR / Swin UNETR / UNesT), unet,
                   transformer, medical3d, cnn
  fonts/           Inter (SIL OFL)
```

## Data flow

```
inputs ─► prepare_inputs ─► trace_metadata ─────────► select_stages ─► build_stage_graph
             (capture.py)      hooks on all modules     (stages.py)       (graph.py)
                               + DataflowTracer
                                                                              │
          ┌───────────────────────────────────────────────────────────────────┘
          ▼
   capture_stages ── hooks on selected stages only; summarize_tensor() reduces each
   (capture.py)      activation on its own device; AttentionTracer records attention
          │
          ▼
   interpret_output (outputs.py) · assign_concepts (adapters) · compute_explanations (explain.py, optional)
          │
          ▼
   FlowResult ─► draw() ─► render_flow (render.py)  or  render_cinematic (cinematic.py)
                             └─ raster.py / artistic.py for pixels, layout.py for geometry
```

## Design rules

* **No permanent changes to the model.** Every hook and torch-function mode is removed in a
  `finally` block, `training` mode is restored, inputs are aliased before tagging, and parameters
  are never touched.
* **Reduce before moving.** Anything proportional to activation size is computed on the tensor's
  device, channel-chunk by channel-chunk. Only summaries travel to `capture_device`.
* **Separate what from how.** `tensors.py` decides what to keep, `raster.py` / `artistic.py` how
  to draw it, `layout.py` where, and the renderers compose. Every stage is one raster; text and
  arrows stay vector.
* **Fail soft.** Unknown tensor kinds render generically, failed summaries become notes, failed
  explanations are skipped, and failed tracing falls back to execution order.
* **Deterministic.** No random sampling. PCA uses SVD with a sign convention, and ties are broken
  by index.

## Tests

`tests/` (pytest, CPU only):

| file | covers |
|---|---|
| `test_capture.py` | hook removal (also on exceptions), train/eval restore, untouched inputs, nested modules, tuple/dict outputs, multiple inputs, reused modules, memory limits |
| `test_stages_graph.py` | ResNet-18/50 stage choice, transformer block sampling, explicit selectors, U-Net skips, merges, FX failure fallback, FX = runtime topology |
| `test_tensors_render.py` | channel strategies, determinism, outlier-robust normalization, 3-D modes, patch grids, Swin layout, rollout, output files, all model families, output semantics |
| `test_interactive_animation.py` | HTML explorer content and self-containment, animation |
| `test_cinematic_movie.py` | explanations, receptive-field growth, cinematic on every family and theme, 16:9 wrapping, fixed channels / PCA basis, movies, sequences, large-output label compression, 3-D token grids |

The toy models used by the tests are in `tests/conftest.py`.
