# User guide

This guide explains the concepts behind `neural_flow` and shows the common recipes. For every
option see the [API reference](api.md).

- [1. The basic call](#1-the-basic-call)
- [2. Styles and themes](#2-styles-and-themes)
- [3. Which stages are shown](#3-which-stages-are-shown)
- [4. Inputs](#4-inputs)
- [5. Outputs and heads](#5-outputs-and-heads)
- [6. How activations are drawn](#6-how-activations-are-drawn)
- [7. Explanations: beams, circles, lines, evidence](#7-explanations)
- [8. 3-D medical volumes](#8-3-d-medical-volumes)
- [9. Transformers](#9-transformers)
- [10. Layout and figure size](#10-layout-and-figure-size)
- [11. Output formats](#11-output-formats)
- [12. Movies](#12-movies)
- [13. Working with the result object](#13-working-with-the-result-object)
- [14. Memory and large models](#14-memory-and-large-models)

## 1. The basic call

```python
from neural_flow import visualize_model

fig = visualize_model(model, x, output="flow.png")
```

* `model`: any `torch.nn.Module`. It is put into `eval()` for the duration of the call and
  restored afterwards. Parameters are never changed.
* `x`: a tensor, a tuple of tensors (positional inputs) or a dict (keyword inputs).
* `output`: optional path; the extension selects the format (`.png`, `.svg`, `.pdf`, `.html`).
* The return value is a matplotlib `Figure`. The computed result is attached as `fig.flow`, so
  `fig.flow.summary_table()` prints the chosen stages.

Use a **real, correctly normalized input**. Random noise gives random-looking activations.

## 2. Styles and themes

| `style=` | intended for | look |
|---|---|---|
| `"technical"` (default) | papers, debugging | white background; channel mosaics; module names, types and exact shapes; skip arcs; attention insets |
| `"story"` | teaching | 5–8 conceptual stages (LOW-LEVEL FEATURES, ENCODER, LATENT SPACE, …) with grouped labels |
| `"cinematic"` | talks, posters, movies | black background; stacks of feature maps with PCA front pages; beams, receptive-field circles, contribution lines, Grad-CAM |

`theme="light" | "dark" | "black"` works with every style. `style="cinematic"` defaults to
`"black"`.

```python
visualize_model(model, x, style="story", output="teaching.png")
visualize_model(model, x, style="cinematic", theme="light", output="poster.pdf")
```

## 3. Which stages are shown

By default the stages are chosen automatically (`layer_selection="auto"`). Activation functions,
normalization, dropout, identity and reshape modules are never shown. The algorithm expands
containers that reveal new tensor shapes, samples long runs of identical blocks (for example, 12
transformer blocks become 6), and absorbs pooling / up-sampling connectors into the next block.
See [stage_selection.md](stage_selection.md).

Control it with:

```python
visualize_model(model, x, max_stages=6)                          # fewer, coarser stages
visualize_model(model, x, target_stages=10)                      # prefer more detail
visualize_model(model, x, layers=["encoder.stage1", "encoder.stage2", "bottleneck", "head"])
visualize_model(model, x, layers=["re:^blocks\\.\\d+$"])         # regular expression on module names
visualize_model(model, x, layers=["type:Conv3d"])                # by module type
visualize_model(model, x, layers=["conv#2"])                     # third call of a reused module
visualize_model(model, x, layer_selection=lambda c: c.type_name.endswith("Block"))
visualize_model(model, x, exclude=["type:Upsample"])
visualize_model(model, x, labels={"encoder.stage1": "stem"})     # rename stages in the figure
```

## 4. Inputs

```python
visualize_model(model, image)                                    # single tensor
visualize_model(model, (image, mask))                            # positional arguments
visualize_model(model, {"image": img, "clinical": tab})          # keyword arguments (or one dict argument)
visualize_model(model, x, input_names=["T1w"])                   # rename inputs in the figure
visualize_model(model, batch, batch_index=3)                     # show sample 3 of a batch
```

Inputs are moved to the model's device automatically. The input tensors you pass in are never
modified.

## 5. Outputs and heads

Tuple, list, dict, namedtuple and dataclass outputs are supported, and every output leaf gets its
own card. Output meaning is inferred **conservatively**:

* A `[K]` vector is shown as a softmax classification only if its values already sum to 1, you
  gave `class_names`, or the head's name suggests classification (`fc`, `head`, `classifier`,
  `logits`, …). Otherwise the raw values are shown and labelled as such.
* A single value is shown as a sigmoid probability when its name suggests one (`prob`, `risk`,
  `present`, …). Otherwise it is shown as a regressed value.
* A spatial output at input resolution, or named `seg`/`mask`, is shown as a segmentation,
  overlaid on the input.

Be explicit when needed:

```python
visualize_model(model, x, class_names=imagenet_labels, top_k=5)
visualize_model(model, x, class_names={"diagnosis": ["CN", "MCI", "AD"]})
visualize_model(model, x, output_types={"lesion": "sigmoid", "age": "regression", "output": "segmentation"})
visualize_model(model, x, output_types={"output": "multilabel_probs"})   # already-calibrated multi-label probabilities
```

Available types: `softmax`, `sigmoid`, `multilabel` (applies a sigmoid per label),
`multilabel_probs`, `regression`, `segmentation`, `embedding`, `raw`. For anything else, pass
`output_interpreter(name, tensor)` returning a string, a dict (`headline`, `subline`, `items`) or
an `OutputView`.

## 6. How activations are drawn

* **2-D feature maps** `[C, H, W]`: the channels with the highest `channel_strategy` score
  (`energy` default; `variance`, `spread`, `mean_abs`, `even`, `pca`). The number of tiles grows
  with channel depth, and panels shrink as resolution drops.
* **Vectors** `[F]`: strips (technical) or dot matrices (cinematic).
* **Tokens** `[N, F]`: a CLS token is split off and patch tokens are reshaped to their grid.
* **3-D volumes** `[C, X, Y, Z]`: see section 8.
* Each map is normalized to its own 1st–99th percentile (`normalize="stage"` shares one range
  across a stage). `diverging=True` uses a zero-centred colour map for signed activations.

Details: [tensor_rendering.md](tensor_rendering.md).

## 7. Explanations

The cinematic style runs one gradient pass and draws the results. With another style, `explain=True`
still computes them (available as `fig.flow.explanation`), but only the cinematic renderer draws
them.

What is drawn:

* **Beams**: the region of the previous stage that feeds each stage's strongest unit. A narrow
  beam means a local window; scattered rays mean global mixing (pooling, attention).
* **Circles**: what that unit sees in the input, i.e. its effective receptive field from
  |∂unit/∂input|.
* **Lines into the head**: the largest weight × activation contributions to the predicted output
  (amber pushes it up, blue pushes it down).
* **Evidence**: Grad-CAM for the predicted class.

Exact definitions: [cinematic.md](cinematic.md).

## 8. 3-D medical volumes

`[B, C, X, Y, Z]` tensors are first-class. The default axis convention is nibabel/MONAI (`X`
left→right, `Y` posterior→anterior, `Z` inferior→superior). Use `volume_axes="dhw"` for
`[D, H, W]` slice stacks.

* **Input**: a cut-away anatomical render plus axial, coronal and sagittal slices through the
  centre of mass.
* **Feature volumes** (`volume_mode="volume"`, default): solid voxels for the strongest
  activations inside a translucent block. Small volumes show discrete cubes. Slices go through the
  activation peak.
* Alternatives: `volume_mode="ortho" | "montage" | "projection"` (with `projection="max" |
  "mean" | "meanabs"`) and `volume_style="voxels" | "cutaway" | "glow"`.
* Segmentation outputs render as glass anatomy with opaque labels. Dense 3-D outputs with many
  classes are stored as argmax labels to save memory.

```python
visualize_model(unet3d, vol, style="cinematic", output_types={"output": "segmentation"})
visualize_model(unet3d, vol, volume_mode="projection", projection="max")
```

## 9. Transformers

* Token tensors `[B, N, F]` are reshaped to their patch grid. CLS or register tokens are detected
  (override with `cls_tokens=` / `patch_grid=`). 3-D token grids (`N = m³`) become volumes.
* Channel-last Swin tensors `[B, H, W, C]` are detected (override with `channels_last=`).
* Attention weights are captured without modifying the model, from
  `F.multi_head_attention_forward` and `F.scaled_dot_product_attention`. They appear as insets
  (technical style) and in the HTML explorer, with attention rollout on the CLS representation.
* The transformer adapter ranks channels by `spread` (p90 − p10), because transformers carry a
  few near-constant "massive" channels.

## 10. Layout and figure size

* `layout="horizontal"` (default), `"vertical"`, or `"wrap"` (wrap long chains into rows sized
  for a 16:9 canvas).
* `figsize=(16, 9)` fixes the canvas; long chains wrap into as many rows as make the figure
  largest.
* Encoder/decoder networks with resolution-matched skip connections get a U-shaped layout
  automatically (`unet_layout=False` disables it).
* Branches fan out and merges converge; side inputs sit next to their fusion stage.
* `dpi`, `font_scale`, `title`, `subtitle`, `cmap` and `edge_width` fine-tune the appearance.

## 11. Output formats

| extension | notes |
|---|---|
| `.png` | raster at `dpi` (default 200) |
| `.svg`, `.pdf` | labels, arrows and outlines stay vector; activation images are embedded crisp (nearest-neighbour) |
| `.html` | self-contained interactive explorer: click a stage, switch channel-ranking strategy, expand single channels, inspect statistics, browse attention heads |

`visualize_model(model, x, output="fig.png", interactive=True)` writes both `fig.png` and
`fig.html`.

## 12. Movies

```python
from neural_flow import animate_inputs, animate_model
from neural_flow.sequences import pan, crossfade, zoom, occlusion_sweep, slices_to_frames

animate_inputs(model, pan(img, window=256, steps=48, out_size=224), output="pan.mp4", class_names=labels)
animate_inputs(model, [{"mri": v, "clinical": c} for v, c in pairs], output="sweep.mp4",
               frame_labels=[f"t={t}" for t in range(len(pairs))])
animate_model(model, x, output="lightup.gif")      # stages light up one after another, for a single input
```

See [movies.md](movies.md) for what is held fixed across frames and how to make your own sequences.

## 13. Working with the result object

```python
from neural_flow import trace_model, draw

res = trace_model(model, x, style="cinematic")    # compute once, no drawing
print(res.summary_table())
for st in res.stages:                             # input, module and representation stages
    print(st.key, st.kind, st.summary.shape if st.summary else None)
res.graph.edges                                   # typed edges: main / skip / merge
res.views                                         # interpreted outputs
res.explanation                                   # receptive fields, Grad-CAM, contributions (if computed)

ff = draw(res)                                    # FlowFigure (fig, ax, per-stage artists)
ff.fig.savefig("again.png")
```

## 14. Memory and large models

* Activations are reduced **on their own device**, channel-chunk by channel-chunk: per-channel
  statistics, a few representative channels at reduced resolution, PCA maps and energy maps. A
  `[1, 1024, 128, 128, 128]` activation is never copied wholesale.
* `max_capture_mb` (default 500) bounds everything that is retained. `max_spatial_2d` /
  `max_spatial_3d` cap the display resolution. Reduced stages are marked `≈`, and a warning is
  issued.
* `explain=False` skips the gradient pass, which matters for very large 3-D networks.
