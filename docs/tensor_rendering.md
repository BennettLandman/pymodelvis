# How tensors become pictures

Two layers are involved. `tensors.py` **interprets and reduces** a tensor into a `TensorSummary`,
on the tensor's own device and under a byte budget. `raster.py` **renders** a summary into one RGBA
image (a `Visual`). `visualize_tensor(summary, cfg)` is the dispatcher.

## 1. Interpretation (`infer_kind`)

The batch item `batch_index` (default 0) is taken first. `[N, B, F]` sequence-first tensors are
detected from the known batch size.

| shape after batch | kind | notes |
|---|---|---|
| `[]`, `[B]` | scalar | |
| `[F]` | vector | also `[C,1,1]` / `[C,1,1,1]` (global pooling) |
| `[N, F]` | tokens | `seq1d` when produced by `Conv1d`/`*Pool1d`/`BatchNorm1d` |
| `[C, H, W]` | image2d | channels-last `[H, W, C]` detected (Swin); `attention` when square and named *attn* |
| `[C, X, Y, Z]` | volume3d | first-class 3-D (MRI/CT); channels-last optional |
| other | generic | flattened, binned strip |

## 2. Reduction (memory-aware, deterministic)

For `image2d` / `volume3d` (and token grids treated as `[F, gh, gw]`), everything is chunked over
channels to bound temporary memory:

* **Per-channel statistics**: mean, variance, mean |x|, energy (mean x²).
* **Channel selections**, one list per strategy, `max_channels` each:
  * `energy`: highest mean x² (default)
  * `variance`: highest spatial variance (default for 3-D; favours structure over constant offsets)
  * `mean_abs`: highest mean |x|
  * `even`: evenly spaced indices
  * `pca`: channels with the largest leverage on the first 3 principal components
* **PCA maps**: channels × positions at ≤ 32² (2-D) / 16³ (3-D) → SVD (float64, sign-fixed) →
  3 loadings. The full-resolution projection is accumulated chunk-wise into 3 spatial maps, which
  give PCA-RGB composites.
* **Retained maps**: the union of all strategies' channels, average-pooled to ≤ 128 px (2-D) or
  ≤ 48 voxels (3-D) and shrunk further until they fit the per-stage byte budget
  (`max_capture_mb / n_stages`).
* **Mean |x| and energy maps** over *all* channels at the reduced resolution. These drive
  activation-driven slice selection.
* **Robust statistics**: min/max plus p1/p99/median/fraction-negative on a strided sample of
  ≤ 2 M values.

If spatial resolution was reduced, `summary.reduced = True`, a note is recorded, the figure shows
`≈` before the shape, a footnote is added, and a warning is emitted.

Tokens: special tokens are split off using the patch-grid inference below. The `[N, F]` matrix is
pooled to ≤ 256 × 256 for heatmaps. Patch tokens → `[F, gh, gw]` → the spatial pipeline above,
plus an L2-norm map.

**Patch-grid inference:** try `N − k` for `k ∈ {0, 1}` as a square (or as a rectangle matching the
input aspect ratio), then `k ∈ {2, 4, 5}` (distillation / register tokens). Override with
`cls_tokens=` and `patch_grid=`.

## 3. Rendering

### 2-D feature maps: mosaics
* Tile count grows with channel depth, so depth increase is visible: C ≤ 4 → all channels,
  ≤ 64 → 2×2, 128–256 → 3×3, ≥ 512 → 4×4 (capped by `max_channels`).
* Each tile is normalised to its own p1–p99 range (`normalize="stage"` shares one range), so a
  single outlier cannot flatten a map. `diverging=True/"auto"` uses a zero-centred `RdBu_r` map.
* Tiles are nearest-upsampled (crisp activation pixels in PNG/SVG/PDF). Panel size shrinks
  logarithmically with spatial resolution, and offset "sheets" behind each mosaic grow with
  log₂(C). Together these show *resolution falling, depth rising*.

### Vectors: latent strips
A compact column-major band of coloured cells (1–8 columns). Longer vectors are binned to ≤ 512
cells. This gives the *spatial → latent → head* transition.

### Tokens
`token_mode="grid"` (default when a grid exists): the patch-token grid is rendered like a CNN
mosaic, with the CLS token as a thin strip on the left and a CLS→patch attention inset on the
right. `"pca"` shows a single PCA-RGB map, `"l2"`/`"mean"` a single reduced map, and `"heatmap"`
the token × feature matrix.

### Attention
Per-head weights `[h, N, N]` come from the capture pass. The views are the head-averaged matrix
(√-scaled), CLS→patch maps, mean entropy, and Abnar & Zuidema **attention rollout** (residual 0.5)
across all captured layers, shown on the CLS representation stage.

### 3-D volumes: `[B, C, X, Y, Z]` as first-class objects
Volumes are converted to canonical `V[x, y, z]` (x: left→right, y: posterior→anterior,
z: inferior→superior). `volume_axes="xyz"` covers nibabel/MONAI order; `"dhw"` covers torch slice
stacks.

* **Input anatomy:** an opaque **cut-away** render. Tissue is separated from background with an
  Otsu-style threshold, and the camera-facing octant at the centre of mass is removed, so the cut
  faces show internal anatomy. Below it are axial/coronal/sagittal slices through the same point.
* **Feature volumes** (`volume_style="voxels"`, default): voxels above the 82nd percentile
  (lightly smoothed) are solid and coloured by activation, inside a faint translucent block that
  shows the volume's extent. The octant at the activation peak is cut away. Volumes ≤ 12 voxels
  per side are nearest-upsampled with thin gaps, so they read as **discrete voxel cubes**. Moving
  down the encoder, the render therefore goes from anatomy-like surfaces to increasingly abstract
  voxel blocks.
* **Channel stacks:** 1–4 channel volumes (more with depth) drawn as an offset deck; back volumes
  are faded.
* **Activation-driven orthogonal slices** under each block go through the peak of the smoothed
  activation of the front channel, not the geometric centre.
* Other modes: `volume_mode="ortho"` (per-channel ortho triplets through each channel's peak),
  `"montage"` (the 4 axial slices with most activation), `"projection"` (max / mean / mean-|x|
  projections along each axis), and `volume_style="glow"` (translucent emission–absorption) or
  `"cutaway"` (opaque).
* The renderer is a ~100-line orthographic ray caster on `torch.nn.functional.grid_sample`: front-
  to-back compositing, opacity correction for step size, Lambert shading from alpha gradients, a
  depth cue, and a wireframe bounding box. It is CPU only and needs no extra dependencies.

### Outputs (heads)
Semantics are inferred **conservatively** (`outputs.py`):
* `[K]` becomes *classification* (softmax top-k) only if the values already sum to 1, class names
  were given, `output_types` says so, or the head/output name suggests classification
  (`fc`, `head`, `classifier`, `logits`, …). Otherwise raw values are shown with a note.
* A single value becomes a *sigmoid probability* when the name suggests it (`prob`, `malig`,
  `risk`, `present`, …); otherwise a *regression* value.
* A spatial output matching the input size (or named `seg`/`mask`) becomes *segmentation*: argmax
  labels, or sigmoid > 0.5 for one channel. It is overlaid on the input (2-D), or shown as a glass
  anatomy volume with opaque labels plus ortho overlays (3-D).
* `output_interpreter(name, tensor)` can return an `OutputView`, a dict, or a string to override
  any of this.
