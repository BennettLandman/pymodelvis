# Movies over changing inputs

`animate_inputs` renders the cinematic flow once per input and writes an MP4 (or GIF). The point
of a movie is comparison across frames, so everything that should **not** change is held fixed:

| held fixed across frames | how |
|---|---|
| stages | selected on the first frame, then passed as explicit `layers` |
| channels on every page | ranked by activity summed over **all** frames (`force_channels`) |
| per-channel display range | 1st–99.5th percentile over all frames |
| PCA colour basis of the front pages | fitted on the middle frame and reused (`force_pca`) |
| latent-vector colour range | percentiles over all frames |

These are recomputed for every frame: activations, beams, receptive-field circles, contribution
lines, Grad-CAM evidence and outputs.

Under the flow, a **timeline** tracks every output (top classes, sigmoid probabilities, regressed
values, segmented volume %), and a **film strip** of the inputs marks the current frame.

## Usage

```python
from neural_flow import animate_inputs

animate_inputs(
    model,
    frames,                         # tensor [T, ...] or list of per-frame inputs (tensors or dicts)
    output="sweep.mp4",             # .mp4 (imageio-ffmpeg) or .gif
    fps=8,
    frame_labels=[...],             # optional text per frame, e.g. "true age 64"
    title="…", subtitle="…",
    class_names=labels,             # any visualize_model option
    figsize=(16, 9), dpi=120,
)
```

## Input sequences

```python
from neural_flow.sequences import pan, crossfade, zoom, occlusion_sweep, slices_to_frames

pan(img_chw, window=256, steps=48, out_size=224)       # camera pan across a wide image
crossfade(img_a, img_b, steps=32)                       # morph between two inputs
zoom(img_chw, start=1.0, end=0.3, center=(0.4, 0.6))   # zoom into a point
occlusion_sweep(img_chw, patch=56, stride=28)           # does the decision need this region?
slices_to_frames(volume, axis=-1)                       # 2-D model applied through a volume
```

Any list of inputs works, for example longitudinal scans of one subject, a dose-response series,
augmentations, or adversarial steps.

## Movies that make sense

A movie is most informative when the expected behaviour is known in advance. Then it tests the
model rather than decorating it:

* **Camera pan** (`movie_pan_resnet.mp4`): the predicted class should switch exactly when the
  window moves from one photograph to the next.
* **Ageing subject** (`movie_aging.mp4`): ventricles enlarge with age while a lesion appears and
  grows. Predicted brain age should rise (it goes 29 → 72), lesion probability should switch on,
  and the segmented volume should grow.
* **Occlusion sweep** (`movie_cxr_occlusion.mp4`): Cardiomegaly should drop when the heart is
  covered. It does, and the model then prefers Hernia.
* **3-D inference** (`movie_sliding_window_unetr.mp4`): a 3-D segmentation network sees one window
  at a time. Each frame is one sliding window; the stages follow the window through the head while
  the fused whole-head segmentation assembles in the output card, and the film strip shows where the
  window is. `sliding_window_movie(...)`, `neural-flow movie MODEL --inference scan.nii.gz`, or
  `python examples/transformer_3d.py --movie`. See [3-D models](volumes_3d.md). The same with a real
  nnU-Net on a CT: `movie_sliding_window_totalseg_organs.mp4`
  (`neural-flow movie totalseg-organs --inference sample:ct`, see [nnU-Net](nnunet.md)).

## Cost

Each frame needs three forward passes (ranking, capture, gradients) plus rendering. On a laptop
CPU that is roughly 2–5 s per frame for 2-D models and 5–10 s for small 3-D models. A 48-frame
movie takes a few minutes.
