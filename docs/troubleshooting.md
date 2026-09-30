# Troubleshooting

**The activations look like noise.**
Use a real input with the model's normalization (e.g. ImageNet mean/std). Random tensors or
unnormalized 0–255 images give meaningless activations. Also check that the weights actually
loaded (`load_state_dict` reports missing keys).

**Too many or too few stages / the wrong ones.**
`max_stages=6` or `target_stages=10` changes the granularity. For full control pass
`layers=[...]` (names, `"re:…"`, `"type:…"`) and `labels={...}`. `fig.flow.trace.calls_in_order()`
lists every module call with its shapes.

**"model could not be symbolically traced" warning.**
It only appears with `topology="fx"`. The default runtime tracing handles data-dependent control
flow, so use `topology="auto"`.

**Skip connections are missing or the graph is a plain chain.**
The model probably bypasses `__torch_function__` (TorchScript, `torch.compile`, custom CUDA ops).
Topology then falls back to execution order. Run the uncompiled model.

**A classification output is shown as "raw values".**
This is intentional when semantics are unknown. Pass `class_names=[...]` or
`output_types={"output": "softmax"}`.

**My segmentation isn't recognized.**
Pass `output_types={"output": "segmentation"}` (or the name of the dict key).

**3-D volumes look rotated or mirrored.**
The default axis order is nibabel/MONAI `[X, Y, Z]`. For torch-style slice stacks `[D, H, W]`
use `volume_axes="dhw"`.

**Out of memory on large 3-D models.**
Gradient explanations need the most memory (a 96³ UNesT window takes 6–8 GB of RAM on a CPU with
them). Set `explain=False` (`--no-explain`), lower `max_capture_mb` and `max_spatial_3d`, and trace one
window (`--crop 96`, or `--sliding-window`, which traces one window and fuses the rest without
gradients). For many-class models, lower `sw_max_mb` so fused logits are accumulated on a coarser grid.

**A 3-D transformer (UNETR, Swin UNETR, UNesT …) is drawn as one box.**
The transformer U-Net adapter needs the runtime dataflow (the default `topology="auto"`). Check with
`neural-flow inspect MODEL -i INPUT`: the adapters line should list `transformer-unet`. If your model
routes hidden states in an unusual way, choose the stages with `--layers`.

**The whole-volume output card shows only one window.**
Pass the *whole* volume with `sliding_window=True` and `roi_size`; on the command line use
`--sliding-window` (it disables the ROI crop). Without it, only the traced window is shown.

**3-D volumes look squashed.**
Set the voxel spacing (`voxel_spacing=(sx, sy, sz)`); the command line reads it from the NIfTI header
unless you pass `--no-spacing`.

**MP4 is written as GIF.**
Install `imageio-ffmpeg` (`pip install -e ".[animation]"`).

**Fonts look different from the gallery.**
The cinematic style registers the bundled Inter font automatically. The technical style uses
matplotlib's default font (DejaVu Sans).

**Tests are slow or fail to import torchvision.**
Install the dev extra: `pip install -e ".[dev]"`. The tests are CPU-only and take a few minutes (the MONAI 3-D tests are skipped without `monai`).
