# Changelog

## 0.2.0 (2026-09-29)

3-D transformer networks and whole-volume inference.

* **Transformer U-Nets as they are.** UNETR, Swin UNETR, UNesT and similar networks are recognised
  from the runtime dataflow: the transformer's tapped blocks / levels become stages (patch embedding →
  transformer encoder → bottleneck → decoder → head), projection branches become skip bridges, and
  the figure is drawn as a U. Previously the backbone was drawn as one box (UNETR) or left out
  (Swin UNETR).
* **Voxel spacing.** `voxel_spacing=(sx, sy, sz)`; the command line reads it from the NIfTI header.
  Volumes, slices and segmentations are drawn to physical scale.
* **Whole-volume output.** `sliding_window=True, roi_size=…`: the stages show the one window the
  network processed, the output card shows the prediction fused from every window (Gaussian
  weighting; memory-bounded for many-class models), with the traced window outlined.
* **3-D inference movie.** `sliding_window_movie(...)` / `neural-flow movie MODEL --inference scan.nii.gz`.
* **Optional flat view.** `flat_3d=True` draws 3-D stages as squashed 2-D projections (off by default).
* Many-label segmentations (e.g. 133 brain structures) get distinct colours, and large structures are
  drawn as glass so small, deep ones stay visible.
* `neural-flow fetch unest` falls back to the MONAI model-zoo sources on GitHub plus the NVIDIA weight
  URL when Hugging Face / NGC are unreachable; bundle structure names are used as class names.
* New example `examples/transformer_3d.py` (UNETR and Swin UNETR trained on a synthetic whole-head
  phantom with five structures), `medical_3d.py --thick`, rewritten `monai_bundle.py`.
* New docs: [3-D models](docs/volumes_3d.md), [About](docs/about.md); references for every demo model.
* Fix: `animate_inputs(..., explain=...)` raised a duplicate-keyword error.

## 0.1.0 (2026-09-29)

First public release.

* `visualize_model`, `trace_model`, `draw`: representation-flow figures for arbitrary PyTorch models
  (CNN, U-Net, ViT/Swin, 3-D medical, multi-input / multi-head).
* Automatic stage selection with explicit overrides (names, regex, type, predicate).
* Runtime dataflow topology (skips, merges, branches) with `torch.fx` and execution-order fallbacks.
* Memory-aware, on-device activation reduction; deterministic channel ranking and PCA.
* Styles: `technical`, `story`, `cinematic` (black background, feature-map stacks, PCA front
  pages, beams, receptive-field circles, contribution lines, Grad-CAM); themes `light`, `dark`,
  `black`.
* First-class 3-D volumes: cut-away anatomy, voxel blocks, activation-driven orthogonal slices,
  projections, 3-D segmentation renders.
* Attention capture (MHA / SDPA) and attention rollout.
* Interactive HTML explorer; "light-up" animation; movies over changing inputs (`animate_inputs`)
  with fixed channels, colours and scales; input-sequence generators.
* `neural-flow` command-line tool: `demo`, `render`, `movie`, `inspect`, `fetch`, `models`.
* Examples with real pretrained models (ResNet-50, ViT-B/16, TorchXRayVision DenseNet-121), demo
  models trained on synthetic data, a MONAI-bundle runner (MASI UNesT), and an example slide deck.
