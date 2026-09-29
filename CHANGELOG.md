# Changelog

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
