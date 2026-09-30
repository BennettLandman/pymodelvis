# Examples

Every script is run from the repository root (`python examples/<script>.py`) and writes to
`examples/outputs/`. The committed outputs were produced by exactly these scripts.
`bash examples/run_all.sh` regenerates all of them.

Weights:

* **Real pretrained** weights are downloaded on first use from public GitHub releases into
  `real_models/` (git-ignored), or taken from there if `tools/checkout_real_models.py` already
  fetched them.
* **Demo models** (U-Net, 3-D U-Net, multi-head) are trained on built-in synthetic data. Their
  weights are cached in `examples/outputs/*.pt`, so re-running only re-renders.

---

## 1. ResNet-50 looks at a cat (`resnet.py`)

![](../examples/outputs/resnet50_cinematic.png)

ResNet-50 with ImageNet weights on the scikit-image "chelsea" photograph (CC0). The cat is
classified "Egyptian cat" (p = 0.44; tabby and tiger cat follow).

```bash
python examples/resnet.py                                   # cinematic, black
python examples/resnet.py --style technical                 # publication figure
python examples/resnet.py --style story                     # teaching figure
python examples/resnet.py --image my_photo.jpg --figsize 16 9 --format pdf
python examples/resnet.py --front channel                   # front page = strongest channel instead of PCA
```

What to look for: receptive fields grow 8 → 22 → 37 → 68 → 224 px. The PCA front pages of
layer3/layer4 separate the face, and the Grad-CAM evidence sits on the face.

## 2. Vision Transformer (`vit.py`)

![](../examples/outputs/vit_b_16_cinematic.png)

timm `vit_base_patch16_224` (ImageNet-1k), the same cat, p = 0.98. The 197 tokens are split into
CLS + a 14 × 14 grid, and 6 of 12 blocks are sampled. Beams become scattered rays and receptive
fields are nearly global by block 4, in contrast with the CNN.

```bash
python examples/vit.py                  # cinematic
python examples/vit.py --style technical   # attention insets (CLS → patch)
python examples/vit.py --model swin_t   # torchvision Swin-T (needs torchvision weights)
```

## 3. Chest X-ray (`chest_xray.py`)

![](../examples/outputs/chest_xray_cinematic.png)

TorchXRayVision DenseNet-121 (`densenet121-res224-all`, 18 pathologies, calibrated probabilities)
on NIH ChestX-ray14 image `00000001_000.png` (labelled cardiomegaly). Cardiomegaly is the top
prediction (p = 0.62). Grad-CAM concentrates on the cardiac silhouette, and the contribution
lines show latent units for and against.

```bash
python examples/chest_xray.py                 # static figure
python examples/chest_xray.py --theme light
python examples/chest_xray.py --movie         # occlusion sweep → movie_cxr_occlusion.mp4
python examples/chest_xray.py --image my_cxr.png
```

## 4. 2-D U-Net (`unet.py`)

![](../examples/outputs/unet2d_cinematic.png)

A small U-Net trained for 200 steps on synthetic microscopy (bright round "cells" and elongated
"debris"). Skip connections are recovered from runtime dataflow, and the layout dips by
resolution level so skips become bridges.

```bash
python examples/unet.py                      # technical
python examples/unet.py --style cinematic
python examples/unet.py --style story
```

## 5. 3-D U-Net on an MRI-like volume (`medical_3d.py`)

![](../examples/outputs/medical_3d_cinematic.png)

A 3-D U-Net trained on synthetic 48³ head phantoms (skull, brain, ventricles, lesion; Dice 0.89 on
the test volume). The input renders as cut-away anatomy with orthogonal slices, features as voxel
blocks, and the output as a glass brain with an opaque lesion.

```bash
python examples/medical_3d.py --style cinematic
python examples/medical_3d.py --mode projection      # or ortho, montage
python examples/medical_3d.py --style story
python examples/medical_3d.py --style cinematic --thick   # 1 × 1 × 3 mm voxels, drawn to scale
```

![](../examples/outputs/medical_3d_cinematic_thick.png)

## 6. Multi-input, multi-head (`multihead.py`)

![](../examples/outputs/multihead_cinematic.png)

MRI volume + clinical vector (sex, field strength, a nuisance variable) → shared 3-D encoder,
clinical MLP, fusion → three heads: lesion segmentation, lesion present (sigmoid), brain age
(regression). Trained 600 steps on synthetic data. Inputs and outputs are dicts.

```bash
python examples/multihead.py --style cinematic
```

## 7. 3-D transformer U-Nets: UNETR and Swin UNETR (`transformer_3d.py`)

![](../examples/outputs/transformer3d_unetr.png)

MONAI's UNETR and Swin UNETR, trained here on 64³ windows of a synthetic whole-head phantom with five
labelled structures (scalp / skull, cortex, white matter, ventricles, lesion; `synthetic.head_labels`).
The transformer blocks / levels are stages of their own, the decoder is drawn as a U with the
transformer taps as skip bridges, and the output card shows the whole head (96 × 112 × 96 voxels of
1.6 mm) fused from every sliding window, with the traced window outlined.

```bash
python examples/transformer_3d.py                          # UNETR (trains ~5 min on first run)
python examples/transformer_3d.py --model swinunetr        # Swin UNETR (~15 min on first run)
python examples/transformer_3d.py --flat                   # optional squashed 2-D view
python examples/transformer_3d.py --movie                  # sliding-window inference, window by window
```

![](../examples/outputs/transformer3d_swinunetr.png)

![](../examples/outputs/transformer3d_unetr_flat.png)

Models: UNETR (Hatamizadeh et al., WACV 2022) and Swin UNETR (Hatamizadeh et al., BrainLes 2021),
both from MONAI (Cardoso et al., 2022). See [3-D models](volumes_3d.md).

## 8. Movies (`movies.py`, `chest_xray.py --movie`, `transformer_3d.py --movie`)

```bash
python examples/movies.py pan      # ResNet-50 camera pan across four photos
python examples/movies.py vit      # the same pan through ViT-B/16
python examples/movies.py aging    # one synthetic subject ages; a lesion appears and grows
python examples/movies.py all --frames 48 --fps 8 [--gif]
python examples/transformer_3d.py --movie       # 3-D inference: UNETR segments a whole head window by window
```

See [movies.md](movies.md).

## 9. Any MONAI bundle / MASI UNesT (`monai_bundle.py`)

Builds the network, weights and preprocessing from a bundle's own `configs/inference.json`.
The default is MASI's UNesT whole-brain segmentation (133 structures; Yu et al., Medical Image
Analysis 2023). The NesT transformer levels become stages, and the output card shows the whole brain
fused from sliding windows. See [real_models.md](real_models.md).

```bash
neural-flow fetch unest                   # bundle sources from GitHub, weights from NVIDIA, MNI152 template
python examples/monai_bundle.py
python examples/monai_bundle.py --movie   # sliding-window inference over the whole head
```

## 10. nnU-Net: TotalSegmentator on a CT (`nnunet_totalseg.py`)

Real TotalSegmentator weights (nnU-Net v2 results folders) on TotalSegmentator's public example CT.
The network is rebuilt from `plans.json` and preprocessed exactly as nnU-Net does (RAS, CT clipping and
normalisation, resampling to the target spacing). Stages are the encoder levels, the bottleneck and
the decoder levels; the output card is the whole CT fused from sliding windows with the plans' patch
size and 50 % overlap.

![TotalSegmentator organs](../examples/outputs/nnunet_totalseg_organs.png)

* `nnunet_totalseg.png`: the fast 3 mm model, 117 structures, patch 112×112×128 (one window covers
  this small CT).
* `nnunet_totalseg_organs.png`: the 1.5 mm organ model, 24 structures, 128³ patches, 27 windows.
* `movie_sliding_window_totalseg_organs.mp4`: the organ model working through the CT window by window
  (all 27 windows are fused; 14 evenly spaced ones are shown, `--max-windows 0` shows all).

```bash
neural-flow fetch totalseg totalseg-organs
python examples/nnunet_totalseg.py
python examples/nnunet_totalseg.py --model totalseg-organs --movie
python examples/nnunet_totalseg.py --results path/to/your/results_folder --image case.nii.gz
```

See [nnU-Net and TotalSegmentator](nnunet.md).

## 11. Extras (`extras.py`)

Interactive HTML explorer (`resnet50_interactive.html`), a 16:9 cinematic figure, a light-theme
cinematic figure, an SVG, and the "light-up" GIF from `animate_model`.

## 12. The slide deck (`docs/deck/`)

[`neural_flow_deck.pptx`](deck/neural_flow_deck.pptx) is a 15-slide deck built entirely from these
outputs. It has title, scoping, design criteria, an examples summary, one slide per example, two
movie slides with embedded videos, user instructions and references. See
[deck/README.md](deck/README.md) to rebuild it.
