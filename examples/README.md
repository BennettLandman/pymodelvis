# Examples

Run from the repository root. All outputs go to `examples/outputs/` (the committed files were
produced by these scripts).

| script | what | first run |
|---|---|---|
| `resnet.py` | ResNet-50 (real ImageNet weights) on the cat; `--style technical\|story\|cinematic` | downloads ~100 MB |
| `vit.py` | ViT-B/16 (real weights); `--model swin_t` | downloads ~350 MB, needs `timm` |
| `chest_xray.py` | TorchXRayVision DenseNet-121 on an NIH chest X-ray; `--movie` for the occlusion sweep | needs `torchxrayvision` |
| `unet.py` | 2-D U-Net trained on synthetic microscopy | uses cached `unet2d_200steps.pt` |
| `medical_3d.py` | 3-D U-Net on a synthetic MRI-like volume; `--mode volume\|ortho\|montage\|projection`, `--thick` | uses cached `unet3d_48_150steps.pt` |
| `multihead.py` | MRI + clinical vector → segmentation, lesion, brain age | uses cached `multihead_32_600steps.pt` |
| `movies.py` | `pan`, `vit`, `aging`, `all` movies | 3–8 min each |
| `transformer_3d.py` | UNETR / Swin UNETR on a synthetic whole head, sliding windows; `--movie`, `--flat` | trains 5–15 min on first run |
| `monai_bundle.py` | any MONAI bundle (default MASI UNesT); `--movie` | run `neural-flow fetch unest` first |
| `extras.py` | HTML explorer, 16:9 / light / SVG variants, light-up GIF | |
| `run_all.sh` | everything above | ~30 min |

Helpers: `real_models.py` (model loaders), `synthetic.py` (synthetic data), `_common.py`.

The same things are available without Python through the command line. See
[docs/cli.md](../docs/cli.md), e.g. `neural-flow demo cat` or
`neural-flow render resnet50 -i photo.jpg --style cinematic`.

Full descriptions: [docs/examples.md](../docs/examples.md).
