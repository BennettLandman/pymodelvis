# Real models

## Downloaded automatically

`examples/real_models.py` loads these on first use and caches them in `real_models/`
(git-ignored):

| model | source | used by |
|---|---|---|
| ResNet-50 | torchvision `IMAGENET1K_V2` (if checked out) or timm `resnet50_ram-a26f946b.pth` (GitHub release) | `resnet.py`, `movies.py pan`, `extras.py` |
| ViT-B/16 | timm `jx_vit_base_p16_224` (GitHub release, JAX port of Google's weights) | `vit.py`, `movies.py vit` |
| DenseNet-121 chest X-ray | TorchXRayVision `densenet121-res224-all` | `chest_xray.py` |
| NIH ChestX-ray14 `00000001_000.png` | TorchXRayVision test image | `chest_xray.py` |

## `tools/checkout_real_models.py`

This script fetches models hosted where the example loaders don't reach (NGC / Hugging Face) plus
the MNI152 template:

```bash
python3 -m pip install torch torchvision "monai[nibabel]" einops nilearn huggingface_hub
python3 tools/checkout_real_models.py                    # everything (~0.5–1 GB)
python3 tools/checkout_real_models.py --only unest       # just the UNesT bundle + MNI152 T1
python3 tools/checkout_real_models.py --only unest --t1 /path/to/sub_T1w_MNI.nii.gz
```

It writes into `real_models/`:

* `monai_bundles/wholeBrainSeg_Large_UNEST_segmentation/`: the MONAI bundle (configs, weights)
* `mni152_t1_1mm.nii.gz`, `mni152_brainmask_1mm.nii.gz`: the input template
* `user_t1.nii.gz`: your own T1, if given
* `resnet50_imagenet1k_v2_fp16.pth`, `vit_imagenet1k_v1_fp16.pth`, `imagenet_classes.json`
* `manifest.json`: what was downloaded, sizes and versions

## MONAI bundles (`examples/monai_bundle.py`)

```bash
python examples/monai_bundle.py                         # default: UNesT whole-brain, cinematic figure
python examples/monai_bundle.py --movie --frames 24     # the ROI window sweeps inferior → superior
python examples/monai_bundle.py --image my_t1.nii.gz
python examples/monai_bundle.py --bundle-dir path/to/any_bundle --roi 96
python examples/monai_bundle.py --no-explain            # skip gradients (faster, less memory)
```

The script:

1. reads the bundle's `configs/inference.json` (or `.yaml`) and instantiates `network_def`;
2. loads `models/model.pt` (plain state dicts and `{"model": …}` / `{"state_dict": …}` checkpoints);
3. runs the bundle's own `preprocessing` transform when it can, or falls back to z-scoring;
4. crops an ROI of the bundle's `roi_size` around the head's centre of mass, because the network
   sees one sliding-window patch at a time;
5. renders with `style="cinematic"`, `output_types={"output": "segmentation"}`.

UNesT expects T1-weighted MRI registered to MNI space. The MNI152 template is a convenient
public input; your own registered T1 gives a more interesting figure. Large multi-class
outputs (133 labels × ROI³) are kept as argmax labels to save memory.

> Status: `monai_bundle.py` was tested with a stand-in bundle built from the same config format.
> The real UNesT bundle could not be downloaded in the development sandbox, so run it once on your
> machine and adjust `--roi` if the figure is too slow.

## Your own model

Nothing here is special to these models. Any `nn.Module` works:

```python
model = MyNet(); model.load_state_dict(torch.load("weights.pt")); model.eval()
visualize_model(model, x, style="cinematic", class_names=[...], output="mynet.png")
```
