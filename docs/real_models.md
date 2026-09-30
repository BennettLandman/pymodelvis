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
neural-flow fetch unest                                 # bundle + MNI152 T1 template into ~/.cache/neural_flow
python examples/monai_bundle.py                         # default: UNesT whole-brain, cinematic figure
python examples/monai_bundle.py --movie                 # sliding-window inference over the whole head
python examples/monai_bundle.py --image my_t1.nii.gz    # a T1 registered to MNI space
python examples/monai_bundle.py --bundle-dir path/to/any_bundle --roi 96
python examples/monai_bundle.py --no-explain            # skip gradients (faster, much less memory)
```

or, without Python:

```bash
neural-flow render unest -i ~/.cache/neural_flow/mni152_t1_1mm.nii.gz --sliding-window --style cinematic
neural-flow movie unest --inference ~/.cache/neural_flow/mni152_t1_1mm.nii.gz --max-windows 16
```

`neural-flow fetch unest` tries Hugging Face (`MONAI/wholeBrainSeg_Large_UNEST_segmentation`) first,
then the bundle's sources from the MONAI model-zoo repository on GitHub plus the weight file listed in
its `large_files.yml` (an NVIDIA download, checked against its MD5), then `monai.bundle.download`.

The loader:

1. reads the bundle's `configs/inference.json` (or `.yaml`) and instantiates `network_def`;
2. loads `models/model.pt` (plain state dicts and `{"model": …}` / `{"state_dict": …}` checkpoints);
3. runs the bundle's own `preprocessing` transform when it can, or falls back to z-scoring;
4. takes the window size from the bundle's `inferer.roi_size` and the 133 structure names from
   `metadata.json`;
5. traces one window at the head's centre and fuses the whole-brain output from every window
   (`sliding_window=True`), drawn to scale from the NIfTI voxel spacing.

UNesT expects T1-weighted MRI affinely registered to MNI space. The MNI152 template is a convenient
public input; your own registered T1 gives a more interesting figure.

Memory: the 133-channel output of a whole head would need several GB, so fused logits are accumulated
on a coarser grid (`sw_max_mb`). With explanations on, a 96³ UNesT window needs about 6–8 GB of RAM on
a CPU; use `--no-explain` on smaller machines.

> Status: the UNesT *architecture* (from the bundle's own sources) is tested here: its patch
> embedding, three NesT levels, bottleneck and decoder are recognised and drawn as a U, with
> sliding-window output over the MNI152 template. The trained *weights* could not be downloaded in the
> development sandbox (the NVIDIA, Hugging Face and NGC hosts are blocked there), so the first run with
> real weights happens on your machine.

## nnU-Net models and TotalSegmentator

```bash
neural-flow fetch totalseg totalseg-organs            # TotalSegmentator weights (GitHub releases) + example CT
python examples/nnunet_totalseg.py                    # both figures
python examples/nnunet_totalseg.py --model totalseg-organs --movie
neural-flow render nnunet:path/to/results_folder -i case.nii.gz --sliding-window --style cinematic
```

Any trained nnU-Net v2 model loads from its results folder (`plans.json`, `dataset.json`,
`fold_N/checkpoint_final.pth`) with nnU-Net's preprocessing, patch size, spacing and window overlap;
nnU-Net itself is not needed, only `dynamic-network-architectures`. TotalSegmentator's models are such
folders; `neural-flow fetch` downloads them from the project's GitHub releases (Apache-2.0) into
`~/.cache/neural_flow/nnunet/`.

> Status: tested here with the real TotalSegmentator weights (3 mm total model and 1.5 mm organ model)
> on TotalSegmentator's example CT. The 3 mm result matches TotalSegmentator's own reference
> segmentation (Dice 0.95–0.98 on the major organs). Details: [nnU-Net and TotalSegmentator](nnunet.md).

## Your own model

Nothing here is special to these models. Any `nn.Module` works:

```python
model = MyNet(); model.load_state_dict(torch.load("weights.pt")); model.eval()
visualize_model(model, x, style="cinematic", class_names=[...], output="mynet.png")
```
