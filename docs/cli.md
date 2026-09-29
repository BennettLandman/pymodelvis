# Command-line guide

Installing the package (`pip install -e .`) adds the **`neural-flow`** command.
`python -m neural_flow` does the same thing. You don't need to write any Python to render figures
or movies.

```
neural-flow demo       render the built-in demos (no arguments needed)
neural-flow render     one figure for a model and an input
neural-flow movie      a movie over a sequence of inputs
neural-flow inspect    list the stages (and every module call) that would be shown
neural-flow fetch      download pretrained weights / bundles into the cache
neural-flow models     list model aliases and where weights are looked up
```

`neural-flow <command> --help` lists every option.

---

## 1. First run: the demos

```bash
neural-flow demo cat          # ResNet-50 looks at the bundled cat photo  → neural_flow_demos/demo_cat.png
neural-flow demo vit          # the same photo through ViT-B/16
neural-flow demo cxr          # TorchXRayVision DenseNet-121 on a public NIH chest X-ray
neural-flow demo technical    # the cat as a light, publication-style figure
neural-flow demo movie        # a short pan across the cat → demo_movie_pan.mp4
neural-flow demo all -o my_demos/
```

Pretrained weights are downloaded once into the cache (`~/.cache/neural_flow`, or
`$NEURAL_FLOW_HOME`). The ViT demo needs `timm` and the chest X-ray demo needs `torchxrayvision`
(`pip install -e ".[examples]"`).

## 2. A figure for your image

```bash
neural-flow render resnet50 -i photo.jpg -o flow.png                       # technical (default)
neural-flow render resnet50 -i photo.jpg -o flow.png --style cinematic     # black, with explanations
neural-flow render resnet50 -i photo.jpg -o flow.pdf --style cinematic --theme light --figsize 16 9
neural-flow render resnet50 -i photo.jpg -o flow.svg --style story         # teaching figure
neural-flow render resnet50 -i photo.jpg -o flow.png --html                # + interactive flow.html
```

The command prints the stage table and writes the file. Images are centre-cropped, resized and
normalized the way the model expects: ImageNet statistics for torchvision/timm, the TorchXRayVision
convention for `cxr`.

## 3. Choosing a model

| you have | model argument |
|---|---|
| a torchvision model | `torchvision:efficientnet_b0`, `torchvision:resnet50:IMAGENET1K_V1` (weights name optional) |
| a timm model | `timm:convnext_tiny` |
| a chest X-ray model | `cxr` or `xrv:densenet121-res224-chex` |
| a MONAI bundle | `monai:path/to/bundle` or `unest` (after `neural-flow fetch unest`) |
| your own class in a file | `my_net.py:UNet --model-args '{"in_ch": 1, "n_classes": 3}' --weights ckpt.pt` |
| your own class in a package | `mypkg.models:build_model --weights ckpt.pt` |
| a whole saved model | `model.pt` (saved with `torch.save(model, "model.pt")`) |

Built-in aliases: `resnet50`, `resnet18`, `vit`, `vit_b_16`, `swin_t`, `densenet121`, `cxr`,
`unest` (`neural-flow models` lists them).

`--weights` accepts plain state dicts and checkpoints that wrap one (`{"state_dict": …}`,
`{"model": …}`); a `module.` prefix from DataParallel is removed automatically. If weights cannot
be downloaded the command stops; add `--allow-random-weights` to continue anyway.

`--device cuda` (or `mps`) runs the model on a GPU.

## 4. Choosing the input

| input | example |
|---|---|
| image (png, jpg, tif, …) | `-i photo.jpg` |
| 3-D volume (NIfTI) | `-i scan.nii.gz --crop 96` (96³ ROI at the centre of mass; MONAI bundles use their own ROI and preprocessing) |
| ready tensor | `-i x.npy` or `-i x.pt` |
| random smoke test | `-i random:1,3,224,224` |
| bundled samples | `-i sample:cat`, `-i sample:cxr` |
| several inputs | `-i image=scan.png -i clinical=features.npy` (names must match the model's `forward` arguments) |

Preprocessing overrides: `--size 256`, `--no-center-crop`, `--preset imagenet|raw|xray|volume`,
`--mean 0.5 0.5 0.5 --std 0.5 0.5 0.5`.

## 5. Finding and choosing stages

```bash
neural-flow inspect resnet50 -i sample:cat                    # stages that would be drawn
neural-flow inspect my_net.py:Net -i random:1,1,64,64,64 --all-modules   # every module call + shape
```

Then pick stages explicitly:

```bash
neural-flow render resnet50 -i photo.jpg --layers conv1,layer2,layer4,fc
neural-flow render my_net.py:Net -i x.npy --layers "re:^encoder\.\d+$,bottleneck,head"
neural-flow render my_net.py:Net -i x.npy --layers "type:Conv3d" --exclude "type:Upsample"
neural-flow render resnet50 -i photo.jpg --max-stages 5
```

## 6. Outputs and labels

```bash
--class-names imagenet                 # or labels.txt (one per line), labels.json, or "cat,dog,bird"
--output-type output=segmentation      # softmax | sigmoid | multilabel | multilabel_probs | regression | segmentation | embedding | raw
--output-type age=regression --output-type lesion=sigmoid     # multi-head models (dict keys)
--top-k 5
```

Without these, outputs are only shown as probabilities when the model makes that clear; otherwise
raw values are shown.

## 7. Appearance

| option | values |
|---|---|
| `--style` | `technical` (default for render), `story`, `cinematic` (default for movie) |
| `--theme` | `light`, `dark`, `black` (cinematic defaults to black) |
| `--figsize W H`, `--dpi N` | canvas size in inches, resolution |
| `--layout` | `horizontal`, `vertical`, `wrap` |
| `--title`, `--subtitle`, `--font-scale` | text |
| `--channel-strategy` | `energy`, `variance`, `spread`, `mean_abs`, `even`, `pca` |
| `--max-channels N` | channels kept per stage |
| `--front-page` | `pca` (all channels as colour) or `channel` (strongest channel), cinematic |
| `--volume-mode`, `--volume-axes` | 3-D: `volume`/`ortho`/`montage`/`projection`; `xyz`/`dhw` |
| `--explain` / `--no-explain` | gradient explanations (on by default for cinematic) |
| `--set KEY=VALUE` | any other option from the [API reference](api.md), e.g. `--set max_capture_mb=200` |

## 8. Movies

```bash
neural-flow movie resnet50 --pan panorama.jpg -o pan.mp4                  # camera pan (wide image)
neural-flow movie resnet50 --zoom photo.jpg --zoom-center 0.4 0.5 -o zoom.mp4
neural-flow movie cxr --occlusion sample:cxr --patch 56 --stride 28 -o occlusion.mp4
neural-flow movie resnet50 --crossfade cat.jpg dog.jpg -o morph.mp4
neural-flow movie my_net.py:Net --weights w.pt --frames "followup/*.nii.gz" --crop 96 -o timecourse.mp4
neural-flow movie unest --volume-sweep T1_mni.nii.gz --steps 24 -o sweep.mp4
```

Common options: `--steps N` (generated frames for pan / zoom / crossfade / volume sweep; the
occlusion grid is set by `--patch` and `--stride` instead), `--fps N`, `--hold N` (repeat last frame),
`-o file.mp4 | file.gif`, plus all appearance options. MP4 needs `imageio-ffmpeg`
(`pip install -e ".[animation]"`); otherwise a GIF is written. Stages, channels, colours and scales
are held fixed across frames (see [movies.md](movies.md)).

## 9. Real models and the cache

```bash
neural-flow fetch resnet50 vit cxr         # pre-download into ~/.cache/neural_flow
neural-flow fetch unest                    # MASI UNesT MONAI bundle + MNI152 T1 (needs monai, huggingface_hub, nilearn)
neural-flow render unest -i ~/.cache/neural_flow/mni152_t1_1mm.nii.gz --style cinematic -o unest.png
neural-flow models                         # aliases and the folders searched for weights
```

Weights are looked up in `./real_models/` (written by `tools/checkout_real_models.py`), then in
the package checkout's `real_models/`, then in the cache.

## 10. Troubleshooting

* `error: …` messages are short on purpose. Set `NEURAL_FLOW_DEBUG=1` for the full traceback.
* "could not download … weights": connect to the internet, run `neural-flow fetch …`, pass
  `--weights`, or use `--allow-random-weights` for a structural preview.
* "a 3-D model needs an input volume": pass `-i scan.nii.gz`.
* More in [troubleshooting.md](troubleshooting.md).

## Recipes

```bash
# poster figure, 16:9, light background, PDF
neural-flow render resnet50 -i photo.jpg --style cinematic --theme light --figsize 16 9 -o poster.pdf

# your segmentation U-Net on a NIfTI volume
neural-flow render models.py:UNet3D --model-args '{"n_classes": 4}' --weights best.pt \
    -i sub-01_T1w.nii.gz --crop 96 --output-type output=segmentation --style cinematic -o unet3d.png

# a classifier with your own labels, explore interactively
neural-flow render net.py:Classifier --weights ckpt.pt -i image.png --class-names labels.txt --html -o cls.png

# which regions matter? occlusion movie
neural-flow movie net.py:Classifier --weights ckpt.pt --occlusion image.png --class-names labels.txt -o occ.mp4
```
