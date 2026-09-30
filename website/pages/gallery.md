---
title: Gallery
description: Figures and movies made with neural_flow, from ImageNet photographs to chest X-rays and 3-D MRI volumes.
hide:
  - toc
---

# Gallery

Every figure below was made by the scripts in [`examples/`](https://github.com/MASILab/pymodelvis/tree/main/examples)
and is committed to the repository. Click a figure to enlarge it. How each was made and what to
look for: [Examples explained](examples.md).

## ResNet-50 looks at a cat

ImageNet weights, the scikit-image "chelsea" photograph (CC0).

<div class="nf-gallery-page" markdown>

![ResNet-50, cinematic](media/resnet50_cinematic.webp){ .nf-wide }

![ResNet-50, technical style](media/resnet50_technical.webp)
![ResNet-50, story style](media/resnet50_story.webp)
![ResNet-50, cinematic 16:9](media/resnet50_cinematic_169.webp)
![ResNet-50, cinematic light theme](media/resnet50_cinematic_light.webp)

</div>

```bash
neural-flow demo cat                                             # or: python examples/resnet.py
neural-flow render resnet50 -i photo.jpg --style technical -o flow.png
```

## Vision transformer (ViT-B/16)

<div class="nf-gallery-page" markdown>

![ViT-B/16, cinematic](media/vit_b_16_cinematic.webp){ .nf-wide }

![ViT-B/16, technical](media/vit_b_16_technical.webp)

</div>

```bash
neural-flow render vit -i photo.jpg --style cinematic -o vit.png   # or: python examples/vit.py
```

## Chest X-ray (TorchXRayVision DenseNet-121)

A public NIH ChestX-ray14 radiograph. 18 pathology outputs, shown as independent probabilities.

<div class="nf-gallery-page" markdown>

![Chest X-ray, cinematic](media/chest_xray_cinematic.webp){ .nf-wide }

![Chest X-ray, light theme](media/chest_xray_cinematic_light.webp){ .nf-wide }

</div>

```bash
neural-flow demo cxr                                              # or: python examples/chest_xray.py
```

## 2-D U-Net

Trained on synthetic microscopy images. Skip connections are drawn as bridges.

<div class="nf-gallery-page" markdown>

![2-D U-Net, cinematic](media/unet2d_cinematic.webp){ .nf-wide }

![2-D U-Net, technical](media/unet2d.webp)
![2-D U-Net, story](media/unet2d_story.webp)

</div>

## 3-D U-Net on an MRI-like volume

`[B, C, X, Y, Z]` as a first-class object: anatomy → translucent voxel blocks → a 3-D segmentation.

<div class="nf-gallery-page" markdown>

![3-D U-Net, cinematic](media/medical_3d_cinematic.webp){ .nf-wide }

![3-D U-Net, technical](media/medical_3d.webp)
![3-D U-Net, story](media/medical_3d_story.webp)
![3-D U-Net, projection mode](media/medical_3d_projection.webp){ .nf-wide }

</div>

```bash
python examples/medical_3d.py --mode volume|ortho|montage|projection
```

## 3-D transformer U-Nets on a whole head

MONAI UNETR and Swin UNETR, trained here on 64³ windows of a synthetic head with five structures. The
transformer levels are stages; the output card is the whole head fused from sliding windows.

<div class="nf-gallery-page" markdown>

![UNETR, cinematic](media/transformer3d_unetr.webp){ .nf-wide }

![Swin UNETR, cinematic](media/transformer3d_swinunetr.webp){ .nf-wide }

![UNETR, optional flat 2-D view](media/transformer3d_unetr_flat.webp){ .nf-wide }

![3-D U-Net on thick slices, drawn to scale](media/medical_3d_cinematic_thick.webp){ .nf-wide }

</div>

```bash
python examples/transformer_3d.py [--model swinunetr] [--flat] [--movie]
neural-flow render unest -i T1_mni.nii.gz --sliding-window --style cinematic      # MASI UNesT, real weights
```

See [3-D models](volumes_3d.md).

## Multi-input, multi-head

An MRI volume and a clinical vector → lesion segmentation, lesion probability and brain age.

<div class="nf-gallery-page" markdown>

![Multi-head, cinematic](media/multihead_cinematic.webp){ .nf-wide }

![Multi-head, technical](media/multihead.webp)
![Multi-head, story](media/multihead_story.webp)

</div>

## Movies

Stages, channels, colours and scales are fixed across frames. See [Movies](movies.md).

<div class="nf-gallery-page nf-gallery-page--movies" markdown>

<figure class="nf-movie" markdown>
<video muted loop playsinline controls preload="none" poster="../media/movie_pan_resnet_poster.webp"><source src="../media/movie_pan_resnet.mp4" type="video/mp4"></video>
<figcaption>Camera pan, ResNet-50</figcaption>
</figure>

<figure class="nf-movie" markdown>
<video muted loop playsinline controls preload="none" poster="../media/movie_pan_vit_poster.webp"><source src="../media/movie_pan_vit.mp4" type="video/mp4"></video>
<figcaption>Camera pan, ViT-B/16</figcaption>
</figure>

<figure class="nf-movie" markdown>
<video muted loop playsinline controls preload="none" poster="../media/movie_aging_poster.webp"><source src="../media/movie_aging.mp4" type="video/mp4"></video>
<figcaption>Ageing subject, multi-head 3-D network</figcaption>
</figure>

<figure class="nf-movie" markdown>
<video muted loop playsinline controls preload="none" poster="../media/movie_cxr_occlusion_poster.webp"><source src="../media/movie_cxr_occlusion.mp4" type="video/mp4"></video>
<figcaption>Occlusion sweep, chest X-ray</figcaption>
</figure>

<figure class="nf-movie" markdown>
<video muted loop playsinline controls preload="none" poster="../media/movie_sliding_window_unetr_poster.webp"><source src="../media/movie_sliding_window_unetr.mp4" type="video/mp4"></video>
<figcaption>3-D inference, UNETR, window by window</figcaption>
</figure>

</div>

## Interactive explorer

[Open the ResNet-50 explorer](media/explorer/resnet50.html){ .md-button .md-button--primary }
&nbsp; made with `neural-flow render resnet50 -i sample:cat --html -o flow.png`.
