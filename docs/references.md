# References

## Architectures

* He K, Zhang X, Ren S, Sun J. *Deep residual learning for image recognition.* CVPR 2016.
* Dosovitskiy A, et al. *An image is worth 16×16 words: Transformers for image recognition at scale.* ICLR 2021.
* Liu Z, et al. *Swin Transformer: Hierarchical vision transformer using shifted windows.* ICCV 2021.
* Huang G, Liu Z, van der Maaten L, Weinberger KQ. *Densely connected convolutional networks.* CVPR 2017.
* Ronneberger O, Fischer P, Brox T. *U-Net: Convolutional networks for biomedical image segmentation.* MICCAI 2015.
* Çiçek Ö, et al. *3D U-Net: Learning dense volumetric segmentation from sparse annotation.* MICCAI 2016.
* Yu X, Yang Q, Zhou Y, Cai LY, Gao R, Lee HH, Li T, Bao S, Xu Z, Lasko TA, Abramson RG, Zhang Z, Huo Y,
  Landman BA, Tang Y. *UNesT: Local spatial representation learning with hierarchical transformer for efficient
  medical segmentation.* Medical Image Analysis 90:102939, 2023.
  [doi:10.1016/j.media.2023.102939](https://doi.org/10.1016/j.media.2023.102939) · [arXiv:2209.14378](https://arxiv.org/abs/2209.14378) · [PubMed 37725868](https://pubmed.ncbi.nlm.nih.gov/37725868) ·
  [code](https://github.com/MASILab/UNesT)
* Yu X, Zhou Y, Tang Y, et al. *Characterizing renal structures with 3D block aggregate transformers.*
  [arXiv:2203.02430](https://arxiv.org/abs/2203.02430), 2022 (the reference given by the MONAI UNesT bundle).
* Zhang Z, Zhang H, Zhao L, Chen T, Arik SÖ, Pfister T. *Nested hierarchical transformer: towards accurate,
  data-efficient and interpretable visual understanding.* AAAI 2022 (NesT, UNesT's backbone).
* Hatamizadeh A, Tang Y, Nath V, Yang D, Myronenko A, Landman B, Roth HR, Xu D. *UNETR: Transformers for 3D
  medical image segmentation.* WACV 2022, pp. 1748–1758. [arXiv:2103.10504](https://arxiv.org/abs/2103.10504)
* Hatamizadeh A, Nath V, Tang Y, Yang D, Roth HR, Xu D. *Swin UNETR: Swin transformers for semantic
  segmentation of brain tumors in MRI images.* BrainLes 2021, LNCS 12962, 2022.
  [arXiv:2201.01266](https://arxiv.org/abs/2201.01266)
* Tang Y, Yang D, Li W, Roth HR, Landman B, Xu D, Nath V, Hatamizadeh A. *Self-supervised pre-training of Swin
  transformers for 3D medical image analysis.* CVPR 2022.
* Myronenko A. *3D MRI brain tumor segmentation using autoencoder regularization.* BrainLes 2018, LNCS 11384
  (SegResNet).
* Huo Y, Xu Z, Xiong Y, Aboud K, Parvathaneni P, Bao S, Bermudez C, Resnick SM, Cutting LE, Landman BA.
  *3D whole brain segmentation using spatially localized atlas network tiles.* NeuroImage 194:105–119, 2019.
* Vaswani A, et al. *Attention is all you need.* NeurIPS 2017.

## Explanation and visualization methods

* Selvaraju RR, et al. *Grad-CAM: Visual explanations from deep networks via gradient-based localization.* ICCV 2017.
* Luo W, Li Y, Urtasun R, Zemel R. *Understanding the effective receptive field in deep convolutional neural networks.* NeurIPS 2016.
* Abnar S, Zuidema W. *Quantifying attention flow in transformers.* ACL 2020.
* Zeiler MD, Fergus R. *Visualizing and understanding convolutional networks.* ECCV 2014.
* Olah C, Mordvintsev A, Schubert L. *Feature visualization.* Distill 2017.
* Roeder L. *Netron: visualizer for neural network models.* https://github.com/lutzroeder/netron

## Inference and geometry

* Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. *nnU-Net: a self-configuring method for deep
  learning-based biomedical image segmentation.* Nature Methods 18:203–211, 2021 (Gaussian-weighted
  sliding-window inference, as used by MONAI's `SlidingWindowInferer` and by `neural_flow.volume3d`).
* Brett M, et al. *NiBabel* (voxel spacing is read from the NIfTI header, `pixdim`). https://nipy.org/nibabel

## Models, data and tools used by the examples

* Cohen JP, et al. *TorchXRayVision: A library of chest X-ray datasets and models.* MIDL 2022.
* Wang X, et al. *ChestX-ray8: Hospital-scale chest X-ray database and benchmarks.* CVPR 2017.
* Cardoso MJ, et al. *MONAI: An open-source framework for deep learning in healthcare.* arXiv:2211.02701, 2022.
* Wightman R. *PyTorch Image Models (timm).* https://github.com/huggingface/pytorch-image-models
* Fonov V, Evans AC, Botteron K, Almli CR, McKinstry RC, Collins DL. *Unbiased average age-appropriate atlases
  for pediatric studies.* NeuroImage 54(1):313–327, 2011, and Fonov V, et al. *Unbiased nonlinear average
  age-appropriate brain templates from birth to adulthood.* NeuroImage 47(S1):S102, 2009 (MNI152 2009 template,
  shipped with nilearn).
* Abraham A, et al. *Machine learning for neuroimaging with scikit-learn.* Frontiers in Neuroinformatics 8:14,
  2014 (nilearn).
* van der Walt S, et al. *scikit-image: Image processing in Python.* PeerJ 2014 (sample photographs).
* Paszke A, et al. *PyTorch: An imperative style, high-performance deep learning library.* NeurIPS 2019.

## Demo models by example

| figure / movie | model | weights | cite |
|---|---|---|---|
| `resnet50_*` | ResNet-50 | ImageNet-1k (timm `resnet50.a1_in1k` / torchvision) | He 2016; Wightman (timm) |
| `vit_b_16_*` | ViT-B/16 | ImageNet (JAX release via timm) | Dosovitskiy 2021 |
| `chest_xray_*`, `movie_cxr_occlusion` | DenseNet-121 | TorchXRayVision `densenet121-res224-all` | Huang 2017; Cohen 2022; Wang 2017 (data) |
| `unet2d_*` | 2-D U-Net (this repo) | trained here on synthetic cells | Ronneberger 2015 |
| `medical_3d_*` | 3-D U-Net (this repo) | trained here on synthetic heads | Çiçek 2016 |
| `multihead_*`, `movie_aging` | multi-input multi-head 3-D CNN (this repo) | trained here on synthetic heads | — |
| `transformer3d_unetr*`, `movie_sliding_window_unetr` | MONAI UNETR | trained here on synthetic heads | Hatamizadeh 2022 (UNETR); Cardoso 2022 (MONAI) |
| `transformer3d_swinunetr*` | MONAI Swin UNETR | trained here on synthetic heads | Hatamizadeh 2022 (Swin UNETR); Tang 2022; Liu 2021 |
| `neural-flow render unest …` | UNesT (MONAI bundle `wholeBrainSeg_Large_UNEST_segmentation`) | MONAI model zoo (133 structures) | Yu 2023 (UNesT); Zhang 2022 (NesT); Huo 2019 |
| movies with `--inference` | any 3-D model | — | Isensee 2021 (sliding-window fusion) |

The synthetic data generators (`examples/synthetic.py`) are part of this project and need no citation.
