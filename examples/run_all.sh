#!/usr/bin/env bash
# Regenerate every example figure and movie (trained demo weights are cached in examples/outputs/*.pt,
# real weights in ../real_models/).  Movies take a few minutes each on a laptop CPU.
set -e
cd "$(dirname "$0")"
# --- real pretrained models
python resnet.py                                   # ResNet-50, cinematic, black  (the cat)
python resnet.py --style technical
python resnet.py --style story
python vit.py                                      # ViT-B/16, cinematic
python vit.py --style technical
python chest_xray.py                               # TorchXRayVision DenseNet-121 on an NIH chest X-ray
# --- trained-on-synthetic demo models
python unet.py && python unet.py --style cinematic && python unet.py --style story
python medical_3d.py && python medical_3d.py --style cinematic && python medical_3d.py --mode projection
python multihead.py && python multihead.py --style cinematic
python medical_3d.py --style cinematic --thick
python transformer_3d.py && python transformer_3d.py --flat && python transformer_3d.py --model swinunetr
# --- extras & movies
python extras.py
python movies.py pan
python movies.py aging
python chest_xray.py --movie
python movies.py vit
python transformer_3d.py --movie
# --- MASI UNesT (after: neural-flow fetch unest)
if python -c "from neural_flow import zoo; zoo.find_bundle(zoo.UNEST_BUNDLE)" 2>/dev/null; then
  python monai_bundle.py && python monai_bundle.py --movie
fi
