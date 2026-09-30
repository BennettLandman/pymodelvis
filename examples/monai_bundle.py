"""Real 3-D medical example — any MONAI bundle (default: MASI's UNesT whole-brain segmentation, 133 structures).

The bundle's own ``configs/inference.json`` builds the network (``network_def``), its
``models/model.pt`` is loaded, and its preprocessing is applied, so the same script
works for other MONAI Model-Zoo bundles.

What the figure shows (UNesT)

* the hierarchical NesT transformer as real stages: patch embedding → three transformer
  levels (24³ → 12³ → 6³ tokens) → bottleneck, and the decoder climbing back to 96³,
  drawn as a U with the transformer levels bridging to the decoder;
* one 96³ window through the head (the stages) and the whole-brain segmentation fused
  from every sliding window (output card; the traced window is outlined);
* true proportions from the NIfTI voxel spacing.

1. fetch the bundle (sources from GitHub, weights from NVIDIA) and the MNI152 template::

       neural-flow fetch unest            # or: python tools/checkout_real_models.py --only unest

2. render::

       python examples/monai_bundle.py                      # cinematic figure (black)
       python examples/monai_bundle.py --movie              # watch sliding-window inference
       python examples/monai_bundle.py --image my_t1.nii.gz # a T1 affinely registered to MNI space
       python examples/monai_bundle.py --flat               # optional squashed 2-D view
       python examples/monai_bundle.py --bundle-dir path/to/other_bundle

The same with the command-line tool::

       neural-flow render unest -i ~/.cache/neural_flow/mni152_t1_1mm.nii.gz --sliding-window --style cinematic
       neural-flow movie unest --inference ~/.cache/neural_flow/mni152_t1_1mm.nii.gz --max-windows 12

Requires ``pip install "monai[nibabel]" nilearn``.  UNesT: Yu et al., Medical Image Analysis 90:102939, 2023.
"""
import argparse
import os
import sys

import torch

from _common import out_path
from neural_flow import visualize_model, zoo
from neural_flow.volume3d import sliding_window_movie


def find_image(explicit=None):
    if explicit:
        return explicit
    for d in zoo.search_dirs():
        for cand in ("user_t1.nii.gz", "user_t1.nii", "mni152_t1_1mm.nii.gz"):
            p = os.path.join(d, cand)
            if os.path.exists(p):
                return p
    sys.exit("no input volume: pass --image T1.nii.gz or run `neural-flow fetch unest` (saves the MNI152 template)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle-dir", default=None, help="MONAI bundle directory (default: UNesT from the cache)")
    ap.add_argument("--image", default=None, help="input NIfTI (default: your user_t1.nii.gz or the MNI152 template)")
    ap.add_argument("--roi", type=int, default=None, help="window size (default: the bundle's roi_size)")
    ap.add_argument("--overlap", type=float, default=0.25, help="sliding-window overlap (the bundle uses 0.7)")
    ap.add_argument("--movie", action="store_true", help="render the sliding-window inference movie")
    ap.add_argument("--max-windows", type=int, default=16, help="movie: windows shown (all are fused)")
    ap.add_argument("--flat", action="store_true", help="draw stages as squashed 2-D projections")
    ap.add_argument("--no-explain", action="store_true", help="skip gradient explanations (faster)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    bdir = args.bundle_dir or zoo.find_bundle(zoo.UNEST_BUNDLE)
    lm = zoo.load_bundle(bdir)
    lm.model.to(args.device)
    image = find_image(args.image)
    vol = zoo.read_volume(image, lm)[None].to(args.device)             # [1, C, X, Y, Z], whole volume
    spacing = zoo.volume_spacing(image)
    roi = (args.roi,) * 3 if args.roi else tuple(lm.roi or (96, 96, 96))
    kw = dict(style="cinematic", output_types={"output": "segmentation"}, class_names=lm.class_names,
              voxel_spacing=spacing, explain=not args.no_explain, flat_3d=args.flat,
              title=lm.name.replace("_", " "))
    if args.movie:
        path = out_path("movie_sliding_window_unest.mp4")
        sliding_window_movie(lm.model, vol, roi, output=path, overlap=args.overlap, max_windows=args.max_windows,
                             fps=3, hold_last=6, **{k: v for k, v in kw.items() if k != "flat_3d"})
    else:
        path = out_path("monai_bundle_cinematic" + ("_flat" if args.flat else "") + ".png")
        fig = visualize_model(lm.model, vol, output=path, sliding_window=True, roi_size=roi, sw_overlap=args.overlap,
                              **kw)
        print(fig.flow.summary_table())
    print("wrote", path)


if __name__ == "__main__":
    main()
