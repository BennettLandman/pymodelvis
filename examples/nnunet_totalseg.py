"""nnU-Net example — TotalSegmentator on a public CT (real weights, no nnU-Net install needed).

TotalSegmentator (Wasserthal et al., Radiology: Artificial Intelligence 2023) is a set of
nnU-Net v2 models; its weights are ordinary nnU-Net results folders, so this script is also a
template for *your* nnU-Net models (``--results path/to/Trainer__Plans__3d_fullres``).

What the figures show

* the plain nnU-Net U-Net level by level: encoder 1 … bottleneck … decoder 1 (each decoder
  level's transposed convolution and skip concatenation folded into its edges), the final
  1×1×1 segmentation layer as the head; deep supervision is switched off as at inference;
* nnU-Net's own preprocessing from ``plans.json`` (RAS reorientation, CT clipping and
  normalisation, resampling to the target spacing) and sliding-window inference with the
  plans' patch size, 50 % overlap and Gaussian weighting;
* stages for one window; the output card is the whole CT fused from every window, drawn to
  scale (the traced window outlined).

Models (downloaded once from the TotalSegmentator GitHub releases, Apache-2.0):

* ``totalseg``        fast model, 3 mm, 117 structures, patch 112×112×128 (one window covers the example CT)
* ``totalseg-organs`` full-resolution organ model, 1.5 mm, 24 structures, patch 128³ (27 windows)

Usage::

    neural-flow fetch totalseg totalseg-organs          # models + the example CT
    python examples/nnunet_totalseg.py                  # both figures
    python examples/nnunet_totalseg.py --model totalseg-organs --movie   # watch sliding-window inference
    python examples/nnunet_totalseg.py --results ~/nnUNet_results/Dataset123_X/nnUNetTrainer__nnUNetPlans__3d_fullres --image case.nii.gz

The same with the command-line tool::

    neural-flow render totalseg -i sample:ct --sliding-window --style cinematic
    neural-flow render nnunet:/path/to/results_folder -i case.nii.gz --sliding-window
    neural-flow movie totalseg-organs --inference sample:ct --max-windows 27

Requires ``pip install dynamic-network-architectures nibabel scipy``.
"""
import argparse

import torch

from _common import out_path
from neural_flow import visualize_model, zoo
from neural_flow.nnunet import ct_sample_path
from neural_flow.volume3d import sliding_window_movie

TITLES = {
    "totalseg": ("TotalSegmentator · nnU-Net on a CT", "fast model, 3 mm"),
    "totalseg-organs": ("TotalSegmentator organs · nnU-Net on a CT", "1.5 mm"),
}


def run(spec, image, args, title=None):
    lm = zoo.load_model(spec if spec in TITLES or spec.startswith("nnunet:") else f"nnunet:{spec}", device=args.device)
    info = {}
    vol = zoo.load_input(image, lm, info=info, whole=True).to(args.device)     # [1, 1, z, y, x], preprocessed
    i = lm.info
    t, what = TITLES.get(spec, (title or f"{lm.name} · nnU-Net", i["configuration"]))
    sub = (f"{what} · {i['architecture']}, fold {i['fold']} · patch {'×'.join(map(str, lm.roi))} (z×y×x) · "
           f"{i['n_classes'] - 1} structures")
    kw = dict(style="cinematic", output_types={"output": "segmentation"}, class_names=lm.class_names,
              voxel_spacing=info["spacing"], volume_axes=lm.volume_axes, explain=not args.no_explain,
              title=t, subtitle=sub, explain_max_mb=args.explain_max_mb)
    name = spec.replace("-", "_") if spec in TITLES else lm.name
    if args.movie:
        path = out_path(f"movie_sliding_window_{name}.mp4")
        kw.pop("subtitle")
        sliding_window_movie(lm.model, vol, lm.roi, output=path, overlap=lm.sw_overlap, max_windows=args.max_windows or None,
                             fps=3, hold_last=6, figsize=(16, 9), dpi=120, progress=True, **kw)
    else:
        path = out_path(f"nnunet_{name}.png")
        fig = visualize_model(lm.model, vol, output=path, sliding_window=True, roi_size=lm.roi,
                              sw_overlap=lm.sw_overlap, **kw)
        print(fig.flow.summary_table())
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(TITLES), default=None, help="one TotalSegmentator model (default: both)")
    ap.add_argument("--results", default=None, help="your own nnU-Net results folder[:fold] instead")
    ap.add_argument("--image", default=None, help="input CT (default: TotalSegmentator's example CT)")
    ap.add_argument("--movie", action="store_true", help="render the sliding-window inference movie")
    ap.add_argument("--max-windows", type=int, default=14,
                    help="movie: windows shown, evenly spaced (all windows are fused; 0 = show all)")
    ap.add_argument("--no-explain", action="store_true", help="skip gradient explanations (faster, less memory)")
    ap.add_argument("--explain-max-mb", type=float, default=None,
                    help="memory allowed for the explanations' gradient pass (default: 80%% of free RAM)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    image = args.image or ct_sample_path()
    if args.results:
        run(args.results, image, args)
        return
    for spec in ([args.model] if args.model else (["totalseg-organs"] if args.movie else list(TITLES))):
        run(spec, image, args)


if __name__ == "__main__":
    main()
