#!/usr/bin/env bash
# End-to-end check with the real MASI UNesT weights (133-structure whole-brain segmentation).
#
# Run on a machine with normal internet access (e.g. your Mac), from the repository root:
#
#     bash tools/test_unest.sh            # figure + short inference movie
#     bash tools/test_unest.sh --quick    # figure only, no gradient explanations
#
# What it does
#   1. installs the package with the medical extras into ./.venv (skipped if it exists)
#   2. `neural-flow fetch unest`: bundle sources (GitHub) + weights (NVIDIA, MD5-checked) + MNI152 T1
#   3. renders the whole-brain figure with sliding windows      -> unest_check/unest_whole_brain.png
#   4. renders the same as a squashed 2-D view                   -> unest_check/unest_flat.png
#   5. renders an 8-window inference movie                       -> unest_check/unest_inference.mp4
#   6. writes unest_check/report.txt (versions, timings, stage table)
#
# Needs ~2 GB of downloads the first time and ~8 GB RAM with explanations (use --quick on smaller machines).
set -euo pipefail
QUICK=0
[[ "${1:-}" == "--quick" ]] && QUICK=1
cd "$(dirname "$0")/.."
OUT=unest_check
mkdir -p "$OUT"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -e ".[all]"

DEVICE=cpu
python - <<'PY' && DEVICE=$(cat /tmp/nf_device) || true
import torch
d = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
open("/tmp/nf_device", "w").write(d)
PY
echo "device: $DEVICE"

neural-flow fetch unest | tee "$OUT/fetch.txt"
T1=$(grep mni152 "$OUT/fetch.txt" | tail -1)
[[ -f "$T1" ]] || T1="$HOME/.cache/neural_flow/mni152_t1_1mm.nii.gz"

EXPLAIN=""
[[ $QUICK == 1 ]] && EXPLAIN="--no-explain"

{
  echo "date:    $(date)"
  echo "python:  $(python --version)"
  python -c "import torch, monai, neural_flow; print('torch:  ', torch.__version__); print('monai:  ', monai.__version__); print('neural_flow:', neural_flow.__version__)"
  echo "device:  $DEVICE"
  echo "input:   $T1"
  echo
  s=$(date +%s)
  neural-flow render unest -i "$T1" --sliding-window --style cinematic --device "$DEVICE" $EXPLAIN \
      -o "$OUT/unest_whole_brain.png"
  echo "figure: $(( $(date +%s) - s )) s"
  s=$(date +%s)
  neural-flow render unest -i "$T1" --sliding-window --style cinematic --flat-3d --device "$DEVICE" --no-explain -q \
      -o "$OUT/unest_flat.png"
  echo "flat figure: $(( $(date +%s) - s )) s"
  if [[ $QUICK == 0 ]]; then
    s=$(date +%s)
    neural-flow movie unest --inference "$T1" --max-windows 8 --device "$DEVICE" -o "$OUT/unest_inference.mp4"
    echo "movie: $(( $(date +%s) - s )) s"
  fi
} 2>&1 | tee "$OUT/report.txt"

echo
echo "done: see $OUT/ (unest_whole_brain.png, unest_flat.png, unest_inference.mp4, report.txt)"
