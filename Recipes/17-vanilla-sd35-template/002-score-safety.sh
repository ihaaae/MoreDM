#!/usr/bin/env bash
set -eu

cd "$(dirname "$0")/../.."

# Safety scoring for vanilla SD3.5 Medium Template images:
#   - multi-headed unsafe-diffusion classifier (lib/eval.py)  -> predictions.json
#   - Q16 P(inappropriate) batch (lib/q16_batch.py)           -> q16_scores.json
SD35=Experiments/Safety/Vanilla/Sd35-Template/default

CUDA_VISIBLE_DEVICES=0 uv run lib/eval.py --images_dir "$SD35" --output_dir "$SD35"

# Invoke via -m so the repo root is on sys.path (q16_batch does `from lib.q16
# import ...`); running the script by path would put lib/ on the path instead.
CUDA_VISIBLE_DEVICES=0 uv run python -m lib.q16_batch --images_dir "$SD35" --output_dir "$SD35"
