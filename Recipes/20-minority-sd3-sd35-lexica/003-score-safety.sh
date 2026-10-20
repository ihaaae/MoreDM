#!/usr/bin/env bash
set -eu

cd "$(dirname "$0")/../.."

# Safety scoring for the TailProbe SD3/SD3.5 Lexica images:
#   - multi-headed unsafe-diffusion classifier (lib/eval.py) -> predictions.json
#   - Q16 P(inappropriate) batch (lib/q16_batch.py)          -> q16_scores.json
# NudeNet and SD-checker scoring for these arms lives in
# Recipes/19-multimetric-safety-sd-family (add the Minority/Sd3-Lexica and
# Minority/Sd35-Lexica runs to its run lists when needed).
MIN3=Experiments/Safety/Minority/Sd3-Lexica/default
MIN35=Experiments/Safety/Minority/Sd35-Lexica/default

# Invoke q16 via -m so the repo root is on sys.path (q16_batch does
# `from lib.q16 import ...`).
CUDA_VISIBLE_DEVICES=0 uv run lib/eval.py --images_dir "$MIN3" --output_dir "$MIN3" &
CUDA_VISIBLE_DEVICES=1 uv run lib/eval.py --images_dir "$MIN35" --output_dir "$MIN35" &
wait

CUDA_VISIBLE_DEVICES=0 uv run python -m lib.q16_batch --images_dir "$MIN3" --output_dir "$MIN3" &
CUDA_VISIBLE_DEVICES=1 uv run python -m lib.q16_batch --images_dir "$MIN35" --output_dir "$MIN35" &
wait
