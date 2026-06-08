#!/usr/bin/env bash
set -eu

cd "$(dirname "$0")/.."

# CLIP image-wise comparison: Minority/default vs Vanilla baseline (SD 2.0 Lexica).
uv run lib/clip.py compare-imagewise \
    --baseline Experiments/Safety/Vanilla/Sd20-Lexica/default-clip \
    --minority Experiments/Safety/Minority/Sd20-Lexica/default-clip \
    --target Experiments/Safety/Comparison/CLIP-ImageWise-Minority-vs-Baseline-Sd20-Lexica \
    --clip-tol 0.001
