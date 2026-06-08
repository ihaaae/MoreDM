#!/usr/bin/env bash
set -eu

cd "$(dirname "$0")/.."

# CLIP prompt-wise comparison: Minority/default vs Vanilla baseline (SD 1.5 Lexica).
uv run lib/clip.py compare-promptwise \
    --baseline Experiments/Safety/Vanilla/Sd15-Lexica/default-clip \
    --minority Experiments/Safety/Minority/Sd15-Lexica/default-clip \
    --target Experiments/Safety/Comparison/CLIP-PromptWise-Minority-vs-Baseline-Sd15-Lexica \
    --clip-tol 0.001
