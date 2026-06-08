#!/usr/bin/env bash
set -eu

cd "$(dirname "$0")/.."

# CLIP-vs-classifier relevance: Minority/default vs Vanilla baseline (SD 1.5 Lexica).
# Classifier counts come from the flat predictions.json produced by lib/eval.py
# (Operations/005-score-sd15-sd20-lexica.sh); lib/clip.py aggregates them per prompt.
uv run lib/clip.py relevance \
    --baseline-log Experiments/Safety/Vanilla/Sd15-Lexica/default/predictions.json \
    --minority-log Experiments/Safety/Minority/Sd15-Lexica/default/predictions.json \
    --baseline-clip Experiments/Safety/Vanilla/Sd15-Lexica/default-clip \
    --minority-clip Experiments/Safety/Minority/Sd15-Lexica/default-clip \
    --target Experiments/Safety/Comparison/CLIP-vs-Classifier-Minority-vs-Baseline-Sd15-Lexica \
    --unsafe-tol 1 \
    --clip-tol 0.001
