#!/usr/bin/env bash
set -eu

cd "$(dirname "$0")/../.."

# TailProbe ideal-prompt selection over the SDXL-Lightning discovery setting
# (200-prompt Lexica, regenerated 10-image runs), reproducing the paper's
# filtered-subset statistics table (Eq. 1 in sec/04_method.tex and
# tab:filtered_subsets in sec/05_experiments.tex).
#
# Requires the regenerated-run detector outputs (predictions.json,
# nudenet_scores.json, sd_safety_scores.json) under both run dirs; the scoring
# recipes under Recipes/19-multimetric-safety-sd-family produce them.
STANDARD=Experiments/Safety/Vanilla/SdxlLight-Lexica/regenerated-default
TAILPROBE=Experiments/Safety/Minority/SdxlLight-Lexica/regenerated-default
TARGET=Experiments/Safety/Comparison/TailProbe-Selection

uv run python -m lib.selection \
    --standard-run "$STANDARD" \
    --tailprobe-run "$TAILPROBE" \
    --target "$TARGET"
