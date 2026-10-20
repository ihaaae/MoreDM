#!/bin/sh
# CLIP prompt-wise comparison: Minority/default vs Baseline on Lexica dataset

base="/home/lxc/MoreDM/Experiments/Safety"

uv run lib/clip.py compare-promptwise \
    --baseline "$base/Dataset/1-SdxlLight-Lexica-clip" \
    --minority "$base/Minority/SdxlLight-Lexica/default-clip" \
    --target "$base/Comparison/CLIP-PromptWise-Minority-vs-Baseline-Lexica" \
    --clip-tol 0.001
