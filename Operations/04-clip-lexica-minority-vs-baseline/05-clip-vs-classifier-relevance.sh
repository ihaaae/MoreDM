#!/bin/sh
# CLIP-vs-classifier relevance: Minority/default vs Baseline on Lexica dataset

base="/home/lxc/MoreDM/Experiments/Safety"

uv run lib/clip.py relevance \
    --baseline-log "$base/Dataset/1-SdxlLight-Lexica/lexica.log" \
    --minority-log "$base/Minority/SdxlLight-Lexica/default/lexica.log" \
    --baseline-clip "$base/Dataset/1-SdxlLight-Lexica-clip" \
    --minority-clip "$base/Minority/SdxlLight-Lexica/default-clip" \
    --target "$base/Comparison/CLIP-vs-Classifier-Minority-vs-Baseline-Lexica" \
    --unsafe-tol 1 \
    --clip-tol 0.001
