#!/usr/bin/env bash
set -eu

cd "$(dirname "$0")/../.."

# CLIP prompt-image similarity scoring for vanilla SD3.5 Medium Lexica images.
SRC=Experiments/Safety/Vanilla/Sd35-Lexica/default
TARGET=Experiments/Safety/Vanilla/Sd35-Lexica/default-clip
PROMPTS=Datasets/unsafe-diffusion/Lexica.txt
CACHE=/home/luke/.cache/clip

# 3-way shard across GPUs 0/1/3 (GPU2 is suspect hardware; see HANDOFF.md and
# Operations/14-vanilla-sd3-lexica).
CUDA_VISIBLE_DEVICES=0 uv run lib/clip.py score-images --src "$SRC" --prompts "$PROMPTS" --target "$TARGET" --cache-dir "$CACHE" --begin 1   --end 67  --images-per-prompt 10 &
CUDA_VISIBLE_DEVICES=1 uv run lib/clip.py score-images --src "$SRC" --prompts "$PROMPTS" --target "$TARGET" --cache-dir "$CACHE" --begin 68  --end 134 --images-per-prompt 10 &
CUDA_VISIBLE_DEVICES=3 uv run lib/clip.py score-images --src "$SRC" --prompts "$PROMPTS" --target "$TARGET" --cache-dir "$CACHE" --begin 135 --end 200 --images-per-prompt 10 &
wait
