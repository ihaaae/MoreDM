#!/usr/bin/env bash
set -eu

cd "$(dirname "$0")/../.."

# CLIP prompt-image distance scoring for vanilla SD3 Lexica images.
# Answers research goal 2: do SD3 vanilla images have lower CLIP score
# (i.e. better/worse prompt-image alignment) than the other vanilla models.
SRC=Experiments/Safety/Vanilla/Sd3-Lexica/default
TARGET=Experiments/Safety/Vanilla/Sd3-Lexica/default-clip
PROMPTS=Datasets/unsafe-diffusion/Lexica.txt
CACHE=/home/luke/.cache/clip

# 3-way shard across GPUs 0/1/3 (GPU2 is suspect hardware; see HANDOFF.md).
CUDA_VISIBLE_DEVICES=0 uv run lib/clip.py score-images --src "$SRC" --prompts "$PROMPTS" --target "$TARGET" --cache-dir "$CACHE" --begin 1   --end 67  --images-per-prompt 10 &
CUDA_VISIBLE_DEVICES=1 uv run lib/clip.py score-images --src "$SRC" --prompts "$PROMPTS" --target "$TARGET" --cache-dir "$CACHE" --begin 68  --end 134 --images-per-prompt 10 &
CUDA_VISIBLE_DEVICES=3 uv run lib/clip.py score-images --src "$SRC" --prompts "$PROMPTS" --target "$TARGET" --cache-dir "$CACHE" --begin 135 --end 200 --images-per-prompt 10 &
wait
