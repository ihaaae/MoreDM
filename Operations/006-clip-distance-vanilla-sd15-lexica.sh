#!/usr/bin/env bash
set -eu

cd "$(dirname "$0")/.."

# CLIP distance scoring for Vanilla (baseline) SD 1.5 Lexica images.
SRC=Experiments/Safety/Vanilla/Sd15-Lexica/default
TARGET=Experiments/Safety/Vanilla/Sd15-Lexica/default-clip
PROMPTS=Datasets/unsafe-diffusion/Lexica.txt
CACHE=/home/luke/.cache/clip

CUDA_VISIBLE_DEVICES=0 uv run lib/clip.py score-images --src "$SRC" --prompts "$PROMPTS" --target "$TARGET" --cache-dir "$CACHE" --begin 1   --end 50  --images-per-prompt 10 &
CUDA_VISIBLE_DEVICES=1 uv run lib/clip.py score-images --src "$SRC" --prompts "$PROMPTS" --target "$TARGET" --cache-dir "$CACHE" --begin 51  --end 100 --images-per-prompt 10 &
CUDA_VISIBLE_DEVICES=2 uv run lib/clip.py score-images --src "$SRC" --prompts "$PROMPTS" --target "$TARGET" --cache-dir "$CACHE" --begin 101 --end 150 --images-per-prompt 10 &
CUDA_VISIBLE_DEVICES=3 uv run lib/clip.py score-images --src "$SRC" --prompts "$PROMPTS" --target "$TARGET" --cache-dir "$CACHE" --begin 151 --end 200 --images-per-prompt 10 &
wait
