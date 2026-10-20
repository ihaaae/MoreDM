#!/usr/bin/env bash
set -eu

cd "$(dirname "$0")/../.."

# CLIP prompt-image similarity scoring for the TailProbe SD3/SD3.5 Lexica
# images, used for the alignment columns of the model-family detector panel.
MIN3=Experiments/Safety/Minority/Sd3-Lexica/default
MIN35=Experiments/Safety/Minority/Sd35-Lexica/default
PROMPTS=Datasets/unsafe-diffusion/Lexica.txt
CACHE=/home/luke/.cache/clip

# 3-way shard across GPUs 0/1/3 (GPU2 was suspect hardware in the SD3
# integration session; see HANDOFF.md).
CUDA_VISIBLE_DEVICES=0 uv run lib/clip.py score-images --src "$MIN3"  --prompts "$PROMPTS" --target "${MIN3}-clip"  --cache-dir "$CACHE" --begin 1   --end 67  --images-per-prompt 10 &
CUDA_VISIBLE_DEVICES=1 uv run lib/clip.py score-images --src "$MIN3"  --prompts "$PROMPTS" --target "${MIN3}-clip"  --cache-dir "$CACHE" --begin 68  --end 134 --images-per-prompt 10 &
CUDA_VISIBLE_DEVICES=3 uv run lib/clip.py score-images --src "$MIN3"  --prompts "$PROMPTS" --target "${MIN3}-clip"  --cache-dir "$CACHE" --begin 135 --end 200 --images-per-prompt 10 &
wait

CUDA_VISIBLE_DEVICES=0 uv run lib/clip.py score-images --src "$MIN35" --prompts "$PROMPTS" --target "${MIN35}-clip" --cache-dir "$CACHE" --begin 1   --end 67  --images-per-prompt 10 &
CUDA_VISIBLE_DEVICES=1 uv run lib/clip.py score-images --src "$MIN35" --prompts "$PROMPTS" --target "${MIN35}-clip" --cache-dir "$CACHE" --begin 68  --end 134 --images-per-prompt 10 &
CUDA_VISIBLE_DEVICES=3 uv run lib/clip.py score-images --src "$MIN35" --prompts "$PROMPTS" --target "${MIN35}-clip" --cache-dir "$CACHE" --begin 135 --end 200 --images-per-prompt 10 &
wait
