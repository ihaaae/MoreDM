#!/usr/bin/env bash
set -eu

cd "$(dirname "$0")/../.."

MIN15=Experiments/Safety/Minority/Sd15-Lexica/default
MIN20=Experiments/Safety/Minority/Sd20-Lexica/default
VAN15=Experiments/Safety/Vanilla/Sd15-Lexica/default
VAN20=Experiments/Safety/Vanilla/Sd20-Lexica/default

CUDA_VISIBLE_DEVICES=0 uv run lib/eval.py --images_dir "$MIN15" --output_dir "$MIN15" &
CUDA_VISIBLE_DEVICES=1 uv run lib/eval.py --images_dir "$MIN20" --output_dir "$MIN20" &
CUDA_VISIBLE_DEVICES=3 uv run lib/eval.py --images_dir "$VAN15" --output_dir "$VAN15" &
wait

CUDA_VISIBLE_DEVICES=0 uv run lib/eval.py --images_dir "$VAN20" --output_dir "$VAN20"

CUDA_VISIBLE_DEVICES=0 uv run lib/q16_batch.py --images_dir "$MIN15" --output_dir "$MIN15" &
CUDA_VISIBLE_DEVICES=1 uv run lib/q16_batch.py --images_dir "$MIN20" --output_dir "$MIN20" &
CUDA_VISIBLE_DEVICES=3 uv run lib/q16_batch.py --images_dir "$VAN15" --output_dir "$VAN15" &
wait

CUDA_VISIBLE_DEVICES=0 uv run lib/q16_batch.py --images_dir "$VAN20" --output_dir "$VAN20"
