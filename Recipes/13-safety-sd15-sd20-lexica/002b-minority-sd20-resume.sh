#!/usr/bin/env bash
set -eu

cd "$(dirname "$0")/../.."

OUTDIR=Experiments/Safety/Minority/Sd20-Lexica/default
PROMPTS=Datasets/unsafe-diffusion/Lexica.txt
MODEL=min-sd20

CUDA_VISIBLE_DEVICES=0 uv run lib/gen.py --outdir "$OUTDIR" --model "$MODEL" --prompts "$PROMPTS" --begin 68  --end 90  --num 10 --default &
CUDA_VISIBLE_DEVICES=1 uv run lib/gen.py --outdir "$OUTDIR" --model "$MODEL" --prompts "$PROMPTS" --begin 91  --end 112 --num 10 --default &
CUDA_VISIBLE_DEVICES=3 uv run lib/gen.py --outdir "$OUTDIR" --model "$MODEL" --prompts "$PROMPTS" --begin 113 --end 134 --num 10 --default &
wait
