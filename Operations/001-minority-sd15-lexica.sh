#!/usr/bin/env bash
set -eu

cd "$(dirname "$0")/.."

OUTDIR=Experiments/Safety/Minority/Sd15-Lexica/default
PROMPTS=Datasets/unsafe-diffusion/Lexica.txt
MODEL=min-sd15

CUDA_VISIBLE_DEVICES=0 uv run lib/gen.py --outdir "$OUTDIR" --model "$MODEL" --prompts "$PROMPTS" --begin 1   --end 50  --num 10 --default &
CUDA_VISIBLE_DEVICES=1 uv run lib/gen.py --outdir "$OUTDIR" --model "$MODEL" --prompts "$PROMPTS" --begin 51  --end 100 --num 10 --default &
CUDA_VISIBLE_DEVICES=2 uv run lib/gen.py --outdir "$OUTDIR" --model "$MODEL" --prompts "$PROMPTS" --begin 101 --end 150 --num 10 --default &
CUDA_VISIBLE_DEVICES=3 uv run lib/gen.py --outdir "$OUTDIR" --model "$MODEL" --prompts "$PROMPTS" --begin 151 --end 200 --num 10 --default &
wait
