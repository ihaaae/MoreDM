#!/usr/bin/env bash
set -eu

cd "$(dirname "$0")/../.."

OUTDIR=Experiments/Safety/Vanilla/Sd20-Lexica/default
PROMPTS=Datasets/unsafe-diffusion/Lexica.txt
MODEL=sd20

CUDA_VISIBLE_DEVICES=0 uv run lib/gen.py --outdir "$OUTDIR" --model "$MODEL" --prompts "$PROMPTS" --begin 1   --end 67  --num 10 &
CUDA_VISIBLE_DEVICES=1 uv run lib/gen.py --outdir "$OUTDIR" --model "$MODEL" --prompts "$PROMPTS" --begin 68  --end 134 --num 10 &
CUDA_VISIBLE_DEVICES=3 uv run lib/gen.py --outdir "$OUTDIR" --model "$MODEL" --prompts "$PROMPTS" --begin 135 --end 200 --num 10 &
wait
