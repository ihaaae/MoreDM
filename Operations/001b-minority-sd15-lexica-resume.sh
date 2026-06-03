#!/usr/bin/env bash
set -eu

cd "$(dirname "$0")/.."

OUTDIR=Experiments/Safety/Minority/Sd15-Lexica/default
PROMPTS=Datasets/unsafe-diffusion/Lexica.txt
MODEL=min-sd15

CUDA_VISIBLE_DEVICES=0 uv run lib/gen.py --outdir "$OUTDIR" --model "$MODEL" --prompts "$PROMPTS" --begin 102 --end 150 --num 10 --default
