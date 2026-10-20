#!/usr/bin/env bash
set -eu

cd "$(dirname "$0")/../.."

# TailProbe (MinorityPrompt) generation for SD3.5 (medium) on the 200-prompt
# Lexica set, 10 images/prompt. lib/gen.py reuses already-cached SD3 text
# encoders/tokenizers and fetches only SD3.5-specific components.
OUTDIR=Experiments/Safety/Minority/Sd35-Lexica/default
PROMPTS=Datasets/unsafe-diffusion/Lexica.txt
MODEL=min-sd35

# 3-way shard across GPUs 0/1/3 (GPU2 was suspect hardware in the SD3
# integration session; see HANDOFF.md).
CUDA_VISIBLE_DEVICES=0 uv run lib/gen.py --outdir "$OUTDIR" --model "$MODEL" --prompts "$PROMPTS" --begin 1   --end 67  --num 10 &
CUDA_VISIBLE_DEVICES=1 uv run lib/gen.py --outdir "$OUTDIR" --model "$MODEL" --prompts "$PROMPTS" --begin 68  --end 134 --num 10 &
CUDA_VISIBLE_DEVICES=3 uv run lib/gen.py --outdir "$OUTDIR" --model "$MODEL" --prompts "$PROMPTS" --begin 135 --end 200 --num 10 &
wait
