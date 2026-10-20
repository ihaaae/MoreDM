#!/usr/bin/env bash
set -eu

cd "$(dirname "$0")/../.."

# TailProbe (MinorityPrompt) generation for SD3 (medium) on the 200-prompt
# Lexica set, 10 images/prompt. One placeholder token, three update iterations
# per selected step (get_sd3_popt_kwargs in lib/gen.py).
# SD3 medium is gated on HuggingFace: accept the license and export a token
# (e.g. `huggingface-cli login` or HF_TOKEN=...) before running.
OUTDIR=Experiments/Safety/Minority/Sd3-Lexica/default
PROMPTS=Datasets/unsafe-diffusion/Lexica.txt
MODEL=min-sd3

# 3-way shard across GPUs 0/1/3 (GPU2 was suspect hardware in the SD3
# integration session; see HANDOFF.md).
CUDA_VISIBLE_DEVICES=0 uv run lib/gen.py --outdir "$OUTDIR" --model "$MODEL" --prompts "$PROMPTS" --begin 1   --end 67  --num 10 &
CUDA_VISIBLE_DEVICES=1 uv run lib/gen.py --outdir "$OUTDIR" --model "$MODEL" --prompts "$PROMPTS" --begin 68  --end 134 --num 10 &
CUDA_VISIBLE_DEVICES=3 uv run lib/gen.py --outdir "$OUTDIR" --model "$MODEL" --prompts "$PROMPTS" --begin 135 --end 200 --num 10 &
wait
