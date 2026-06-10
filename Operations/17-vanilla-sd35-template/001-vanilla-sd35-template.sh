#!/usr/bin/env bash
set -eu

cd "$(dirname "$0")/../.."

# Vanilla SD3.5 Medium generation on the 30-prompt Unsafe Diffusion Template set,
# 10 images/prompt. SD3.5 Medium is gated on HuggingFace: accept the license and
# export a token (e.g. `huggingface-cli login` or HF_TOKEN=...) before running.
# lib/gen.py reuses already-cached SD3 text encoders/tokenizers and fetches only
# SD3.5-specific components from the SD3.5 repo.
OUTDIR=Experiments/Safety/Vanilla/Sd35-Template/default
PROMPTS=Datasets/unsafe-diffusion/Template.txt
MODEL=sd35

# 3-way shard across GPUs 0/1/3 (GPU2 is suspect hardware; see HANDOFF.md and
# Operations/14-vanilla-sd3-lexica).
CUDA_VISIBLE_DEVICES=0 uv run lib/gen.py --outdir "$OUTDIR" --model "$MODEL" --prompts "$PROMPTS" --begin 1  --end 10 --num 10 &
CUDA_VISIBLE_DEVICES=1 uv run lib/gen.py --outdir "$OUTDIR" --model "$MODEL" --prompts "$PROMPTS" --begin 11 --end 20 --num 10 &
CUDA_VISIBLE_DEVICES=3 uv run lib/gen.py --outdir "$OUTDIR" --model "$MODEL" --prompts "$PROMPTS" --begin 21 --end 30 --num 10 &
wait
