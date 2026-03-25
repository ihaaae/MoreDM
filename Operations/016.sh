#!/bin/sh
# Minority generation (default config): SdxlLight model with Lexica prompts 101-200
# Prompts 1-100 already generated. Parallel across 4 GPUs.

outdir="/home/lxc/MoreDM/Experiments/Text2Image/Minority/SdxlLight-Lexica/default"
prompts="/home/lxc/MoreDM/Datasets/unsafe-diffusion/Lexica.txt"
model="min-sdxl-light"

mkdir -p "$outdir"

CUDA_VISIBLE_DEVICES=0 uv run bin/gen.py \
    --outdir "$outdir" --model "$model" --prompts "$prompts" \
    --default --begin 101 --end 125 &

CUDA_VISIBLE_DEVICES=1 uv run bin/gen.py \
    --outdir "$outdir" --model "$model" --prompts "$prompts" \
    --default --begin 126 --end 150 &

CUDA_VISIBLE_DEVICES=2 uv run bin/gen.py \
    --outdir "$outdir" --model "$model" --prompts "$prompts" \
    --default --begin 151 --end 175 &

CUDA_VISIBLE_DEVICES=3 uv run bin/gen.py \
    --outdir "$outdir" --model "$model" --prompts "$prompts" \
    --default --begin 176 --end 200 &

wait
echo "Done: minority Lexica 101-200"
