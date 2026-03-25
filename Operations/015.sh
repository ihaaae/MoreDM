#!/bin/sh
# Baseline generation: SdxlLight model with Lexica prompts 51-200
# Parallel across 4 GPUs: GPU0=51-88, GPU1=89-125, GPU2=126-163, GPU3=164-200

outdir="/home/lxc/MoreDM/Experiments/Text2Image/Baseline/SdxlLight-Lexica"
prompts="/home/lxc/MoreDM/Datasets/unsafe-diffusion/Lexica.txt"
model="sdxl-light"

mkdir -p "$outdir"

CUDA_VISIBLE_DEVICES=0 uv run bin/gen.py \
    --outdir "$outdir" --model "$model" --prompts "$prompts" \
    --begin 51 --end 88 &

CUDA_VISIBLE_DEVICES=1 uv run bin/gen.py \
    --outdir "$outdir" --model "$model" --prompts "$prompts" \
    --begin 89 --end 125 &

CUDA_VISIBLE_DEVICES=2 uv run bin/gen.py \
    --outdir "$outdir" --model "$model" --prompts "$prompts" \
    --begin 126 --end 163 &

CUDA_VISIBLE_DEVICES=3 uv run bin/gen.py \
    --outdir "$outdir" --model "$model" --prompts "$prompts" \
    --begin 164 --end 200 &

wait
echo "Done: baseline Lexica 51-200"
