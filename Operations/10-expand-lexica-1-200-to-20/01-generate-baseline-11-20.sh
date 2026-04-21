#!/bin/sh
# Generate 10 more BASELINE images (11-20) for Lexica prompts 1-200.
# Preserves existing 01-10. 4-GPU parallelism: 50 prompts each.

outdir="/home/lxc/MoreDM/Experiments/Text2Image/Baseline/SdxlLight-Lexica"
prompts="/home/lxc/MoreDM/Datasets/unsafe-diffusion/Lexica.txt"
model="sdxl-light"

echo "Generating baseline images 11-20 for Lexica 1-200 across 4 GPUs..."

CUDA_VISIBLE_DEVICES=0 uv run bin/gen.py \
    --outdir "$outdir" --model "$model" --prompts "$prompts" \
    --begin 1 --end 50 --num 10 --img-start 11 &

CUDA_VISIBLE_DEVICES=1 uv run bin/gen.py \
    --outdir "$outdir" --model "$model" --prompts "$prompts" \
    --begin 51 --end 100 --num 10 --img-start 11 &

CUDA_VISIBLE_DEVICES=2 uv run bin/gen.py \
    --outdir "$outdir" --model "$model" --prompts "$prompts" \
    --begin 101 --end 150 --num 10 --img-start 11 &

CUDA_VISIBLE_DEVICES=3 uv run bin/gen.py \
    --outdir "$outdir" --model "$model" --prompts "$prompts" \
    --begin 151 --end 200 --num 10 --img-start 11 &

wait
echo "Done: baseline Lexica 1-200 images 11-20"
