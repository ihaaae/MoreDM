#!/bin/sh
# Erase stale minority images for Lexica prompts 51-100 and regenerate them.
# The old images (Oct 2025) were from an undocumented run. This produces
# fresh images under the current pipeline.
#
# Also removes stale safety JSONs for 51-100 so they are re-evaluated.

img_dir="/home/lxc/MoreDM/Experiments/Text2Image/Minority/SdxlLight-Lexica/default"
safety_dir="/home/lxc/MoreDM/Experiments/Safety/Minority/SdxlLight-Lexica/default"
prompts="/home/lxc/MoreDM/Datasets/unsafe-diffusion/Lexica.txt"
model="min-sdxl-light"

echo "Removing stale minority images for prompts 51-100..."
for i in $(seq 51 100); do
    name=$(printf "%03d" "$i")
    rm -rf "$img_dir/$name"
done

echo "Removing stale safety JSONs for prompts 51-100..."
for i in $(seq 51 100); do
    name=$(printf "%03d" "$i")
    name2=$(printf "%02d" "$i")
    rm -f "$safety_dir/$name.json" "$safety_dir/$name2.json"
done

echo "Regenerating minority images for prompts 51-100 across 4 GPUs..."

CUDA_VISIBLE_DEVICES=0 uv run bin/gen.py \
    --outdir "$img_dir" --model "$model" --prompts "$prompts" \
    --default --begin 51 --end 63 &

CUDA_VISIBLE_DEVICES=1 uv run bin/gen.py \
    --outdir "$img_dir" --model "$model" --prompts "$prompts" \
    --default --begin 64 --end 75 &

CUDA_VISIBLE_DEVICES=2 uv run bin/gen.py \
    --outdir "$img_dir" --model "$model" --prompts "$prompts" \
    --default --begin 76 --end 88 &

CUDA_VISIBLE_DEVICES=3 uv run bin/gen.py \
    --outdir "$img_dir" --model "$model" --prompts "$prompts" \
    --default --begin 89 --end 100 &

wait
echo "Done: minority Lexica 51-100 regenerated"
