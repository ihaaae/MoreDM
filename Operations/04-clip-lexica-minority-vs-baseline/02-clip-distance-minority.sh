#!/bin/sh
# CLIP distance scoring for minority/default Lexica images.

src="/home/lxc/MoreDM/Experiments/Text2Image/Minority/SdxlLight-Lexica/default"
prompts="/home/lxc/MoreDM/Datasets/unsafe-diffusion/Lexica.txt"
target="/home/lxc/MoreDM/Experiments/Safety/Minority/SdxlLight-Lexica/default-clip"

CUDA_VISIBLE_DEVICES=0 uv run lib/clip.py score-images --src "$src" --prompts "$prompts" --target "$target" --begin 1 --end 13 --images-per-prompt 10 &
CUDA_VISIBLE_DEVICES=1 uv run lib/clip.py score-images --src "$src" --prompts "$prompts" --target "$target" --begin 14 --end 26 --images-per-prompt 10 &
CUDA_VISIBLE_DEVICES=2 uv run lib/clip.py score-images --src "$src" --prompts "$prompts" --target "$target" --begin 27 --end 38 --images-per-prompt 10 &
CUDA_VISIBLE_DEVICES=3 uv run lib/clip.py score-images --src "$src" --prompts "$prompts" --target "$target" --begin 39 --end 50 --images-per-prompt 10 &
wait
