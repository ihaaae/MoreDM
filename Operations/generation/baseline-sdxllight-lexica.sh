#!/bin/sh
# Baseline generation: SdxlLight model with Lexica prompts (1-50)

mkdir -p /home/lxc/MoreDM/Experiments/Text2Image/Baseline/SdxlLight-Lexica/
uv run bin/gen.py --outdir /home/lxc/MoreDM/Experiments/Text2Image/Baseline/SdxlLight-Lexica \
        --model sdxl-light \
        --prompts /home/lxc/MoreDM/Datasets/unsafe-diffusion/Lexica.txt \
        --begin 1 \
        --end 50
