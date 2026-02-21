#!/bin/sh
# Baseline generation: SdxlLight model with 4Chan prompts (1-50)

mkdir -p /home/lxc/MoreDM/Experiments/Text2Image/Baseline/SdxlLight-4Chan/
uv run bin/gen.py --outdir /home/lxc/MoreDM/Experiments/Text2Image/Baseline/SdxlLight-4Chan \
        --model sdxl-light \
        --prompts /home/lxc/MoreDM/Datasets/unsafe-diffusion/4chan.txt \
        --begin 1 \
        --end 50
