#!/bin/sh
# Minority generation (default config): SdxlLight model with 4Chan prompts (1-50)

mkdir -p /home/lxc/MoreDM/Experiments/Text2Image/Minority/SdxlLight-4Chan/default/
uv run bin/gen.py --outdir /home/lxc/MoreDM/Experiments/Text2Image/Minority/SdxlLight-4Chan/default \
        --model min-sdxl-light \
        --prompts /home/lxc/MoreDM/Datasets/unsafe-diffusion/4chan.txt \
        --default \
        --begin 1 \
        --end 50
