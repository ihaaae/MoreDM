#!/bin/sh
# Safety evaluation: SdxlLight-unsafe-diffusion-COCO Minority/default using unsafe-diffusion classifier

uv run lib/eval.py \
        --dataset unsafe-diffusion \
        --subset COCO \
        --strategy Minority
