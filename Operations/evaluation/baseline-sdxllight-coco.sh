#!/bin/sh
# Safety evaluation: SdxlLight-unsafe-diffusion-COCO Vanilla using unsafe-diffusion classifier

uv run lib/eval.py \
        --dataset unsafe-diffusion \
        --subset COCO \
        --strategy Vanilla
