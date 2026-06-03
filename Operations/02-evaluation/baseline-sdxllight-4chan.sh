#!/bin/sh
# Safety evaluation: SdxlLight-unsafe-diffusion-4Chan Vanilla using unsafe-diffusion classifier

uv run lib/eval.py \
        --dataset unsafe-diffusion \
        --subset 4Chan \
        --strategy Vanilla
