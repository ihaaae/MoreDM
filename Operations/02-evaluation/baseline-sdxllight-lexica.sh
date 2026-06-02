#!/bin/sh
# Safety evaluation: SdxlLight-unsafe-diffusion-Lexica Vanilla using unsafe-diffusion classifier

uv run lib/eval.py \
        --dataset unsafe-diffusion \
        --subset Lexica \
        --strategy Vanilla
