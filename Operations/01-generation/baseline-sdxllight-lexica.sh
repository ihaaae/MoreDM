#!/bin/sh
# Vanilla generation: SdxlLight model with Lexica prompts (1-50)

uv run bin/gen \
        --dataset unsafe-diffusion \
        --subset Lexica \
        --strategy Vanilla
