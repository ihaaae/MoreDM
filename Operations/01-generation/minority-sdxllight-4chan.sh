#!/bin/sh
# Minority generation (default config): SdxlLight model with 4Chan prompts (1-50)

uv run bin/gen \
        --dataset unsafe-diffusion \
        --subset 4Chan \
        --strategy Minority
