#!/bin/sh
# Safety comparison: Minority/default vs Vanilla on COCO dataset

uv run lib/compare.py strategy \
        --dataset unsafe-diffusion \
        --subset COCO
