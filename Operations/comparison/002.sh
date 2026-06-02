#!/bin/sh
# Prompt-wise safety comparison: Minority/default vs Vanilla on COCO dataset

uv run lib/compare.py promptwise \
        --dataset unsafe-diffusion \
        --subset COCO
