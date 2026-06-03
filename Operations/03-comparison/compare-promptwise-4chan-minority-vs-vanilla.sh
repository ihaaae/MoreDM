#!/bin/sh
# Prompt-wise safety comparison: Minority/default vs Vanilla on 4Chan dataset

uv run lib/compare.py promptwise \
        --dataset unsafe-diffusion \
        --subset 4Chan
