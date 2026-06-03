#!/bin/sh
# Prompt-wise safety comparison: Minority/default vs Vanilla on expanded Lexica dataset

uv run lib/compare.py promptwise \
        --dataset unsafe-diffusion \
        --subset Lexica
