#!/bin/sh
# Safety comparison: Minority/default vs Vanilla on expanded Lexica dataset

uv run lib/compare.py strategy \
        --dataset unsafe-diffusion \
        --subset Lexica
