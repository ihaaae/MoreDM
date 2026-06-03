#!/bin/sh
# Safety comparison: Minority/default vs Vanilla on 4Chan dataset

uv run lib/compare.py strategy \
        --dataset unsafe-diffusion \
        --subset 4Chan
