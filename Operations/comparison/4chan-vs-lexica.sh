#!/bin/sh
# Safety comparison: 4Chan vs Lexica datasets

uv run lib/compare.py dataset \
        --left 4Chan \
        --right Lexica
