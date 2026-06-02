#!/bin/sh
# Safety comparison: COCO vs Lexica datasets

uv run lib/compare.py dataset \
        --left COCO \
        --right Lexica
