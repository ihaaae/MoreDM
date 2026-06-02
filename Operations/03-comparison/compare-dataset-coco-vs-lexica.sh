#!/bin/sh
# Safety comparison: COCO sample vs expanded Lexica dataset

uv run lib/compare.py dataset \
        --left COCO \
        --right Lexica
