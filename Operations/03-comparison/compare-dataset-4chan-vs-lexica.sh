#!/bin/sh
# Safety comparison: 4Chan sample vs expanded Lexica dataset

uv run lib/compare.py dataset \
        --left 4Chan \
        --right Lexica
