#!/bin/sh
# Safety evaluation: SdxlLight-unsafe-diffusion-Lexica Vanilla prompts 1-200.

set -eu

CUDA_VISIBLE_DEVICES=0 uv run lib/eval.py \
        --dataset unsafe-diffusion \
        --subset Lexica \
        --strategy Vanilla \
        --begin 1 \
        --end 50 \
        --no-log &

CUDA_VISIBLE_DEVICES=1 uv run lib/eval.py \
        --dataset unsafe-diffusion \
        --subset Lexica \
        --strategy Vanilla \
        --begin 51 \
        --end 100 \
        --no-log &

CUDA_VISIBLE_DEVICES=2 uv run lib/eval.py \
        --dataset unsafe-diffusion \
        --subset Lexica \
        --strategy Vanilla \
        --begin 101 \
        --end 150 \
        --no-log &

CUDA_VISIBLE_DEVICES=3 uv run lib/eval.py \
        --dataset unsafe-diffusion \
        --subset Lexica \
        --strategy Vanilla \
        --begin 151 \
        --end 200 \
        --no-log &

wait

uv run lib/eval.py \
        --dataset unsafe-diffusion \
        --subset Lexica \
        --strategy Vanilla \
        --begin 1 \
        --end 200 \
        --rebuild-log
