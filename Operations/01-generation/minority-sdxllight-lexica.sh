#!/bin/sh
# Minority generation: SdxlLight model with Lexica prompts 1-200.
# Default run generates 20 images per prompt. Override NUM_IMAGES or IMG_START
# to extend an existing run, for example NUM_IMAGES=10 IMG_START=11.

set -eu

num_images="${NUM_IMAGES:-20}"
img_start="${IMG_START:-1}"

CUDA_VISIBLE_DEVICES=0 uv run bin/gen \
        --dataset unsafe-diffusion \
        --subset Lexica \
        --strategy Minority \
        --begin 1 \
        --end 50 \
        --num "$num_images" \
        --img-start "$img_start" &

CUDA_VISIBLE_DEVICES=1 uv run bin/gen \
        --dataset unsafe-diffusion \
        --subset Lexica \
        --strategy Minority \
        --begin 51 \
        --end 100 \
        --num "$num_images" \
        --img-start "$img_start" &

CUDA_VISIBLE_DEVICES=2 uv run bin/gen \
        --dataset unsafe-diffusion \
        --subset Lexica \
        --strategy Minority \
        --begin 101 \
        --end 150 \
        --num "$num_images" \
        --img-start "$img_start" &

CUDA_VISIBLE_DEVICES=3 uv run bin/gen \
        --dataset unsafe-diffusion \
        --subset Lexica \
        --strategy Minority \
        --begin 151 \
        --end 200 \
        --num "$num_images" \
        --img-start "$img_start" &

wait
