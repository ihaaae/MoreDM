#!/usr/bin/env bash
set -eu

cd "$(dirname "$0")/../.."

# Vanilla SD 1.5, SD 2.0, and SDXL-Light generation on the 30-prompt Unsafe
# Diffusion Template set, 10 images/prompt. SDXL-Light expects the local
# SDXL-Lightning 4-step UNet checkpoint under modules/MinorityPrompt/models.
# Sharded across GPUs 0/1/3 to match recent SD3-family runs that avoid suspect
# GPU2.
PROMPTS=Datasets/unsafe-diffusion/Template.txt

wait_all() {
    local status=0
    local pid
    for pid in "$@"; do
        if ! wait "$pid"; then
            status=1
        fi
    done
    return "$status"
}

run_model() {
    local outdir=$1
    local model=$2

    CUDA_VISIBLE_DEVICES=0 uv run lib/gen.py --outdir "$outdir" --model "$model" --prompts "$PROMPTS" --begin 1  --end 10 --num 10 & p0=$!
    CUDA_VISIBLE_DEVICES=1 uv run lib/gen.py --outdir "$outdir" --model "$model" --prompts "$PROMPTS" --begin 11 --end 20 --num 10 & p1=$!
    CUDA_VISIBLE_DEVICES=3 uv run lib/gen.py --outdir "$outdir" --model "$model" --prompts "$PROMPTS" --begin 21 --end 30 --num 10 & p3=$!
    wait_all "$p0" "$p1" "$p3"
}

run_model Experiments/Safety/Vanilla/Sd15-Template/default sd15
run_model Experiments/Safety/Vanilla/Sd20-Template/default sd20
run_model Experiments/Safety/Vanilla/SdxlLight-Template/default sdxl-light
