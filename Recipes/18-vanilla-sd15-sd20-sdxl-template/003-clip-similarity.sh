#!/usr/bin/env bash
set -eu

cd "$(dirname "$0")/../.."

# CLIP prompt-image similarity scoring for vanilla SD 1.5, SD 2.0, and
# SDXL-Light Template images. `lib/clip.py` stores cosine similarity values in
# files named distances.txt for historical compatibility.
PROMPTS=Datasets/unsafe-diffusion/Template.txt
CACHE=/home/luke/.cache/clip

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
    local src=$1
    local target=$2

    CUDA_VISIBLE_DEVICES=0 uv run lib/clip.py score-images --src "$src" --prompts "$PROMPTS" --target "$target" --cache-dir "$CACHE" --begin 1  --end 10 --images-per-prompt 10 & p0=$!
    CUDA_VISIBLE_DEVICES=1 uv run lib/clip.py score-images --src "$src" --prompts "$PROMPTS" --target "$target" --cache-dir "$CACHE" --begin 11 --end 20 --images-per-prompt 10 & p1=$!
    CUDA_VISIBLE_DEVICES=3 uv run lib/clip.py score-images --src "$src" --prompts "$PROMPTS" --target "$target" --cache-dir "$CACHE" --begin 21 --end 30 --images-per-prompt 10 & p3=$!
    wait_all "$p0" "$p1" "$p3"
}

run_model Experiments/Safety/Vanilla/Sd15-Template/default Experiments/Safety/Vanilla/Sd15-Template/default-clip
run_model Experiments/Safety/Vanilla/Sd20-Template/default Experiments/Safety/Vanilla/Sd20-Template/default-clip
run_model Experiments/Safety/Vanilla/SdxlLight-Template/default Experiments/Safety/Vanilla/SdxlLight-Template/default-clip
