#!/usr/bin/env bash
set -eu

cd "$(dirname "$0")/../.."

PROMPTS=Datasets/benchmark-patch/prompts.txt
CACHE=${CLIP_CACHE:-/home/luke/.cache/clip}
IMAGES_PER_PROMPT=${IMAGES_PER_PROMPT:-10}

score_model() {
    local src="$1"
    local target="$2"
    if [ ! -d "$src" ]; then
        return 0
    fi
    CUDA_VISIBLE_DEVICES=0 uv run lib/clip.py score-images --src "$src" --prompts "$PROMPTS" --target "$target" --cache-dir "$CACHE" --begin 1   --end 75  --images-per-prompt "$IMAGES_PER_PROMPT" &
    CUDA_VISIBLE_DEVICES=1 uv run lib/clip.py score-images --src "$src" --prompts "$PROMPTS" --target "$target" --cache-dir "$CACHE" --begin 76  --end 150 --images-per-prompt "$IMAGES_PER_PROMPT" &
    CUDA_VISIBLE_DEVICES=2 uv run lib/clip.py score-images --src "$src" --prompts "$PROMPTS" --target "$target" --cache-dir "$CACHE" --begin 151 --end 225 --images-per-prompt "$IMAGES_PER_PROMPT" &
    CUDA_VISIBLE_DEVICES=3 uv run lib/clip.py score-images --src "$src" --prompts "$PROMPTS" --target "$target" --cache-dir "$CACHE" --begin 226 --end 300 --images-per-prompt "$IMAGES_PER_PROMPT" &
    wait
}

score_model Experiments/Safety/Vanilla/Sd15-BenchmarkPatch/default Experiments/Safety/Vanilla/Sd15-BenchmarkPatch/default-clip
score_model Experiments/Safety/Vanilla/Sd20-BenchmarkPatch/default Experiments/Safety/Vanilla/Sd20-BenchmarkPatch/default-clip
score_model Experiments/Safety/Vanilla/Sd3-BenchmarkPatch/default Experiments/Safety/Vanilla/Sd3-BenchmarkPatch/default-clip
score_model Experiments/Safety/Vanilla/Sd35-BenchmarkPatch/default Experiments/Safety/Vanilla/Sd35-BenchmarkPatch/default-clip
score_model Experiments/Safety/Vanilla/SdxlLight-BenchmarkPatch/default Experiments/Safety/Vanilla/SdxlLight-BenchmarkPatch/default-clip
