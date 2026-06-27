#!/usr/bin/env bash
set -eu

cd "$(dirname "$0")/../.."

PROMPTS=Datasets/benchmark-patch/prompts.txt
NUM=${NUM:-10}
DRY_RUN=${DRY_RUN:-}

run_model() {
    local model="$1"
    local outdir="$2"
    local extra=()
    if [ -n "$DRY_RUN" ]; then
        extra+=(--dry_run)
    fi
    CUDA_VISIBLE_DEVICES=0 uv run lib/gen.py --outdir "$outdir" --model "$model" --prompts "$PROMPTS" --begin 1   --end 75  --num "$NUM" "${extra[@]}" &
    CUDA_VISIBLE_DEVICES=1 uv run lib/gen.py --outdir "$outdir" --model "$model" --prompts "$PROMPTS" --begin 76  --end 150 --num "$NUM" "${extra[@]}" &
    CUDA_VISIBLE_DEVICES=2 uv run lib/gen.py --outdir "$outdir" --model "$model" --prompts "$PROMPTS" --begin 151 --end 225 --num "$NUM" "${extra[@]}" &
    CUDA_VISIBLE_DEVICES=3 uv run lib/gen.py --outdir "$outdir" --model "$model" --prompts "$PROMPTS" --begin 226 --end 300 --num "$NUM" "${extra[@]}" &
    wait
}

run_model sd15 Experiments/Safety/Vanilla/Sd15-BenchmarkPatch/default
run_model sd20 Experiments/Safety/Vanilla/Sd20-BenchmarkPatch/default
run_model sd3 Experiments/Safety/Vanilla/Sd3-BenchmarkPatch/default
run_model sd35 Experiments/Safety/Vanilla/Sd35-BenchmarkPatch/default

# Uncomment if the SDXL-Lightning checkpoint is present and this arm is desired.
# run_model sdxl-light Experiments/Safety/Vanilla/SdxlLight-BenchmarkPatch/default
