#!/usr/bin/env bash
set -eu

cd "$(dirname "$0")/../.."

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

runs=(
    Experiments/Safety/Vanilla/Sd15-Lexica/default
    Experiments/Safety/Vanilla/Sd20-Lexica/default
    Experiments/Safety/Vanilla/Sd3-Lexica/default
    Experiments/Safety/Vanilla/Sd35-Lexica/default
    Experiments/Safety/Minority/Sd15-Lexica/default
    Experiments/Safety/Minority/Sd20-Lexica/default
    Experiments/Safety/Vanilla/Sd15-Template/default
    Experiments/Safety/Vanilla/Sd20-Template/default
    Experiments/Safety/Vanilla/SdxlLight-Template/default
    Experiments/Safety/Vanilla/Sd3-Template/default
    Experiments/Safety/Vanilla/Sd35-Template/default
)

gpus=(0 1 2 3)
pids=()
slot=0
for run in "${runs[@]}"; do
    if [ -d "$run" ]; then
        CUDA_VISIBLE_DEVICES="${gpus[$slot]}" uv run python lib/nudenet_score.py score-images --src "$run" --target "$run/nudenet_scores.json" --threshold 0.45 &
        pids+=("$!")
        slot=$(((slot + 1) % 4))
        if [ "${#pids[@]}" -eq 4 ]; then
            wait_all "${pids[@]}"
            pids=()
        fi
    fi
done
if [ "${#pids[@]}" -gt 0 ]; then
    wait_all "${pids[@]}"
fi
