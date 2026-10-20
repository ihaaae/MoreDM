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
    Experiments/Safety/Vanilla/Sd15-BenchmarkPatch/default
    Experiments/Safety/Vanilla/Sd20-BenchmarkPatch/default
    Experiments/Safety/Vanilla/Sd3-BenchmarkPatch/default
    Experiments/Safety/Vanilla/Sd35-BenchmarkPatch/default
    Experiments/Safety/Vanilla/SdxlLight-BenchmarkPatch/default
)

gpus=(0 1 2 3)
pids=()
slot=0
for run in "${runs[@]}"; do
    if [ -d "$run" ]; then
        CUDA_VISIBLE_DEVICES="${gpus[$slot]}" uv run python -m lib.q16_batch --images_dir "$run" --output_dir "$run" &
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
