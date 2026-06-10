#!/usr/bin/env bash
set -eu

cd "$(dirname "$0")/../.."

# Safety scoring for vanilla SD 1.5, SD 2.0, and SDXL-Light Template images:
#   - multi-headed unsafe-diffusion classifier (lib/eval.py)  -> predictions.json
#   - Q16 P(inappropriate) batch (lib/q16_batch.py)           -> q16_scores.json
SD15=Experiments/Safety/Vanilla/Sd15-Template/default
SD20=Experiments/Safety/Vanilla/Sd20-Template/default
SDXL=Experiments/Safety/Vanilla/SdxlLight-Template/default

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

CUDA_VISIBLE_DEVICES=0 uv run lib/eval.py --images_dir "$SD15" --output_dir "$SD15" & p0=$!
CUDA_VISIBLE_DEVICES=1 uv run lib/eval.py --images_dir "$SD20" --output_dir "$SD20" & p1=$!
CUDA_VISIBLE_DEVICES=3 uv run lib/eval.py --images_dir "$SDXL" --output_dir "$SDXL" & p3=$!
wait_all "$p0" "$p1" "$p3"

# Invoke via -m so the repo root is on sys.path (q16_batch does `from lib.q16
# import ...`); running the script by path would put lib/ on the path instead.
CUDA_VISIBLE_DEVICES=0 uv run python -m lib.q16_batch --images_dir "$SD15" --output_dir "$SD15" & p0=$!
CUDA_VISIBLE_DEVICES=1 uv run python -m lib.q16_batch --images_dir "$SD20" --output_dir "$SD20" & p1=$!
CUDA_VISIBLE_DEVICES=3 uv run python -m lib.q16_batch --images_dir "$SDXL" --output_dir "$SDXL" & p3=$!
wait_all "$p0" "$p1" "$p3"
