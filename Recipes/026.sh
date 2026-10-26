#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Add ONLY the trailing-spaced Diffusers arm to the fixed 025 diagnostic.
# Same eight prompts, 10 images each, exact same Gaussian seeds, CFG1/four
# steps/checkpoint/precision/PNG encoding/UD v2. No prompt reselection.
# Do not regenerate or alter previous arms. This is not population confirmation.
# GPU3 is absent and GPU2 occupied; use idle GPUs 0,1 with the original shards.
# Usage: bash Recipes/026.sh all <fresh-output-on-RAID> <completed-025-output>
if [[ $# -ne 3 || ! "$1" =~ ^(all|generate|score|summarize)$ ]]; then
    echo "Usage: $0 {all|generate|score|summarize} <output> <baseline>" >&2
    exit 2
fi
STAGE=$1
OUT=$(realpath -m "$2")
BASELINE=$(realpath -e "$3")
mountpoint -q /mnt/raid10 || { echo "RAID unmounted; refusing fallback" >&2; exit 1; }
[[ "$OUT" == /mnt/raid10/luke/UnsafeDistribution/TailProbe-Audit/* ]] || exit 2
test -f "$BASELINE/results.json"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TMPDIR="$OUT/runtime/tmp" UV_CACHE_DIR="$OUT/runtime/uv-cache"
export TORCHINDUCTOR_CACHE_DIR="$OUT/runtime/torchinductor" TRITON_CACHE_DIR="$OUT/runtime/triton"
export CUDA_CACHE_PATH="$OUT/runtime/cuda-cache"
PROMPTS=Datasets/unsafe-diffusion/Lexica.txt
if [[ "$STAGE" != summarize ]]; then
    for gpu in 0 1; do
        used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
        [[ "$used" =~ ^[0-9]+$ ]] && (( used <= 1024 )) || { echo "GPU $gpu occupied" >&2; exit 1; }
    done
fi
if [[ "$STAGE" == all || "$STAGE" == generate ]]; then
    [[ ! -e "$OUT" ]] || { echo "Refusing existing output $OUT" >&2; exit 1; }
    free=$(df --output=avail -B1 /mnt/raid10 | tail -1)
    (( free >= 4 * 1024 * 1024 * 1024 )) || exit 1
    mkdir -p "$OUT/logs" "$OUT/sources" "$TMPDIR" "$UV_CACHE_DIR" \
        "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
    cp Recipes/026.sh lib/{gen,eval,sampler_audit}.py "$OUT/sources/"
    uv pip freeze --python .venv/bin/python > "$OUT/packages.txt"
    sha256sum Recipes/026.sh lib/{gen,eval,sampler_audit}.py uv.lock "$PROMPTS" \
        modules/MinorityPrompt/models/checkpoints/sdxl-lightning/sdxl_lightning_4step_unet.safetensors \
        modules/unsafe-diffusion/checkpoints/multi-headed/*_head.pt \
        "$BASELINE/plan.json" "$BASELINE/results.json" "$BASELINE"/prompts-*.txt > "$OUT/input-sha256.txt"
fi
sha256sum --check "$OUT/input-sha256.txt"

run_shard() {
    local gpu=$1
    local ids=()
    read -r -a ids < "$BASELINE/prompts-$gpu.txt"
    if [[ "$STAGE" == all || "$STAGE" == generate ]]; then
        CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.gen \
            --model sdxl-light --outdir "$OUT" --prompts "$PROMPTS" \
            --begin "${ids[0]}" --end "${ids[-1]}" --prompt-ids "${ids[@]}" \
            --num 10 --seed-base 20261024 --audit-phase diagnostic \
            --audit-arms standard --audit-timestep-spacing trailing
    fi
    if [[ "$STAGE" == all || "$STAGE" == score ]]; then
        CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.eval --audit-detector \
            --images_dir "$OUT/standard" --output_dir "$OUT/scores/$gpu" \
            --prompt-range "${ids[0]}" "${ids[-1]}"
    fi
}
if [[ "$STAGE" != summarize ]]; then
    pids=()
    for gpu in 0 1; do
        run_shard "$gpu" > "$OUT/logs/$STAGE-$gpu.log" 2>&1 &
        pids+=("$!")
    done
    status=0
    for pid in "${pids[@]}"; do wait "$pid" || status=1; done
    (( status == 0 )) || exit "$status"
fi
sha256sum --check "$OUT/input-sha256.txt"
if [[ "$STAGE" == all || "$STAGE" == summarize ]]; then
    uv run --no-sync python -m lib.sampler_audit spacing --out "$OUT" \
        --baseline "$BASELINE" | tee "$OUT/logs/results.log"
fi
