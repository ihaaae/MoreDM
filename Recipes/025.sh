#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Small generation-path diagnostic, not a new efficacy confirmation.
# Eight fixed prompts: high historical gain 108,131,163,173; zero-gain controls
# 001,002,014,027 (first two IDs in each of two predeclared history groups).
# Ten NEW paired seeds per prompt; original Diffusers standard, custom standard,
# and custom TailProbe. Same raw Gaussian, checkpoint, CFG1, four steps, PNG
# encoding and UD v2. Preserve each path's native schedule/scaling/precision.
# No semantic gate and no claim of isolating timestep spacing alone.
# No regeneration/overwrites or changes to prior experiments.
# GPU3 is absent; GPU2 is occupied by another task. Use only idle GPUs 0,1.
# Usage: bash Recipes/025.sh all <fresh-output-on-RAID>

if [[ $# -ne 2 || ! "$1" =~ ^(all|generate|score|summarize)$ ]]; then
    echo "Usage: $0 {all|generate|score|summarize} <output-on-RAID>" >&2
    exit 2
fi
STAGE=$1
OUT=$(realpath -m "$2")
BASE=/mnt/raid10/luke/UnsafeDistribution/TailProbe-Audit
mountpoint -q /mnt/raid10 || { echo "RAID unmounted; refusing fallback" >&2; exit 1; }
[[ "$OUT" == "$BASE/"* ]] || exit 2
PROMPTS=Datasets/unsafe-diffusion/Lexica.txt
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TMPDIR="$OUT/runtime/tmp" UV_CACHE_DIR="$OUT/runtime/uv-cache"
export TORCHINDUCTOR_CACHE_DIR="$OUT/runtime/torchinductor" TRITON_CACHE_DIR="$OUT/runtime/triton"
export CUDA_CACHE_PATH="$OUT/runtime/cuda-cache"
if [[ "$STAGE" != summarize ]]; then
    for gpu in 0 1; do
        used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
        [[ "$used" =~ ^[0-9]+$ ]] && (( used <= 1024 )) || { echo "GPU $gpu occupied" >&2; exit 1; }
    done
fi
if [[ "$STAGE" == all || "$STAGE" == generate ]]; then
    [[ ! -e "$OUT" ]] || { echo "Refusing existing output $OUT" >&2; exit 1; }
    free=$(df --output=avail -B1 /mnt/raid10 | tail -1)
    (( free >= 8 * 1024 * 1024 * 1024 )) || exit 1
    mkdir -p "$OUT/logs" "$OUT/sources" "$TMPDIR" "$UV_CACHE_DIR" \
        "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
    cp Recipes/025.sh lib/{gen,eval,sampler_audit}.py "$OUT/sources/"
    sha256sum Recipes/025.sh lib/{gen,eval,sampler_audit}.py uv.lock "$PROMPTS" \
        modules/MinorityPrompt/latent_sdxl.py \
        modules/MinorityPrompt/models/checkpoints/sdxl-lightning/sdxl_lightning_4step_unet.safetensors \
        modules/unsafe-diffusion/checkpoints/multi-headed/*_head.pt \
        "$BASE/confirmation-001/cohort.json" "$BASE/historical-rescore-001/prompt-exclusions.json" \
        > "$OUT/input-sha256.txt"
    uv pip freeze --python .venv/bin/python > "$OUT/packages.txt"
    uv run --no-sync python -m lib.sampler_audit prepare --out "$OUT" \
        --history "$BASE/historical-rescore-001" > "$OUT/logs/prepare.log"
    sha256sum "$OUT/plan.json" "$OUT"/prompts-*.txt >> "$OUT/input-sha256.txt"
fi
sha256sum --check "$OUT/input-sha256.txt"

generate_shard() {
    local gpu=$1
    local ids=()
    read -r -a ids < "$OUT/prompts-$gpu.txt"
    CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.gen \
        --model sdxl-light --outdir "$OUT/original" --prompts "$PROMPTS" \
        --begin "${ids[0]}" --end "${ids[-1]}" --prompt-ids "${ids[@]}" \
        --num 10 --seed-base 20261024 --audit-phase diagnostic --audit-arms standard
    CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.gen \
        --model min-sdxl-light --outdir "$OUT/matched" --prompts "$PROMPTS" \
        --begin "${ids[0]}" --end "${ids[-1]}" --prompt-ids "${ids[@]}" \
        --num 10 --seed-base 20261024 --audit-phase diagnostic --audit-arms standard tailprobe
}

score_shard() {
    local gpu=$1 arm path
    local ids=()
    read -r -a ids < "$OUT/prompts-$gpu.txt"
    for arm in pipeline matched tailprobe; do
        case "$arm" in
            pipeline) path=original/standard ;;
            matched) path=matched/standard ;;
            tailprobe) path=matched/tailprobe ;;
        esac
        CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.eval --audit-detector \
            --images_dir "$OUT/$path" --output_dir "$OUT/scores/$arm/$gpu" \
            --prompt-range "${ids[0]}" "${ids[-1]}"
    done
}

run_parallel() {
    local action=$1 gpu pid status=0
    local pids=()
    for gpu in 0 1; do
        "${action}_shard" "$gpu" > "$OUT/logs/$action-$gpu.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid" || status=1; done
    return "$status"
}

if [[ "$STAGE" == all || "$STAGE" == generate ]]; then
    run_parallel generate
fi
if [[ "$STAGE" == all || "$STAGE" == score ]]; then
    sha256sum --check "$OUT/input-sha256.txt"
    run_parallel score
fi
if [[ "$STAGE" == all || "$STAGE" == summarize ]]; then
    sha256sum --check "$OUT/input-sha256.txt"
    uv run --no-sync python -m lib.sampler_audit summarize --out "$OUT" \
        --confirmation "$BASE/confirmation-001" | tee "$OUT/logs/results.log"
fi
