#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Base SDXL, NOT Lightning: frozen eight diagnostic prompts, three paired
# seeds each, three arms (72 images). Same first three noise seeds as 025.
# 50 steps / CFG7.5 / 1024px / same fp16-fix VAE; no refiner.
# Standard Diffusers Euler versus upstream DDIM without/with TailProbe.
# Non-Lightning upstream defaults: ten updates at t>900, fixed p_ratio=.75;
# keep one handsome token, lr=.01, lambda=1. No outcome-driven tuning.
# UD v2 and existing FP16 SD checker, separately. No semantic gate.
# This does NOT isolate distillation: steps/CFG/optimization timing also differ.
# GPU3 absent, GPU2 occupied: use only GPUs 0,1 and preserve prior artifacts.
# Usage: bash Recipes/027.sh all <fresh-RAID-output> <local-base-SDXL-snapshot>
if [[ $# -ne 3 || ! "$1" =~ ^(all|generate|score|summarize)$ ]]; then
    echo "Usage: $0 {all|generate|score|summarize} <output> <base-model>" >&2
    exit 2
fi
STAGE=$1
OUT=$(realpath -m "$2")
MODEL=$(realpath -e "$3")
BASE=/mnt/raid10/luke/UnsafeDistribution/TailProbe-Audit
mountpoint -q /mnt/raid10 || { echo "RAID unmounted; refusing fallback" >&2; exit 1; }
[[ "$OUT" == "$BASE/"* && "$MODEL" == "$BASE/"* ]] || exit 2
test -f "$MODEL/unet/diffusion_pytorch_model.safetensors"
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
    cp Recipes/027.sh lib/{gen,eval,sampler_audit,sd_safety_score,metric_utils}.py "$OUT/sources/"
    uv pip freeze --python .venv/bin/python > "$OUT/packages.txt"
    sha256sum Recipes/027.sh lib/{gen,eval,sampler_audit,sd_safety_score,metric_utils}.py \
        modules/MinorityPrompt/latent_sdxl.py uv.lock "$PROMPTS" \
        "$MODEL/unet/diffusion_pytorch_model.safetensors" "$MODEL/unet/config.json" \
        "$MODEL/model_index.json" "$MODEL/scheduler/scheduler_config.json" \
        "$BASE/historical-rescore-001/prompt-exclusions.json" "$BASE/confirmation-001/cohort.json" \
        > "$OUT/input-sha256.txt"
    uv run --no-sync python -m lib.sampler_audit prepare-base --out "$OUT" \
        --history "$BASE/historical-rescore-001" > "$OUT/logs/prepare.log"
    sha256sum "$OUT/plan.json" "$OUT"/prompts-*.txt >> "$OUT/input-sha256.txt"
fi
sha256sum --check "$OUT/input-sha256.txt"

run_shard() {
    local gpu=$1 arm path
    local ids=()
    read -r -a ids < "$OUT/prompts-$gpu.txt"
    if [[ "$STAGE" == all || "$STAGE" == generate ]]; then
        CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.gen \
            --model sdxl --sdxl-base-path "$MODEL" --outdir "$OUT/original" --prompts "$PROMPTS" \
            --begin "${ids[0]}" --end "${ids[-1]}" --prompt-ids "${ids[@]}" \
            --num 3 --seed-base 20261024 --audit-phase diagnostic --audit-arms standard
        CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.gen \
            --model min-sdxl --sdxl-base-path "$MODEL" --outdir "$OUT/matched" --prompts "$PROMPTS" \
            --begin "${ids[0]}" --end "${ids[-1]}" --prompt-ids "${ids[@]}" \
            --num 3 --seed-base 20261024 --audit-phase diagnostic --audit-arms standard tailprobe
    fi
    if [[ "$STAGE" == all || "$STAGE" == score ]]; then
        for arm in pipeline matched tailprobe; do
            case "$arm" in
                pipeline) path=original/standard ;;
                matched) path=matched/standard ;;
                tailprobe) path=matched/tailprobe ;;
            esac
            CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.eval --audit-detector \
                --images_dir "$OUT/$path" --output_dir "$OUT/scores/$arm/$gpu" \
                --prompt-range "${ids[0]}" "${ids[-1]}"
            test ! -e "$OUT/sd-scores/$arm/$gpu.json"
            CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.sd_safety_score score-images \
                --src "$OUT/$path" --target "$OUT/sd-scores/$arm/$gpu.json" \
                --prompt-range "${ids[0]}" "${ids[-1]}"
        done
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
    uv run --no-sync python -m lib.sampler_audit summarize --out "$OUT" \
        --confirmation "$BASE/confirmation-001" | tee "$OUT/logs/results.log"
fi
