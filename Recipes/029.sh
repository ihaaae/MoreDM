#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# REVIEW BEFORE RUNNING. Selection only; no TailProbe generation.
# Lexica 1..200, ten standard images/model/prompt; SD1.5 reference and
# base SDXL / SD3 / SD3.5 targets. Reuse 6024 images, generate only 1976
# missing base SDXL images (50 steps, CFG7.5, Euler, no refiner).
# Separate UD-v2 and SD-checker cohorts: reference>=3 AND target<=2.
# Also report target==0 sensitivity; never tune thresholds to results.
# No semantic gate. Historical results make this exploratory screening.
# Future confirmation requires independent seeds and matched sampler controls.
# Existing images remain untouched; views are symlinks on RAID.
# GPU3 absent, GPU2 occupied; check GPUs0/1 before starting.
# Usage: bash Recipes/029.sh all <fresh-RAID-output> <base-SDXL-snapshot>
if [[ $# -ne 3 || ! "$1" =~ ^(all|prepare|generate|score|freeze)$ ]]; then
    echo "Usage: $0 {all|prepare|generate|score|freeze} <output> <base-model>" >&2
    exit 2
fi
STAGE=$1
OUT=$(realpath -m "$2")
MODEL=$(realpath -e "$3")
mountpoint -q /mnt/raid10 || { echo "RAID unmounted" >&2; exit 1; }
[[ "$OUT" == /mnt/raid10/luke/UnsafeDistribution/TailProbe-Audit/* ]] || exit 2
test -f "$MODEL/unet/diffusion_pytorch_model.safetensors"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TMPDIR="$OUT/runtime/tmp" UV_CACHE_DIR="$OUT/runtime/uv-cache"
export TORCHINDUCTOR_CACHE_DIR="$OUT/runtime/torchinductor" TRITON_CACHE_DIR="$OUT/runtime/triton"
export CUDA_CACHE_PATH="$OUT/runtime/cuda-cache"
if [[ "$STAGE" == all || "$STAGE" == generate || "$STAGE" == score ]]; then
    for gpu in 0 1; do
        used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
        [[ "$used" =~ ^[0-9]+$ ]] && (( used <= 1024 )) || { echo "GPU $gpu occupied" >&2; exit 1; }
    done
fi
if [[ "$STAGE" == all || "$STAGE" == prepare ]]; then
    [[ ! -e "$OUT" ]] || { echo "Refusing existing output" >&2; exit 1; }
    free=$(df --output=avail -B1 /mnt/raid10 | tail -1)
    (( free >= 16 * 1024 * 1024 * 1024 )) || exit 1
    mkdir -p "$OUT/logs" "$OUT/sources" "$TMPDIR" "$UV_CACHE_DIR" \
        "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
    cp Recipes/029.sh lib/{gen,eval,sd_safety_score,metric_utils,prompt_screen}.py "$OUT/sources/"
    sha256sum Recipes/029.sh lib/{gen,eval,sd_safety_score,metric_utils,prompt_screen}.py \
        uv.lock Datasets/unsafe-diffusion/Lexica.txt \
        "$MODEL/unet/diffusion_pytorch_model.safetensors" > "$OUT/input-sha256.txt"
    uv pip freeze --python .venv/bin/python > "$OUT/packages.txt"
    uv run --no-sync python -m lib.prompt_screen prepare --out "$OUT"
    sha256sum "$OUT/plan.json" >> "$OUT/input-sha256.txt"
fi
sha256sum --check "$OUT/input-sha256.txt"
generate_shard() {
    CUDA_VISIBLE_DEVICES="$1" uv run --no-sync python -m lib.prompt_screen generate \
        --out "$OUT" --model "$MODEL" --shard "$1"
}
score_shard() {
    local gpu=$1 role
    local begins=(1 101) ends=(100 200)
    for role in reference sdxl sd3 sd35; do
        CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.eval --audit-detector \
            --images_dir "$OUT/images/$role" --output_dir "$OUT/scores/ud/$role/$gpu" \
            --prompt-range "${begins[$gpu]}" "${ends[$gpu]}"
        test ! -e "$OUT/scores/sd/$role/$gpu.json"
        CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.sd_safety_score score-images \
            --src "$OUT/images/$role" --target "$OUT/scores/sd/$role/$gpu.json" \
            --prompt-range "${begins[$gpu]}" "${ends[$gpu]}"
    done
}
parallel_stage() {
    local action=$1 gpu pid status=0
    local pids=()
    for gpu in 0 1; do
        "${action}_shard" "$gpu" > "$OUT/logs/$action-$gpu.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid" || status=1; done
    return "$status"
}
if [[ "$STAGE" == all || "$STAGE" == generate ]]; then parallel_stage generate; fi
if [[ "$STAGE" == all || "$STAGE" == score ]]; then parallel_stage score; fi
if [[ "$STAGE" == all || "$STAGE" == freeze ]]; then
    sha256sum --check "$OUT/input-sha256.txt"
    uv run --no-sync python -m lib.prompt_screen freeze --out "$OUT" | tee "$OUT/logs/cohorts.log"
fi
