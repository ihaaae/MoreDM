#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Approved two-group comparison: reuse 029 original SDXL images and scores.
# Union of UD's 11 and SD checker's 32 selected prompts: 39 prompts x 10.
# Reuse 3 matching TailProbe images from 027; generate only 387 missing ones.
# Same seeds, base SDXL, 50 steps, CFG7.5; no third arm or semantic gate.
# TailProbe retains upstream DDIM / t>900 / ten updates / p_ratio=.75.
# GPUs0/1 only: GPU2 belongs to another job; GPU3 is absent.
[[ $# == 2 ]] || { echo "Usage: $0 <fresh-RAID-output> <base-model>" >&2; exit 2; }
OUT=$(realpath -m "$1")
MODEL=$(realpath -e "$2")
BASE=/mnt/raid10/luke/UnsafeDistribution/TailProbe-Audit
mountpoint -q /mnt/raid10
[[ "$OUT" == "$BASE/"* && "$MODEL" == "$BASE/"* && ! -e "$OUT" ]] || exit 2
for gpu in 0 1; do
    used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
    [[ "$used" =~ ^[0-9]+$ ]] && (( used <= 1024 )) || exit 1
done
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TMPDIR="$OUT/runtime/tmp" UV_CACHE_DIR="$OUT/runtime/uv-cache"
export TORCHINDUCTOR_CACHE_DIR="$OUT/runtime/torchinductor" TRITON_CACHE_DIR="$OUT/runtime/triton"
export CUDA_CACHE_PATH="$OUT/runtime/cuda-cache"
mkdir -p "$OUT/logs" "$OUT/sources" "$TMPDIR" "$UV_CACHE_DIR" \
    "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
cp Recipes/030.sh lib/{gen,eval,sd_safety_score,metric_utils,prompt_screen}.py "$OUT/sources/"
sha256sum Recipes/030.sh lib/{gen,eval,sd_safety_score,metric_utils,prompt_screen}.py \
    modules/MinorityPrompt/latent_sdxl.py uv.lock Datasets/unsafe-diffusion/Lexica.txt \
    "$MODEL/unet/diffusion_pytorch_model.safetensors" "$BASE/prompt-screen-001/cohorts.json" \
    > "$OUT/input-sha256.txt"
uv pip freeze --python .venv/bin/python > "$OUT/packages.txt"
uv run --no-sync python -m lib.prompt_screen prepare-tailprobe --out "$OUT"
sha256sum "$OUT/plan.json" "$OUT"/prompts-*.txt >> "$OUT/input-sha256.txt"
run_shard() {
    local gpu=$1
    local ids=()
    read -r -a ids < "$OUT/prompts-$gpu.txt"
    CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.prompt_screen generate-tailprobe \
        --out "$OUT" --model "$MODEL" --shard "$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.eval --audit-detector \
        --images_dir "$OUT/tailprobe" --output_dir "$OUT/scores/ud/$gpu" \
        --prompt-range "${ids[0]}" "${ids[-1]}"
    CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.sd_safety_score score-images \
        --src "$OUT/tailprobe" --target "$OUT/scores/sd/$gpu.json" \
        --prompt-range "${ids[0]}" "${ids[-1]}"
}
pids=()
for gpu in 0 1; do
    run_shard "$gpu" > "$OUT/logs/shard-$gpu.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
(( status == 0 )) || exit "$status"
sha256sum --check "$OUT/input-sha256.txt"
uv run --no-sync python -m lib.prompt_screen summarize-tailprobe --out "$OUT" | tee "$OUT/logs/results.log"
