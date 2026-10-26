#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Attribute initializer experiment on delta>=3 SDXL cohorts (40 prompts).
# H100 generates 230 new attribute images; prior images are linked unchanged.
# A800 independently supplies 77 missing handsome images, imported before score.
# GPU2 occupied, GPU3 absent: use H100 GPUs0/1 only.
[[ $# == 3 ]] || { echo "Usage: $0 {generate|score} <RAID-output> <base-model>" >&2; exit 2; }
STAGE=$1
[[ "$STAGE" == generate || "$STAGE" == score ]] || exit 2
OUT=$(realpath -m "$2")
MODEL=$(realpath -e "$3")
BASE=/mnt/raid10/luke/UnsafeDistribution/TailProbe-Audit
mountpoint -q /mnt/raid10
[[ "$OUT" == "$BASE/"* && "$MODEL" == "$BASE/"* ]] || exit 2
for gpu in 0 1; do
    used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
    [[ "$used" =~ ^[0-9]+$ ]] && (( used <= 1024 )) || exit 1
done
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TMPDIR="$OUT/runtime/tmp" UV_CACHE_DIR="$OUT/runtime/uv-cache"
export TORCHINDUCTOR_CACHE_DIR="$OUT/runtime/torchinductor" TRITON_CACHE_DIR="$OUT/runtime/triton"
export CUDA_CACHE_PATH="$OUT/runtime/cuda-cache"
if [[ "$STAGE" == generate ]]; then
    [[ ! -e "$OUT" ]] || exit 2
    mkdir -p "$OUT/logs" "$OUT/sources" "$TMPDIR" "$UV_CACHE_DIR" \
        "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
    cp Recipes/032.sh lib/{gen,eval,sd_safety_score,metric_utils,prompt_screen}.py "$OUT/sources/"
    uv run --no-sync python -m lib.prompt_screen prepare-attributes --out "$OUT"
    sha256sum Recipes/032.sh lib/{gen,eval,sd_safety_score,metric_utils,prompt_screen}.py \
        modules/MinorityPrompt/latent_sdxl.py uv.lock Datasets/unsafe-diffusion/Lexica.txt \
        "$OUT/selections.json" "$OUT/plan.json" "$OUT"/prompts-*.txt \
        "$MODEL/unet/diffusion_pytorch_model.safetensors" > "$OUT/input-sha256.txt"
    uv pip freeze --python .venv/bin/python > "$OUT/packages.txt"
fi
sha256sum --check "$OUT/input-sha256.txt"
run_shard() {
    local gpu=$1 arm
    local ids=()
    read -r -a ids < "$OUT/prompts-$gpu.txt"
    if [[ "$STAGE" == generate ]]; then
        CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.gen \
            --model min-sdxl --sdxl-base-path "$MODEL" --outdir "$OUT/h100" \
            --prompts Datasets/unsafe-diffusion/Lexica.txt --begin "${ids[0]}" --end "${ids[-1]}" \
            --prompt-ids "${ids[@]}" --num 10 --seed-base 20261024 \
            --audit-phase diagnostic --audit-arms tailprobe --audit-init-words "$OUT/selections.json"
    else
        local begins=(1 101) ends=(100 200)
        for arm in handsome attribute; do
            CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.eval --audit-detector \
                --images_dir "$OUT/$arm" --output_dir "$OUT/scores/$arm/ud/$gpu" \
                --prompt-range "${begins[$gpu]}" "${ends[$gpu]}"
            CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.sd_safety_score score-images \
                --src "$OUT/$arm" --target "$OUT/scores/$arm/sd/$gpu.json" \
                --prompt-range "${begins[$gpu]}" "${ends[$gpu]}"
        done
    fi
}
pids=()
for gpu in 0 1; do
    run_shard "$gpu" > "$OUT/logs/$STAGE-$gpu.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
(( status == 0 )) || exit "$status"
sha256sum --check "$OUT/input-sha256.txt"
if [[ "$STAGE" == score ]]; then
    uv run --no-sync python -m lib.prompt_screen summarize-attributes --out "$OUT" | tee "$OUT/logs/results.log"
fi
