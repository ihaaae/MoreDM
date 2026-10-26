#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# User-approved Jev initializer experiment. No baseline regeneration.
# H100: 33 prompts x 10; A800 independently owns IDs188,196,197,199.
# IDs94,186 retain handsome and reuse their 20 previous images.
# Only init_word changes; 50 steps, CFG7.5, paired diagnostic seeds unchanged.
# GPU2 occupied, GPU3 absent: use H100 GPUs0/1 only.
[[ $# == 3 ]] || { echo "Usage: $0 {generate|score} <fresh-RAID-output> <base-model>" >&2; exit 2; }
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
    cp "$BASE/jev-init-selection-001/selections.json" "$OUT/selections.json"
    cp Recipes/031.sh lib/{gen,eval,sd_safety_score,metric_utils}.py "$OUT/sources/"
    sha256sum Recipes/031.sh lib/{gen,eval,sd_safety_score,metric_utils}.py \
        modules/MinorityPrompt/latent_sdxl.py uv.lock Datasets/unsafe-diffusion/Lexica.txt \
        "$OUT/selections.json" "$MODEL/unet/diffusion_pytorch_model.safetensors" > "$OUT/input-sha256.txt"
    uv pip freeze --python .venv/bin/python > "$OUT/packages.txt"
fi
sha256sum --check "$OUT/input-sha256.txt"
run_shard() {
    local gpu=$1
    local ids=()
    if [[ "$gpu" == 0 ]]; then
        ids=(12 13 14 15 17 21 29 31 38 43 53 57 71 76 79 87 88)
    else
        ids=(95 101 102 104 114 118 129 132 139 141 142 146 154 160 164 167)
    fi
    if [[ "$STAGE" == generate ]]; then
        CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.gen \
            --model min-sdxl --sdxl-base-path "$MODEL" --outdir "$OUT/h100" \
            --prompts Datasets/unsafe-diffusion/Lexica.txt --begin "${ids[0]}" --end "${ids[-1]}" \
            --prompt-ids "${ids[@]}" --num 10 --seed-base 20261024 \
            --audit-phase diagnostic --audit-arms tailprobe --audit-init-words "$OUT/selections.json"
    else
        local begins=(1 101) ends=(100 200)
        CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.eval --audit-detector \
            --images_dir "$OUT/tailprobe" --output_dir "$OUT/scores/ud/$gpu" \
            --prompt-range "${begins[$gpu]}" "${ends[$gpu]}"
        CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.sd_safety_score score-images \
            --src "$OUT/tailprobe" --target "$OUT/scores/sd/$gpu.json" \
            --prompt-range "${begins[$gpu]}" "${ends[$gpu]}"
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
