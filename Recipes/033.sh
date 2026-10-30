#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Approved 2026-10-28. Native SD3/SD3.5 schedule sensitivity, no TailProbe.
# 14 Lexica prompts (1,16,...,196), selected without outcomes; four paired seeds;
# shifts 1.5/3/6; 2 models = 336 images. Existing steps/CFG stay at 28/7 and 40/4.5.
# Four independent GPU workers own 4/4/3/3 prompts and ALL their conditions.
# User restricted both hosts to GPUs0/1; H100 GPU3 is also absent this session.
# H100 owns global shards0/1; A800 owns shards2/3. Leave all other cards alone.
# Recheck availability before execution; these checks do not reserve GPUs.
#
# Preparation after approval: transfer the exact reviewed sources/uv.lock and
# both SD3 model caches to an isolated A800 experiment checkout. Its existing
# checkout has unrelated edits; do not overwrite them. Use the existing locked
# environment and hash-verified local assets, with all HF downloads disabled.
# After both generate stages finish, copy A800 model image trees and manifests
# into the H100 output, preserving shard IDs; never overwrite existing images.
# Score ONLY the merged tree on H100. Review images blind to shift before any
# claim that lower detector scores mean greater safety rather than worse images.
# A positive pilot requires an independent prompt/seed confirmation, then a
# separate fixed-sampler TailProbe interaction test; neither is run by this recipe.
# Usage: bash Recipes/033.sh {generate|score|summarize} {h100|a800} <fresh-output>
[[ $# == 3 && "$1" =~ ^(generate|score|summarize)$ && "$2" =~ ^(h100|a800)$ ]] || {
    echo "Usage: $0 {generate|score|summarize} {h100|a800} <output>" >&2; exit 2; }
STAGE=$1
HOST=$2
OUT=$(realpath -m "$3")
[[ "$(hostname | tr '[:upper:]' '[:lower:]')" == "$HOST" ]] || { echo "Wrong host" >&2; exit 2; }
if [[ "$HOST" == h100 ]]; then
    mountpoint -q /mnt/raid10
    [[ "$OUT" == /mnt/raid10/luke/UnsafeDistribution/TailProbe-Audit/* ]] || exit 2
    GPUS=(0 1)
    OFFSET=0
else
    [[ "$STAGE" == generate ]] || { echo "Score merged results on H100" >&2; exit 2; }
    mountpoint -q /mnt/data
    [[ "$OUT" == /mnt/data/luke/UnsafeDistribution/experiment-output/* ]] || exit 2
    GPUS=(0 1)
    OFFSET=2
fi
export PATH="$HOME/.local/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TMPDIR="$OUT/runtime/$HOST/tmp" UV_CACHE_DIR="$OUT/runtime/$HOST/uv-cache"
export TORCHINDUCTOR_CACHE_DIR="$OUT/runtime/$HOST/torchinductor" TRITON_CACHE_DIR="$OUT/runtime/$HOST/triton"
export CUDA_CACHE_PATH="$OUT/runtime/$HOST/cuda-cache"
PROMPTS=Datasets/unsafe-diffusion/Lexica.txt
if [[ "$STAGE" != summarize ]]; then
    for gpu in "${GPUS[@]}"; do
        used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
        [[ "$used" =~ ^[0-9]+$ ]] && (( used <= 1024 )) || { echo "GPU $gpu occupied" >&2; exit 1; }
    done
fi
if [[ "$STAGE" == generate ]]; then
    [[ ! -e "$OUT" ]] || { echo "Refusing existing output: $OUT" >&2; exit 1; }
    mkdir -p "$OUT/logs" "$OUT/sources/$HOST" "$TMPDIR" "$UV_CACHE_DIR" \
        "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
    cp Recipes/033.sh lib/{flow_schedule_audit,gen,metric_utils,eval,sd_safety_score}.py \
        "$OUT/sources/$HOST/"
    sha256sum Recipes/033.sh lib/{flow_schedule_audit,gen,metric_utils,eval,sd_safety_score}.py \
        uv.lock "$PROMPTS" > "$OUT/input-$HOST.sha256"
    uv pip freeze --python .venv/bin/python > "$OUT/packages-$HOST.txt"
fi
sha256sum --check "$OUT/input-$HOST.sha256"
if [[ "$STAGE" == summarize ]]; then
    uv run --no-sync python -m lib.flow_schedule_audit summarize --out "$OUT" --prompts "$PROMPTS"
    exit
fi
if [[ "$STAGE" == score ]]; then
    [[ ! -e "$OUT/scores" ]] || { echo "Refusing existing scores" >&2; exit 1; }
    for shard in {0..3}; do
        test -f "$OUT/manifests/shard-$shard.json"
    done
fi
run_worker() {
    local gpu=$1 model shift root scores
    if [[ "$STAGE" == generate ]]; then
        CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.flow_schedule_audit generate \
            --out "$OUT" --prompts "$PROMPTS" --begin 1 --end 200 --stride 15 --num 4 \
            --seed-base 20261028 --shard-index "$((gpu + OFFSET))" --num-shards 4
    else
        local models=(sd3 sd35)
        model=${models[$gpu]}
        for shift in 1.5 3 6; do
            root="$OUT/$model/shift-$shift"
            scores="$OUT/scores/$model/shift-$shift"
            CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.eval --audit-detector \
                --images_dir "$root" --output_dir "$scores/ud"
            CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.sd_safety_score score-images \
                --src "$root" --target "$scores/sd.json"
            CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.flow_schedule_audit alignment \
                --out "$OUT" --prompts "$PROMPTS" --model "$model" --shift "$shift"
        done
    fi
}
pids=()
for gpu in "${GPUS[@]}"; do
    run_worker "$gpu" > "$OUT/logs/$STAGE-$HOST-$gpu.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
(( status == 0 )) || exit "$status"
sha256sum --check "$OUT/input-$HOST.sha256"
