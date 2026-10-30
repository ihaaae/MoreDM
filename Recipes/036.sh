#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Lightning mechanism diagnostic: 2 starts x 2 later grids x 2 scaling rules.
# 8 historically selected prompts x 10 paired seeds = 640 images; no TailProbe.
# Start: 751/999; later grid: 501,251,1 / 749,499,249; terminal sigma always 0.
# Scaling rule: sqrt(sigma_start^2+1) / sigma_start, not a fixed amplitude.
# Native leading/trailing anchors are checked in smoke before the full run.
# User requested only GPUs0/1 on each host; all cells for a prompt stay together.
# H100 shards 0/1, A800 shards 2/3. Merge A800 outputs without overwriting files
# and exclude runtime/. Score centrally on H100. Do not change old experiments.
# Usage: bash Recipes/036.sh {smoke|generate|score|summarize} {h100|a800} OUT
[[ $# == 3 && "$1" =~ ^(smoke|generate|score|summarize)$ && "$2" =~ ^(h100|a800)$ ]] || exit 2
STAGE=$1
HOST=$2
OUT=$(realpath -m "$3")
[[ "$(hostname | tr '[:upper:]' '[:lower:]')" == "$HOST" ]] || exit 2
if [[ "$HOST" == h100 ]]; then
    mountpoint -q /mnt/raid10
    [[ "$OUT" == /mnt/raid10/luke/UnsafeDistribution/TailProbe-Audit/* ]] || exit 2
    OFFSET=0
else
    [[ "$STAGE" == smoke || "$STAGE" == generate ]] || exit 2
    mountpoint -q /mnt/data
    [[ "$OUT" == /mnt/data/luke/UnsafeDistribution/experiment-output/* ]] || exit 2
    OFFSET=2
fi
export PATH="$HOME/.local/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TMPDIR="$OUT/runtime/$HOST/tmp" UV_CACHE_DIR="$OUT/runtime/$HOST/uv-cache"
export TORCHINDUCTOR_CACHE_DIR="$OUT/runtime/$HOST/torchinductor" TRITON_CACHE_DIR="$OUT/runtime/$HOST/triton"
export CUDA_CACHE_PATH="$OUT/runtime/$HOST/cuda-cache"
PROMPTS=Datasets/unsafe-diffusion/Lexica.txt
if [[ "$STAGE" != summarize ]]; then
    for gpu in 0 1; do
        used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
        [[ "$used" =~ ^[0-9]+$ ]] && (( used <= 1024 )) || { echo "GPU $gpu occupied" >&2; exit 1; }
    done
fi
if [[ "$STAGE" == smoke || "$STAGE" == generate ]]; then
    [[ ! -e "$OUT" ]] || { echo "Refusing existing output: $OUT" >&2; exit 1; }
    mkdir -p "$OUT/logs" "$OUT/sources/$HOST" "$TMPDIR" "$UV_CACHE_DIR" \
        "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
    cp Recipes/036.sh lib/{lightning_schedule_audit,flow_schedule_audit,gen,metric_utils,eval,sd_safety_score}.py \
        "$OUT/sources/$HOST/"
    sha256sum Recipes/036.sh lib/{lightning_schedule_audit,flow_schedule_audit,gen,metric_utils,eval,sd_safety_score}.py \
        uv.lock "$PROMPTS" > "$OUT/input-$HOST.sha256"
    uv pip freeze --python .venv/bin/python > "$OUT/packages-$HOST.txt"
    export HF_HUB_CACHE="$PWD/modules/MinorityPrompt/models/huggingface"
fi
sha256sum --check "$OUT/input-$HOST.sha256"
if [[ "$STAGE" == summarize ]]; then
    uv run --no-sync python -m lib.lightning_schedule_audit summarize --out "$OUT" --prompts "$PROMPTS"
    exit
fi
if [[ "$STAGE" == score ]]; then
    for shard in {0..3}; do test -f "$OUT/manifests/shard-$shard.json"; done
    for cell in s{0,1}-g{0,1}-a{0,1}; do
        [[ ! -e "$OUT/$cell/scores" ]] || { echo "Refusing existing scores" >&2; exit 1; }
    done
fi
run_worker() {
    local gpu=$1 cell root
    if [[ "$STAGE" == smoke ]]; then
        CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.lightning_schedule_audit generate \
            --out "$OUT" --prompts "$PROMPTS" --prompt-ids 1 163 --num 2 \
            --shard-index "$gpu" --num-shards 2 --native-check
    elif [[ "$STAGE" == generate ]]; then
        CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.lightning_schedule_audit generate \
            --out "$OUT" --prompts "$PROMPTS" --num 10 --shard-index "$((gpu + OFFSET))" --num-shards 4
    else
        for cell in s"$gpu"-g{0,1}-a{0,1}; do
            root="$OUT/$cell"
            CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.eval --audit-detector \
                --images_dir "$root/images" --output_dir "$root/scores/ud"
            CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.sd_safety_score score-images \
                --src "$root/images" --target "$root/scores/sd.json"
            CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.lightning_schedule_audit alignment \
                --out "$root" --prompts "$PROMPTS"
        done
    fi
}
pids=()
for gpu in 0 1; do
    run_worker "$gpu" > "$OUT/logs/$STAGE-$HOST-$gpu.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
(( status == 0 )) || exit "$status"
sha256sum --check "$OUT/input-$HOST.sha256"
