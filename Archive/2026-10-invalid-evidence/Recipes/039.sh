#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# SD3/SD3.5 JEV attribute initialization; reuse historical A and D.
# User authorized smoke and full runs. Only GPUs0/1 on either host.
# H100 owns shard0 of each model, A800 shard1. Never modify its main checkout.
# Copy the frozen plan, legacy solver and exact sources to an isolated A800 root.
# Usage: bash Recipes/039.sh {smoke|full} {h100|a800} <output> <plan> <legacy-solver>
[[ $# == 5 && "$1" =~ ^(smoke|full)$ && "$2" =~ ^(h100|a800)$ ]] || exit 2
STAGE=$1 HOST=$2 OUT=$(realpath -m "$3") PLAN=$(realpath -e "$4") SOLVER=$(realpath -e "$5")
[[ "$(hostname | tr '[:upper:]' '[:lower:]')" == "$HOST" ]] || exit 2
[[ ! -e "$OUT" ]] || { echo "Refusing existing output" >&2; exit 2; }
for gpu in 0 1; do
    used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
    [[ "$used" =~ ^[0-9]+$ ]] && (( used <= 1024 )) || exit 1
done
export PATH="$HOME/.local/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TMPDIR="$OUT/runtime/tmp" UV_CACHE_DIR="$OUT/runtime/uv-cache"
export TORCHINDUCTOR_CACHE_DIR="$OUT/runtime/torchinductor" TRITON_CACHE_DIR="$OUT/runtime/triton"
export CUDA_CACHE_PATH="$OUT/runtime/cuda"
mkdir -p "$OUT/logs" "$OUT/sources" "$TMPDIR" "$UV_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
cp lib/{flow_initializer_audit,gen}.py Recipes/039.sh "$SOLVER" "$PLAN" "$OUT/sources/"
sha256sum lib/{flow_initializer_audit,gen}.py Recipes/039.sh "$SOLVER" "$PLAN" uv.lock > "$OUT/input.sha256"
uv pip freeze --python .venv/bin/python > "$OUT/packages.txt"
SHARD=0
[[ "$HOST" == h100 ]] || SHARD=1
extra=()
[[ "$STAGE" != smoke ]] || extra=(--smoke)
models=(sd3 sd35)
pids=()
for gpu in 0 1; do
    CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.flow_initializer_audit generate \
        --out "$OUT" --plan "$PLAN" --solver "$SOLVER" --model "${models[$gpu]}" \
        --shard "$SHARD" --shards 2 "${extra[@]}" > "$OUT/logs/${models[$gpu]}.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
sha256sum --check "$OUT/input.sha256"
exit "$status"
