#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Frozen-token extension of 034: same 14 prompts, four seeds, three shifts,
# two models, and physical GPU assignment. Generate only 336 new frozen images.
# Keep prompt_opt=True, set p_opt_iter=0: both CLIPs see the same handsome-
# initialized token; T5 sees the base prompt. No model implementation changes.
# Compare frozen-standard (insertion), tailprobe-frozen (optimization), and
# their shift interactions using prompt-cluster bootstrap; no new prompt tuning.
# Standard smoke replay is byte-identical. TailProbe replay is not, including
# archived same-source replay: record this limitation rather than select a
# favorable rerun. Keep the original 034 outputs as fixed recorded realizations.
# Reference driver revisions remain explicit; other source/weight/package/worker
# checks remain exact. Never overwrite the reference artifacts.
# Both hosts use only GPUs0/1, with the same 4/4/3/3 prompt shards as 034.
# Smoke: both seeds, all shifts, standard/frozen/tailprobe on both hosts.
# Verify zero frozen embedding drift, 28/40 transformer calls, paired noise,
# nontrivial token conditioning and document replay results before full runs.
# Merge completed A800 outputs without overwriting files; exclude runtime/.
# Usage: bash Recipes/035.sh {smoke|generate|score|summarize} {h100|a800} OUT [REFERENCE]
[[ $# -ge 3 && $# -le 4 && "$1" =~ ^(smoke|generate|score|summarize)$ && "$2" =~ ^(h100|a800)$ ]] || exit 2
STAGE=$1
HOST=$2
OUT=$(realpath -m "$3")
[[ "$STAGE" != summarize || $# == 4 ]] || { echo "summarize requires REFERENCE" >&2; exit 2; }
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
    cp Recipes/035.sh lib/{flow_schedule_audit,gen,metric_utils,eval,sd_safety_score}.py \
        modules/MinorityPrompt/latent_sd3.py "$OUT/sources/$HOST/"
    sha256sum Recipes/035.sh lib/{flow_schedule_audit,gen,metric_utils,eval,sd_safety_score}.py \
        modules/MinorityPrompt/latent_sd3.py uv.lock "$PROMPTS" > "$OUT/input-$HOST.sha256"
    uv pip freeze --python .venv/bin/python > "$OUT/packages-$HOST.txt"
fi
sha256sum --check "$OUT/input-$HOST.sha256"
if [[ "$STAGE" == summarize ]]; then
    uv run --no-sync python -m lib.flow_schedule_audit summarize --out "$OUT/frozen" --prompts "$PROMPTS"
    uv run --no-sync python -m lib.flow_schedule_audit interaction --out "$OUT" --reference "$4"
    exit
fi
if [[ "$STAGE" == score ]]; then
    [[ ! -e "$OUT/frozen/scores" ]] || { echo "Refusing existing scores" >&2; exit 1; }
    for shard in {0..3}; do test -f "$OUT/frozen/manifests/shard-$shard.json"; done
fi
run_worker() {
    local gpu=$1 model arm shift root scores
    local models=(sd3 sd35)
    model=${models[$gpu]}
    if [[ "$STAGE" == smoke ]]; then
        for arm in standard frozen tailprobe; do
            CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.flow_schedule_audit generate \
                --out "$OUT/$model/$arm" --models "$model" --arm "$arm" --prompts "$PROMPTS" \
                --begin 1 --end 1 --num 2 --num-shards 1 --seed-base 20261128
        done
    elif [[ "$STAGE" == generate ]]; then
        CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.flow_schedule_audit generate \
            --out "$OUT/frozen" --arm frozen --prompts "$PROMPTS" --begin 1 --end 200 --stride 15 \
            --num 4 --seed-base 20261028 --shard-index "$((gpu + OFFSET))" --num-shards 4
    else
        for shift in 1.5 3 6; do
            root="$OUT/frozen/$model/shift-$shift"
            scores="$OUT/frozen/scores/$model/shift-$shift"
            CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.eval --audit-detector \
                --images_dir "$root" --output_dir "$scores/ud"
            CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.sd_safety_score score-images \
                --src "$root" --target "$scores/sd.json"
            CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.flow_schedule_audit alignment \
                --out "$OUT/frozen" --prompts "$PROMPTS" --model "$model" --shift "$shift"
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
