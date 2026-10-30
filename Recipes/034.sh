#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# User approved smoke and full execution: SD3/SD3.5 TailProbe x shift interaction.
# Same 14 prompts (1,16,...,196), four seeds, shifts 1.5/3/6, two models.
# Two freshly generated matched custom FlowMatch arms = 672 images:
# standard: prompt_opt=False; tailprobe: current Eq.6 defaults, three updates
# per denoising step, lr .01, fixed re-noise sigma .75, handsome initialization.
# Both CLIP encoders are FP32, transformer/T5 BF16; steps/CFG remain 28/7 and
# 40/4.5. No original-pipeline controls reused: those used BF16 CLIP encoders.
# Treat TailProbe as a whole intervention (token insertion + optimization).
# Do not tune parameters or select prompts based on first-round outcomes.
# Primary contrast: (TailProbe_s - Standard_s) - (TailProbe_3 - Standard_3),
# with prompt-cluster bootstrap. Inspect quality and CLIP as well as detectors.
# This is an exploratory extension on the same cohort, not independent confirmation.
#
# User restricts each host to GPUs0/1. Four shards own 4/4/3/3 prompts, keeping
# all conditions for each prompt on one physical GPU. H100 GPU3 is absent.
# Before the full run, smoke-test both models/arms/shifts on each host in fresh
# directories, checking repeated samples, embedding drift, timing and pairing.
# Synchronize exact reviewed sources including modules/MinorityPrompt/latent_sd3.py
# to an isolated A800 checkout, preserving prior runs and unrelated edits.
# Weights are already cached from 033. Merge completed A800 arm trees/manifests
# without overwriting files, then score and summarize centrally on H100.
# Usage: bash Recipes/034.sh {generate|score|summarize} {h100|a800} <fresh-output>
[[ $# == 3 && "$1" =~ ^(generate|score|summarize)$ && "$2" =~ ^(h100|a800)$ ]] || {
    echo "Usage: $0 {generate|score|summarize} {h100|a800} <output>" >&2; exit 2; }
STAGE=$1
HOST=$2
OUT=$(realpath -m "$3")
[[ "$(hostname | tr '[:upper:]' '[:lower:]')" == "$HOST" ]] || { echo "Wrong host" >&2; exit 2; }
if [[ "$HOST" == h100 ]]; then
    mountpoint -q /mnt/raid10
    [[ "$OUT" == /mnt/raid10/luke/UnsafeDistribution/TailProbe-Audit/* ]] || exit 2
    OFFSET=0
else
    [[ "$STAGE" == generate ]] || { echo "Score merged results on H100" >&2; exit 2; }
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
if [[ "$STAGE" == generate ]]; then
    [[ ! -e "$OUT" ]] || { echo "Refusing existing output: $OUT" >&2; exit 1; }
    mkdir -p "$OUT/logs" "$OUT/sources/$HOST" "$TMPDIR" "$UV_CACHE_DIR" \
        "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
    cp Recipes/034.sh lib/{flow_schedule_audit,gen,metric_utils,eval,sd_safety_score}.py \
        modules/MinorityPrompt/latent_sd3.py "$OUT/sources/$HOST/"
    sha256sum Recipes/034.sh lib/{flow_schedule_audit,gen,metric_utils,eval,sd_safety_score}.py \
        modules/MinorityPrompt/latent_sd3.py uv.lock "$PROMPTS" > "$OUT/input-$HOST.sha256"
    uv pip freeze --python .venv/bin/python > "$OUT/packages-$HOST.txt"
fi
sha256sum --check "$OUT/input-$HOST.sha256"
if [[ "$STAGE" == summarize ]]; then
    for arm in standard tailprobe; do
        uv run --no-sync python -m lib.flow_schedule_audit summarize --out "$OUT/$arm" --prompts "$PROMPTS"
    done
    uv run --no-sync python -m lib.flow_schedule_audit interaction --out "$OUT"
    exit
fi
if [[ "$STAGE" == score ]]; then
    for arm in standard tailprobe; do
        [[ ! -e "$OUT/$arm/scores" ]] || { echo "Refusing existing $arm scores" >&2; exit 1; }
        for shard in {0..3}; do
            test -f "$OUT/$arm/manifests/shard-$shard.json"
        done
    done
fi
run_worker() {
    local gpu=$1 arm model shift root scores
    local models=(sd3 sd35)
    for arm in standard tailprobe; do
        if [[ "$STAGE" == generate ]]; then
            CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.flow_schedule_audit generate \
                --out "$OUT/$arm" --arm "$arm" --prompts "$PROMPTS" --begin 1 --end 200 --stride 15 \
                --num 4 --seed-base 20261028 --shard-index "$((gpu + OFFSET))" --num-shards 4
        else
            model=${models[$gpu]}
            for shift in 1.5 3 6; do
                root="$OUT/$arm/$model/shift-$shift"
                scores="$OUT/$arm/scores/$model/shift-$shift"
                CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.eval --audit-detector \
                    --images_dir "$root" --output_dir "$scores/ud"
                CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.sd_safety_score score-images \
                    --src "$root" --target "$scores/sd.json"
                CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.flow_schedule_audit alignment \
                    --out "$OUT/$arm" --prompts "$PROMPTS" --model "$model" --shift "$shift"
            done
        fi
    done
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
