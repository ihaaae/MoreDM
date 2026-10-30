#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# SD3 Medium only. Approved by user 2026-10-25.
# Question: is the historical SD3 TailProbe noise-field degradation caused by the
# custom sampler, the all-step update window, or the velocity-deviation objective?
# Eight prompts chosen without SD3 outcomes (Lexica IDs 1,26,...,176, stride 25),
# five paired seeds each, four arms (160 images), one shared custom FlowMatch
# sampler, 28 steps, CFG 7.0, 1024px, clamp PNG encoding (no min/max stretch):
#   standard        no optimization
#   velocity-all    historical objective, t_lo=0 (reproduction)
#   velocity-early  historical objective, t_lo=0.9 (window contrast)
#   eq6-all         Eq. 6 port via x0 = x_t - sigma*v, t_lo=0 (objective contrast)
# Other settings frozen from lib/gen.py: handsome init, lr .01, 3 iters/step,
# p_ratio .75 (also the Eq. 6 re-noise sigma), lambda 1, CLIP-L only.
# Endpoints: unvalidated noise proxy, UD v2, FP16 SD checker; per-step embedding
# norm/drift and ||v(c_v)-v(c)||. No semantic gate; not an efficacy estimate.
# Predeclared reading: standard noise ~0 rules out the sampler; velocity-early
# or eq6-all with far fewer noise images than velocity-all implicates that factor.
# GPU3 absent, GPU2 occupied: use GPUs 0,1 only; coordinate with the main thread first.
# Results: TailProbe-Audit/sd3-ablation-001. Historical reproduction only:
# lib/sd3_ablation.py requires MinorityPrompt at 0a48703 (pre-Eq. 6 solver).
# Usage: bash Recipes/028.sh {all|generate|score|summarize} <fresh-RAID-output>
SOLVER_REV=0a48703a6024dd3fa9da44b7e2726c577e076993
[[ "$(git -C modules/MinorityPrompt rev-parse HEAD)" == "$SOLVER_REV" ]] \
    && git -C modules/MinorityPrompt diff --quiet HEAD -- latent_sd3.py || {
    echo "Needs unmodified modules/MinorityPrompt/latent_sd3.py at $SOLVER_REV" >&2; exit 1; }
if [[ $# -ne 2 || ! "$1" =~ ^(all|generate|score|summarize)$ ]]; then
    echo "Usage: $0 {all|generate|score|summarize} <output>" >&2
    exit 2
fi
STAGE=$1
OUT=$(realpath -m "$2")
BASE=/mnt/raid10/luke/UnsafeDistribution/TailProbe-Audit
mountpoint -q /mnt/raid10 || { echo "RAID unmounted; refusing fallback" >&2; exit 1; }
[[ "$OUT" == "$BASE/"* ]] || exit 2
PROMPTS=Datasets/unsafe-diffusion/Lexica.txt
SHARDS=("1 26 51 76" "101 126 151 176")
ARMS="standard velocity-all velocity-early eq6-all"
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
    cp Recipes/028.sh lib/{sd3_ablation,eval,sd_safety_score,metric_utils}.py \
        modules/MinorityPrompt/latent_sd3.py "$OUT/sources/"
    uv pip freeze --python .venv/bin/python > "$OUT/packages.txt"
    sha256sum Recipes/028.sh lib/{sd3_ablation,eval,sd_safety_score,metric_utils}.py \
        modules/MinorityPrompt/latent_sd3.py uv.lock "$PROMPTS" \
        modules/unsafe-diffusion/checkpoints/multi-headed/*_head.pt > "$OUT/input-sha256.txt"
fi
sha256sum --check "$OUT/input-sha256.txt"

run_shard() {
    local gpu=$1 arm
    local ids=()
    read -r -a ids <<< "${SHARDS[$gpu]}"
    if [[ "$STAGE" == all || "$STAGE" == generate ]]; then
        CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.sd3_ablation generate \
            --out "$OUT" --prompts "$PROMPTS" --prompt-ids "${ids[@]}" \
            --arms $ARMS --num 5 --seed-base 20261025
    fi
    if [[ "$STAGE" == all || "$STAGE" == score ]]; then
        for arm in $ARMS; do
            CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.eval --audit-detector \
                --images_dir "$OUT/$arm" --output_dir "$OUT/scores/ud/$arm/$gpu" \
                --prompt-range "${ids[0]}" "${ids[-1]}"
            mkdir -p "$OUT/scores/sd-checker/$arm"
            CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python lib/sd_safety_score.py score-images \
                --src "$OUT/$arm" --target "$OUT/scores/sd-checker/$arm/$gpu.json" \
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
    (( status == 0 )) || exit 1
fi
if [[ "$STAGE" == all || "$STAGE" == summarize ]]; then
    sha256sum --check "$OUT/input-sha256.txt"
    uv run --no-sync python -m lib.sd3_ablation summarize --out "$OUT" | tee "$OUT/logs/results.log"
fi
