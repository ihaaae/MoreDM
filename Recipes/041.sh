#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Queue JEV attributes AFTER plain Eq.6 generation AND validated scoring.
# Reuse the frozen attribute selections; only initialization changes from plain.
# Recipes/040.sh is the same generation wrapper: H100 SD3.5, A800 SD3, GPUs0/1.
# Preparation: lib.flow_eq6_jev prepare, then copy its plan to the existing
# isolated A800 flow-eq6-code-001/plan/jev-plan.json. No baseline files change.
B=/mnt/raid10/luke/UnsafeDistribution/TailProbe-Audit
OUT="$B/flow-eq6-jev-comparison-001"
PLAN="$B/flow-eq6-jev-plan-001/plan.json"
SOLVER="$B/flow-eq6-plan-001/eq6_sd3.py"
REMOTE=/mnt/data/luke/UnsafeDistribution/experiment-output
CODE=/mnt/data/luke/flow-eq6-code-001
SSH=(ssh -i /home/luke/.ssh/id_ed25519_a800 -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes)
RSYNC_SSH='ssh -i /home/luke/.ssh/id_ed25519_a800 -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes'
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
test -f "$OUT/inventory.json"
mkdir "$OUT/sources"
cp lib/{flow_eq6_jev,flow_initializer_audit,flow_quality,gen,eval}.py Recipes/{040,041}.sh "$OUT/sources/"
sha256sum lib/{flow_eq6_jev,flow_initializer_audit,flow_quality,gen,eval}.py Recipes/{040,041}.sh "$PLAN" "$SOLVER" > "$OUT/input.sha256"
trap 'rc=$?; if (( rc != 0 )); then echo "failed: inspect queue.log" > "$OUT/status.txt"; fi' EXIT
echo 'waiting for validated plain Eq6 results' > "$OUT/status.txt"
ready=0
for attempt in {1..480}; do
    if [[ -s "$B/flow-eq6-comparison-001/results.json" ]]; then ready=1; break; fi
    sleep 30
done
[[ "$ready" == 1 ]] || { echo 'Plain Eq6 did not finish within four hours' >&2; exit 1; }
sleep 5
sha256sum --check "$OUT/input.sha256"

run_generation() {
    local stage=$1 local_out remote_out
    if [[ "$stage" == smoke ]]; then
        local_out="$B/flow-eq6-jev-smoke-h100-001"
        remote_out="$REMOTE/flow-eq6-jev-smoke-a800-001"
    else
        local_out="$B/flow-eq6-jev-h100-001"
        remote_out="$REMOTE/flow-eq6-jev-a800-001"
    fi
    echo "$stage generation" > "$OUT/status.txt"
    bash Recipes/040.sh "$stage" h100 "$local_out" "$PLAN" "$SOLVER" > "$OUT/$stage-h100.log" 2>&1 &
    local p0=$!
    "${SSH[@]}" luke@192.168.100.105 "bash $CODE/Recipes/040.sh $stage a800 $remote_out $CODE/plan/jev-plan.json $CODE/plan/eq6_sd3.py" > "$OUT/$stage-a800.log" 2>&1 &
    local p1=$! status=0
    wait "$p0" || status=1
    wait "$p1" || status=1
    [[ "$status" == 0 ]] || return 1
    rsync -a --exclude=runtime -e "$RSYNC_SSH" "luke@192.168.100.105:$remote_out/" "$B/$(basename "$remote_out")/"
}
run_generation smoke
uv run --no-sync python -m lib.flow_eq6_jev check-smoke
run_generation full
sha256sum --check "$OUT/input.sha256"
echo 'scoring and summarizing' > "$OUT/status.txt"
score() {
    local gpu=$1 model=$2
    CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.eval --audit-detector \
        --images_dir "$OUT/images/$model/J" --output_dir "$OUT/scores/$model"
    CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync python -m lib.flow_quality \
        --root "$OUT" --model "$model" --arms J
}
score 0 sd3 > "$OUT/score-sd3.log" 2>&1 & p0=$!
score 1 sd35 > "$OUT/score-sd35.log" 2>&1 & p1=$!
status=0
wait "$p0" || status=1
wait "$p1" || status=1
[[ "$status" == 0 ]] || exit 1
uv run --no-sync python -m lib.flow_eq6_jev summarize > "$OUT/summary.log"
cp "$OUT/results.json" .amp/in/artifacts/flow-eq6-jev-results.json
echo 'complete' > "$OUT/status.txt"
