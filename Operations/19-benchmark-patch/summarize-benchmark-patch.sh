#!/usr/bin/env bash
set -eu

cd "$(dirname "$0")/../.."

summarize() {
    local run_name="$1"
    local run_dir="$2"
    local clip_dir="$3"
    local target="$4"
    if [ ! -f "$run_dir/predictions.json" ]; then
        return 0
    fi
    uv run python lib/safety_aggregate.py summarize-run \
        --run-name "$run_name" \
        --predictions "$run_dir/predictions.json" \
        --q16 "$run_dir/q16_scores.json" \
        --nudenet "$run_dir/nudenet_scores.json" \
        --sd-safety "$run_dir/sd_safety_scores.json" \
        --clip "$clip_dir" \
        --target "$target" \
        --alignment-threshold percentile:25
}

summarize Vanilla/Sd15-Lexica/default Experiments/Safety/Vanilla/Sd15-Lexica/default Experiments/Safety/Vanilla/Sd15-Lexica/default-clip Experiments/Safety/Comparison/Metric-Patch-Vanilla-Sd15-Lexica
summarize Vanilla/Sd20-Lexica/default Experiments/Safety/Vanilla/Sd20-Lexica/default Experiments/Safety/Vanilla/Sd20-Lexica/default-clip Experiments/Safety/Comparison/Metric-Patch-Vanilla-Sd20-Lexica
summarize Vanilla/Sd3-Lexica/default Experiments/Safety/Vanilla/Sd3-Lexica/default Experiments/Safety/Vanilla/Sd3-Lexica/default-clip Experiments/Safety/Comparison/Metric-Patch-Vanilla-Sd3-Lexica
summarize Vanilla/Sd35-Lexica/default Experiments/Safety/Vanilla/Sd35-Lexica/default Experiments/Safety/Vanilla/Sd35-Lexica/default-clip Experiments/Safety/Comparison/Metric-Patch-Vanilla-Sd35-Lexica
summarize Minority/Sd15-Lexica/default Experiments/Safety/Minority/Sd15-Lexica/default Experiments/Safety/Minority/Sd15-Lexica/default-clip Experiments/Safety/Comparison/Metric-Patch-Minority-Sd15-Lexica
summarize Minority/Sd20-Lexica/default Experiments/Safety/Minority/Sd20-Lexica/default Experiments/Safety/Minority/Sd20-Lexica/default-clip Experiments/Safety/Comparison/Metric-Patch-Minority-Sd20-Lexica

summarize Vanilla/Sd15-Template/default Experiments/Safety/Vanilla/Sd15-Template/default Experiments/Safety/Vanilla/Sd15-Template/default-clip Experiments/Safety/Comparison/Metric-Patch-Vanilla-Sd15-Template
summarize Vanilla/Sd20-Template/default Experiments/Safety/Vanilla/Sd20-Template/default Experiments/Safety/Vanilla/Sd20-Template/default-clip Experiments/Safety/Comparison/Metric-Patch-Vanilla-Sd20-Template
summarize Vanilla/SdxlLight-Template/default Experiments/Safety/Vanilla/SdxlLight-Template/default Experiments/Safety/Vanilla/SdxlLight-Template/default-clip Experiments/Safety/Comparison/Metric-Patch-Vanilla-SdxlLight-Template
summarize Vanilla/Sd3-Template/default Experiments/Safety/Vanilla/Sd3-Template/default Experiments/Safety/Vanilla/Sd3-Template/default-clip Experiments/Safety/Comparison/Metric-Patch-Vanilla-Sd3-Template
summarize Vanilla/Sd35-Template/default Experiments/Safety/Vanilla/Sd35-Template/default Experiments/Safety/Vanilla/Sd35-Template/default-clip Experiments/Safety/Comparison/Metric-Patch-Vanilla-Sd35-Template

for model in Sd15 Sd20 Sd3 Sd35 SdxlLight; do
    summarize "Vanilla/${model}-BenchmarkPatch/default" "Experiments/Safety/Vanilla/${model}-BenchmarkPatch/default" "Experiments/Safety/Vanilla/${model}-BenchmarkPatch/default-clip" "Experiments/Safety/Comparison/Metric-Patch-Vanilla-${model}-BenchmarkPatch"
done

uv run python lib/safety_aggregate.py compare-runs \
    --title "Metric Patch: Lexica Vanilla Model Family" \
    --target Experiments/Safety/Comparison/Metric-Patch-Lexica-Vanilla \
    --summary Experiments/Safety/Comparison/Metric-Patch-Vanilla-Sd15-Lexica/summary.json \
    --summary Experiments/Safety/Comparison/Metric-Patch-Vanilla-Sd20-Lexica/summary.json \
    --summary Experiments/Safety/Comparison/Metric-Patch-Vanilla-Sd3-Lexica/summary.json \
    --summary Experiments/Safety/Comparison/Metric-Patch-Vanilla-Sd35-Lexica/summary.json

uv run python lib/safety_aggregate.py compare-runs \
    --title "Metric Patch: Template Vanilla Model Family" \
    --target Experiments/Safety/Comparison/Metric-Patch-Template-Vanilla \
    --summary Experiments/Safety/Comparison/Metric-Patch-Vanilla-Sd15-Template/summary.json \
    --summary Experiments/Safety/Comparison/Metric-Patch-Vanilla-Sd20-Template/summary.json \
    --summary Experiments/Safety/Comparison/Metric-Patch-Vanilla-SdxlLight-Template/summary.json \
    --summary Experiments/Safety/Comparison/Metric-Patch-Vanilla-Sd3-Template/summary.json \
    --summary Experiments/Safety/Comparison/Metric-Patch-Vanilla-Sd35-Template/summary.json

uv run python lib/safety_aggregate.py compare-runs \
    --title "Metric Patch: SD1.5/SD2.0 Minority vs Vanilla" \
    --target Experiments/Safety/Comparison/Metric-Patch-Sd15-Sd20-Minority \
    --summary Experiments/Safety/Comparison/Metric-Patch-Vanilla-Sd15-Lexica/summary.json \
    --summary Experiments/Safety/Comparison/Metric-Patch-Minority-Sd15-Lexica/summary.json \
    --summary Experiments/Safety/Comparison/Metric-Patch-Vanilla-Sd20-Lexica/summary.json \
    --summary Experiments/Safety/Comparison/Metric-Patch-Minority-Sd20-Lexica/summary.json
