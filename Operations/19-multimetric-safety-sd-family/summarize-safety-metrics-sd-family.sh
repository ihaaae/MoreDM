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

summarize_run() {
    local strategy="$1"
    local model="$2"
    local dataset="$3"
    local run_dir="Experiments/Safety/$strategy/$model-$dataset/default"
    local clip_dir="$run_dir-clip"
    local target="Experiments/Safety/Comparison/Safety-Metrics-$strategy-$model-$dataset"
    summarize "$strategy/$model-$dataset/default" "$run_dir" "$clip_dir" "$target"
}

summarize_run Vanilla Sd15 Lexica
summarize_run Vanilla Sd20 Lexica
summarize_run Vanilla Sd3 Lexica
summarize_run Vanilla Sd35 Lexica
summarize_run Minority Sd15 Lexica
summarize_run Minority Sd20 Lexica

summarize_run Vanilla Sd15 Template
summarize_run Vanilla Sd20 Template
summarize_run Vanilla SdxlLight Template
summarize_run Vanilla Sd3 Template
summarize_run Vanilla Sd35 Template

uv run python lib/safety_aggregate.py compare-runs \
    --title "Safety Metrics: Lexica Vanilla Model Family" \
    --target Experiments/Safety/Comparison/Safety-Metrics-Lexica-Vanilla \
    --summary Experiments/Safety/Comparison/Safety-Metrics-Vanilla-Sd15-Lexica/summary.json \
    --summary Experiments/Safety/Comparison/Safety-Metrics-Vanilla-Sd20-Lexica/summary.json \
    --summary Experiments/Safety/Comparison/Safety-Metrics-Vanilla-Sd3-Lexica/summary.json \
    --summary Experiments/Safety/Comparison/Safety-Metrics-Vanilla-Sd35-Lexica/summary.json

uv run python lib/safety_aggregate.py compare-runs \
    --title "Safety Metrics: Template Vanilla Model Family" \
    --target Experiments/Safety/Comparison/Safety-Metrics-Template-Vanilla \
    --summary Experiments/Safety/Comparison/Safety-Metrics-Vanilla-Sd15-Template/summary.json \
    --summary Experiments/Safety/Comparison/Safety-Metrics-Vanilla-Sd20-Template/summary.json \
    --summary Experiments/Safety/Comparison/Safety-Metrics-Vanilla-SdxlLight-Template/summary.json \
    --summary Experiments/Safety/Comparison/Safety-Metrics-Vanilla-Sd3-Template/summary.json \
    --summary Experiments/Safety/Comparison/Safety-Metrics-Vanilla-Sd35-Template/summary.json

uv run python lib/safety_aggregate.py compare-runs \
    --title "Safety Metrics: SD1.5/SD2.0 Minority vs Vanilla" \
    --target Experiments/Safety/Comparison/Safety-Metrics-Sd15-Sd20-Minority \
    --summary Experiments/Safety/Comparison/Safety-Metrics-Vanilla-Sd15-Lexica/summary.json \
    --summary Experiments/Safety/Comparison/Safety-Metrics-Minority-Sd15-Lexica/summary.json \
    --summary Experiments/Safety/Comparison/Safety-Metrics-Vanilla-Sd20-Lexica/summary.json \
    --summary Experiments/Safety/Comparison/Safety-Metrics-Minority-Sd20-Lexica/summary.json
