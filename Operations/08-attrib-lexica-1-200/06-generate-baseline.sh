#!/bin/sh
# Baseline image generation for attribution families (round 3)
# Batches 4 families at a time (one per GPU) to avoid OOM.

base="/home/lxc/MoreDM/Experiments/Attribution3"
families_dir="$base/Families"
outbase="$base/Text2Image/Baseline"

if ! test -d "$families_dir"; then
    echo "Missing families directory: $families_dir (run 05-generate-families.sh first)"
    exit 1
fi

all_families=""
count=0
for family_dir in "$families_dir"/sp-*; do
    test -d "$family_dir" || continue
    all_families="$all_families $family_dir"
    count=$((count + 1))
done

if [ "$count" -eq 0 ]; then
    echo "No families found in $families_dir"
    exit 1
fi

echo "Generating baseline for $count families across 4 GPUs (batches of 4)..."

gpu=0
batch=0
for family_dir in $all_families; do
    sp_id=$(basename "$family_dir")
    family_file="$family_dir/family.txt"

    if ! test -f "$family_file"; then
        echo "Skipping $sp_id: no family.txt"
        continue
    fi

    num_lines=$(wc -l < "$family_file" | tr -d ' ')
    outdir="$outbase/$sp_id"
    mkdir -p "$outdir"

    echo "[$sp_id] GPU $gpu: $num_lines prompt(s) -> $outdir"
    CUDA_VISIBLE_DEVICES=$gpu uv run bin/gen.py \
        --outdir "$outdir" \
        --model sdxl-light \
        --prompts "$family_file" \
        --begin 1 \
        --end "$num_lines" &

    gpu=$(( (gpu + 1) % 4 ))
    batch=$((batch + 1))

    # Wait after every 4 jobs
    if [ $((batch % 4)) -eq 0 ]; then
        wait
        echo "--- batch done ---"
    fi
done

wait
echo "Baseline generation complete."
