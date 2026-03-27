#!/bin/sh
# Generate 20 more MINORITY images (11-30) for attribution2 families.
# Preserves existing 01-10. Uses 4-GPU parallelism.

base="/home/lxc/MoreDM/Experiments/Attribution2"
families_dir="$base/Families"
outbase="$base/Text2Image/Minority"

if ! test -d "$families_dir"; then
    echo "Missing families directory: $families_dir"
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

echo "Generating minority images 11-30 for $count families across 4 GPUs..."

gpu=0
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
        --model min-sdxl-light \
        --default \
        --prompts "$family_file" \
        --begin 1 \
        --end "$num_lines" \
        --num 20 \
        --img-start 11 &

    gpu=$(( (gpu + 1) % 4 ))
done

wait
echo "Minority generation (11-30) complete."
