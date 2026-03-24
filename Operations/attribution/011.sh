#!/bin/sh
# Baseline image generation for attribution families (sdxl-light, 10 images per variant)

base="/home/lxc/MoreDM/Experiments/Attribution"
families_dir="$base/Families"
outbase="$base/Text2Image/Baseline"

if ! test -d "$families_dir"; then
    echo "Missing families directory: $families_dir (run 010.sh first)"
    exit 1
fi

for family_dir in "$families_dir"/sp-*; do
    test -d "$family_dir" || continue
    sp_id=$(basename "$family_dir")
    family_file="$family_dir/family.txt"

    if ! test -f "$family_file"; then
        echo "Skipping $sp_id: no family.txt"
        continue
    fi

    num_lines=$(wc -l < "$family_file" | tr -d ' ')
    outdir="$outbase/$sp_id"
    mkdir -p "$outdir"

    echo "[$sp_id] Generating baseline: $num_lines prompt(s) -> $outdir"
    uv run bin/gen.py \
        --outdir "$outdir" \
        --model sdxl-light \
        --prompts "$family_file" \
        --begin 1 \
        --end "$num_lines"
done

echo "Baseline generation complete."
