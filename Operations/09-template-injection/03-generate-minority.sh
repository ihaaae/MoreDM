#!/bin/sh
# Minority (low-density) image generation for template injection experiment.
# Generates images for both key and neutral prompt variants.
# 4-GPU parallel.

base="/home/lxc/MoreDM/Experiments/Injection"
outbase="$base/Text2Image/Minority"

if ! test -d "$base"; then
    echo "Missing $base (run 01-build-templates.sh first)"
    exit 1
fi

gpu=0
batch=0

for etype_dir in "$base"/person "$base"/artist "$base"/mood; do
    test -d "$etype_dir" || continue
    etype=$(basename "$etype_dir")

    for variant in key neutral; do
        prompts="$etype_dir/prompts_${variant}.txt"
        if ! test -f "$prompts"; then
            echo "Skipping $etype/$variant: no prompt file"
            continue
        fi

        num_lines=$(wc -l < "$prompts" | tr -d ' ')
        outdir="$outbase/$etype/$variant"
        mkdir -p "$outdir"

        echo "[$etype/$variant] GPU $gpu: $num_lines prompts -> $outdir"
        CUDA_VISIBLE_DEVICES=$gpu uv run bin/gen.py \
            --outdir "$outdir" \
            --model min-sdxl-light \
            --default \
            --prompts "$prompts" \
            --begin 1 \
            --end "$num_lines" &

        gpu=$(( (gpu + 1) % 4 ))
        batch=$((batch + 1))

        if [ $((batch % 4)) -eq 0 ]; then
            wait
            echo "--- batch done ---"
        fi
    done
done

wait
echo "Minority generation complete."
