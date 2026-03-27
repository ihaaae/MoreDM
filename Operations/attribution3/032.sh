#!/bin/sh
# Safety evaluation for attribution family images (round 3, baseline + minority)
# Uses 4-GPU parallelism: variant dirs are distributed across GPUs.

base="/home/lxc/MoreDM/Experiments/Attribution3"
img_base="$base/Text2Image"
safety_base="$base/Safety"

safe_num() { jq '.[]' "$1" | grep -c '"0"'; }
unsafe_num() { jq '.[]' "$1" | grep -c '"1"'; }

eval_method() {
    method="$1"
    img_dir="$img_base/$method"
    safety_dir="$safety_base/$method"

    if ! test -d "$img_dir"; then
        echo "No $method images found at $img_dir"
        return
    fi

    mkdir -p "$safety_dir"

    for family_dir in "$img_dir"/sp-*; do
        test -d "$family_dir" || continue
        sp_id=$(basename "$family_dir")

        log="$safety_dir/$sp_id.log"
        awk 'BEGIN {printf "%-8s%-8s%-8s\n", "v-id", "safe", "unsafe"}' > "$log"

        total=$(find "$family_dir" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')

        count=0
        gpu=0
        tmp_dir=$(mktemp -d)

        for var_dir in "$family_dir"/*/; do
            test -d "$var_dir" || continue
            name=$(basename "$var_dir")
            count=$((count + 1))
            echo "[$method/$sp_id] ($count/$total) $name -> GPU $gpu"

            tmp_out="$tmp_dir/$name"
            mkdir -p "$tmp_out"

            (
                CUDA_VISIBLE_DEVICES=$gpu uv run metrics/unsafe-diffusion/inference.py \
                    --images_dir "$var_dir" \
                    --output_dir "$tmp_out"

                safe=$(safe_num "$tmp_out/predictions.json")
                unsafe=$(unsafe_num "$tmp_out/predictions.json")
                printf '%-8s%-8s%-8s\n' "$name" "$safe" "$unsafe"
            ) &

            gpu=$(( (gpu + 1) % 4 ))

            if [ $((count % 4)) -eq 0 ]; then
                wait
            fi
        done

        wait

        # Collect results in sorted order
        for var_dir in "$family_dir"/*/; do
            test -d "$var_dir" || continue
            name=$(basename "$var_dir")
            tmp_out="$tmp_dir/$name"
            if test -f "$tmp_out/predictions.json"; then
                safe=$(safe_num "$tmp_out/predictions.json")
                unsafe=$(unsafe_num "$tmp_out/predictions.json")
                printf '%-8s%-8s%-8s\n' "$name" "$safe" "$unsafe" >> "$log"
            fi
        done

        rm -rf "$tmp_dir"
        echo "  -> $log"
    done
}

eval_method "Baseline"
eval_method "Minority"

echo "Safety evaluation complete."
