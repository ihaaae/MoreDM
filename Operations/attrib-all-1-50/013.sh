#!/bin/sh
# Safety evaluation for attribution family images (baseline + minority)

base="/home/lxc/MoreDM/Experiments/Attribution"
img_base="$base/Text2Image"
safety_base="$base/Safety"

tmp=$(mktemp -d)

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

        count=0
        total=$(find "$family_dir" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')

        for var_dir in "$family_dir"/*/; do
            test -d "$var_dir" || continue
            name=$(basename "$var_dir")
            count=$((count + 1))
            echo "[$method/$sp_id] ($count/$total) $name"

            uv run metrics/unsafe-diffusion/inference.py \
                --images_dir "$var_dir" \
                --output_dir "$tmp"

            safe=$(safe_num "$tmp/predictions.json")
            unsafe=$(unsafe_num "$tmp/predictions.json")
            printf '%-8s%-8s%-8s\n' "$name" "$safe" "$unsafe" >> "$log"
        done

        echo "  -> $log"
    done
}

eval_method "Baseline"
eval_method "Minority"

rm -rf "$tmp"
echo "Safety evaluation complete."
