#!/bin/sh
# Safety evaluation for template injection experiment.
# Evaluates all generated images (Baseline + Minority) × (key + neutral)
# for each element type.  4-GPU parallel across prompt dirs.
#
# Output per (method, element_type, variant):
#   Experiments/Injection/Safety/<Method>/<etype>/<variant>.log
#   Format: p-id  safe  unsafe

base="/home/lxc/MoreDM/Experiments/Injection"
img_base="$base/Text2Image"
safety_base="$base/Safety"

safe_num()  { jq '.[]' "$1" | grep -c '"0"'; }
unsafe_num(){ jq '.[]' "$1" | grep -c '"1"'; }

eval_variant() {
    method="$1"    # Baseline or Minority
    etype="$2"     # person, artist, mood
    variant="$3"   # key or neutral

    img_dir="$img_base/$method/$etype/$variant"
    if ! test -d "$img_dir"; then
        echo "No images at $img_dir — skipping"
        return
    fi

    safety_dir="$safety_base/$method/$etype"
    mkdir -p "$safety_dir"
    log="$safety_dir/${variant}.log"
    printf '%-8s%-8s%-8s\n' "p-id" "safe" "unsafe" > "$log"

    total=$(find "$img_dir" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
    count=0
    gpu=0
    tmp_dir=$(mktemp -d)

    for p_dir in "$img_dir"/*/; do
        test -d "$p_dir" || continue
        name=$(basename "$p_dir")
        count=$((count + 1))
        echo "[$method/$etype/$variant] ($count/$total) $name -> GPU $gpu"

        tmp_out="$tmp_dir/$name"
        mkdir -p "$tmp_out"

        (
            CUDA_VISIBLE_DEVICES=$gpu uv run metrics/unsafe-diffusion/inference.py \
                --images_dir "$p_dir" \
                --output_dir "$tmp_out"
        ) &

        gpu=$(( (gpu + 1) % 4 ))
        if [ $((count % 4)) -eq 0 ]; then
            wait
        fi
    done

    wait

    # Collect results in sorted order
    for p_dir in "$img_dir"/*/; do
        test -d "$p_dir" || continue
        name=$(basename "$p_dir")
        tmp_out="$tmp_dir/$name"
        if test -f "$tmp_out/predictions.json"; then
            safe=$(safe_num "$tmp_out/predictions.json")
            unsafe=$(unsafe_num "$tmp_out/predictions.json")
            printf '%-8s%-8s%-8s\n' "$name" "$safe" "$unsafe" >> "$log"
        fi
    done

    rm -rf "$tmp_dir"
    echo "  -> $log"
}

for method in Baseline Minority; do
    for etype in person artist mood; do
        for variant in key neutral; do
            eval_variant "$method" "$etype" "$variant"
        done
    done
done

echo "Safety evaluation complete."
