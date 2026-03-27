#!/bin/sh
# Safety evaluation for the newly generated minority Lexica 51-100 images.
# Run after 025.sh.

min_src="/home/lxc/MoreDM/Experiments/Text2Image/Minority/SdxlLight-Lexica/default"
min_tgt="/home/lxc/MoreDM/Experiments/Safety/Minority/SdxlLight-Lexica/default"

mkdir -p "$min_tgt"

eval_range() {
    gpu="$1"; lo="$2"; hi="$3"
    tmp=$(mktemp -d)
    for i in $(seq "$lo" "$hi"); do
        name=$(printf "%03d" "$i")
        if test -d "$min_src/$name" && ! test -f "$min_tgt/$name.json"; then
            CUDA_VISIBLE_DEVICES="$gpu" uv run metrics/unsafe-diffusion/inference.py \
                --images_dir "$min_src/$name" --output_dir "$tmp"
            mv "$tmp/predictions.json" "$min_tgt/$name.json"
        fi
    done
    rm -rf "$tmp"
}

echo "Evaluating minority Lexica 51-100 across 4 GPUs..."

eval_range 0 51 63 &
eval_range 1 64 75 &
eval_range 2 76 88 &
eval_range 3 89 100 &

wait
echo "Done: safety eval for minority Lexica 51-100"
