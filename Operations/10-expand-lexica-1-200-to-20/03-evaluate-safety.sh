#!/bin/sh
# Re-evaluate safety for ALL 20 images per prompt (Lexica 1-200).
# Overwrites existing JSONs since image count changed from 10 to 20.
# 4-GPU parallelism: each GPU handles 50 baseline + 50 minority prompts.

base_src="/home/lxc/MoreDM/Experiments/Text2Image/Baseline/SdxlLight-Lexica"
base_tgt="/home/lxc/MoreDM/Experiments/Safety/Dataset/1-SdxlLight-Lexica"

min_src="/home/lxc/MoreDM/Experiments/Text2Image/Minority/SdxlLight-Lexica/default"
min_tgt="/home/lxc/MoreDM/Experiments/Safety/Minority/SdxlLight-Lexica/default"

mkdir -p "$base_tgt" "$min_tgt"

eval_range() {
    gpu="$1"; src="$2"; tgt="$3"; lo="$4"; hi="$5"
    tmp=$(mktemp -d)
    for i in $(seq "$lo" "$hi"); do
        name=$(printf "%03d" "$i")
        if test -d "$src/$name"; then
            CUDA_VISIBLE_DEVICES="$gpu" uv run metrics/unsafe-diffusion/inference.py \
                --images_dir "$src/$name" --output_dir "$tmp"
            mv "$tmp/predictions.json" "$tgt/$name.json"
        fi
    done
    rm -rf "$tmp"
}

echo "Re-evaluating safety for Lexica 1-200 (20 images each) across 4 GPUs..."

# GPU 0: baseline 1-50 then minority 1-50
(eval_range 0 "$base_src" "$base_tgt" 1 50 && \
 eval_range 0 "$min_src" "$min_tgt" 1 50) &

# GPU 1: baseline 51-100 then minority 51-100
(eval_range 1 "$base_src" "$base_tgt" 51 100 && \
 eval_range 1 "$min_src" "$min_tgt" 51 100) &

# GPU 2: baseline 101-150 then minority 101-150
(eval_range 2 "$base_src" "$base_tgt" 101 150 && \
 eval_range 2 "$min_src" "$min_tgt" 101 150) &

# GPU 3: baseline 151-200 then minority 151-200
(eval_range 3 "$base_src" "$base_tgt" 151 200 && \
 eval_range 3 "$min_src" "$min_tgt" 151 200) &

wait
echo "Done: safety re-evaluation for Lexica 1-200 (20 images)"
