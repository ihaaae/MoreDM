#!/bin/sh
# Safety evaluation for expanded Lexica prompts
#   Baseline: 51-200  (150 new evals)
#   Minority: 101-200 (100 new evals; 1-100 json already exists)
# Total: ~250 evals, split across 4 GPUs (~63 each)
# Parallel across 4 GPUs.

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
        if test -d "$src/$name" && ! test -f "$tgt/$name.json"; then
            CUDA_VISIBLE_DEVICES="$gpu" uv run metrics/unsafe-diffusion/inference.py \
                --images_dir "$src/$name" --output_dir "$tmp"
            mv "$tmp/predictions.json" "$tgt/$name.json"
        fi
    done
    rm -rf "$tmp"
}

# GPU 0: baseline 51-113 (63 evals)
eval_range 0 "$base_src" "$base_tgt" 51 113 &

# GPU 1: baseline 114-175 (62 evals)
eval_range 1 "$base_src" "$base_tgt" 114 175 &

# GPU 2: baseline 176-200 + minority 101-138 (25+38=63 evals)
(eval_range 2 "$base_src" "$base_tgt" 176 200 && \
 eval_range 2 "$min_src" "$min_tgt" 101 138) &

# GPU 3: minority 139-200 (62 evals)
eval_range 3 "$min_src" "$min_tgt" 139 200 &

wait
echo "Done: safety eval for expanded Lexica"
