#!/bin/sh
# Safety evaluation: SdxlLight-Lexica baseline (50 prompts) using unsafe-diffusion classifier

src="/home/lxc/MoreDM/Experiments/Text2Image/Baseline/SdxlLight-Lexica"
target="/home/lxc/MoreDM/Experiments/Safety/Dataset/SdxlLight-Lexica"
tmp=$(mktemp -d)

pad() { printf "%03d" "$1"; }
safe_num() { jq '.[]' "$1" | grep -c '"0"'; }
unsafe_num() { jq '.[]' "$1" | grep -c '"1"'; }

mkdir -p "$target"

for i in "$src"/*; do
    if test -d "$i"; then
        name=$(basename "$i")
        uv run metrics/unsafe-diffusion/inference.py --images_dir "$i" --output_dir "$tmp"
        mv "$tmp/predictions.json" "$target/$name.json"
    fi
done

rm -rf "$tmp"

out="$target/lexica.log"
awk 'BEGIN {printf "%-8s%-8s%-8s\n", "p-id", "safe", "unsafe"}' > "$out"
for i in $(seq 1 50); do
    name=$(pad "$i")
    json="$target/$name.json"
    if test -f "$json"; then
        echo "$name" "$(safe_num "$json")" "$(unsafe_num "$json")" |
            awk '{printf "%-8s%-8s%-8s\n", $1, $2, $3}' >> "$out"
    fi
done
