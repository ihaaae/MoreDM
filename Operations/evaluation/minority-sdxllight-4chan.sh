#!/bin/sh
# Safety evaluation: SdxlLight-4Chan Minority/default using unsafe-diffusion classifier

src="/home/lxc/MoreDM/Experiments/Text2Image/Minority/SdxlLight-4Chan/default"
target="/home/lxc/MoreDM/Experiments/Safety/Minority/SdxlLight-4Chan/default"
tmp=$(mktemp -d)

pad() { printf "%03d" "$1"; }
safe_num() { jq '.[]' "$1" | grep -c '"0"'; }
unsafe_num() { jq '.[]' "$1" | grep -c '"1"'; }

echo "src: $src"
echo "target: $target"

mkdir -p "$target"

count=0
for i in "$src"/*; do
    if test -d "$i"; then
        name=$(basename "$i")
        count=$((count + 1))
        echo "[$count/50] $name"
        uv run metrics/unsafe-diffusion/inference.py --images_dir "$i" --output_dir "$tmp"
        mv "$tmp/predictions.json" "$target/$name.json"
    fi
done

rm -rf "$tmp"

out="$target/4chan.log"
awk 'BEGIN {printf "%-8s%-8s%-8s\n", "p-id", "safe", "unsafe"}' > "$out"
for i in $(seq 1 50); do
    name=$(pad "$i")
    json="$target/$name.json"
    if test -f "$json"; then
        echo "$name" "$(safe_num "$json")" "$(unsafe_num "$json")" |
            awk '{printf "%-8s%-8s%-8s\n", $1, $2, $3}' >> "$out"
    fi
done

echo "done: $out"
