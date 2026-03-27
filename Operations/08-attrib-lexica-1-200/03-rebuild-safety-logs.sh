#!/bin/sh
# Rebuild safety logs for Lexica 1-200 (baseline + minority).
# Now includes the freshly evaluated 51-100 minority data.
# Run after 02-evaluate-safety-minority-51-100.sh.

base_tgt="/home/lxc/MoreDM/Experiments/Safety/Dataset/1-SdxlLight-Lexica"
min_tgt="/home/lxc/MoreDM/Experiments/Safety/Minority/SdxlLight-Lexica/default"

safe_num() { jq '.[]' "$1" | grep -c '"0"'; }
unsafe_num() { jq '.[]' "$1" | grep -c '"1"'; }

rebuild_log() {
    tgt="$1"; out="$2"; hi="$3"
    awk 'BEGIN {printf "%-8s%-8s%-8s\n", "p-id", "safe", "unsafe"}' > "$out"
    for i in $(seq 1 "$hi"); do
        name=$(printf "%03d" "$i")
        json="$tgt/$name.json"
        # Fall back to 2-digit name for legacy evals
        if ! test -f "$json"; then
            name=$(printf "%02d" "$i")
            json="$tgt/$name.json"
        fi
        if test -f "$json"; then
            echo "$name" "$(safe_num "$json")" "$(unsafe_num "$json")" |
                awk '{printf "%-8s%-8s%-8s\n", $1, $2, $3}' >> "$out"
        fi
    done
}

echo "Rebuilding baseline log (1-200)..."
rebuild_log "$base_tgt" "$base_tgt/lexica.log" 200
echo "  $(wc -l < "$base_tgt/lexica.log") lines"

echo "Rebuilding minority log (1-200)..."
rebuild_log "$min_tgt" "$min_tgt/lexica.log" 200
echo "  $(wc -l < "$min_tgt/lexica.log") lines"

echo "Done: logs rebuilt for full 1-200 range"
