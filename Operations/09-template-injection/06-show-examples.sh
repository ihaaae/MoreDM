#!/bin/sh
# Show example prompt pairs where the key-element version has a larger
# low-density boost (delta = minority_unsafe − baseline_unsafe) than the
# neutral version.
#
# Filter: delta_key − delta_neutral >= MIN_DIFF
#
# Usage: sh Operations/09-template-injection/06-show-examples.sh [MIN_DIFF]
#   MIN_DIFF: minimum interaction (delta_key − delta_neutral) to show (default: 4)

base="/home/lxc/MoreDM/Experiments/Injection"
MIN_DIFF=${1:-4}

for etype in person artist mood; do
    manifest="$base/$etype/manifest.tsv"
    key_prompts="$base/$etype/prompts_key.txt"
    neutral_prompts="$base/$etype/prompts_neutral.txt"
    bl_key_log="$base/Safety/Baseline/$etype/key.log"
    bl_neutral_log="$base/Safety/Baseline/$etype/neutral.log"
    min_key_log="$base/Safety/Minority/$etype/key.log"
    min_neutral_log="$base/Safety/Minority/$etype/neutral.log"

    for f in "$manifest" "$key_prompts" "$neutral_prompts" \
             "$bl_key_log" "$bl_neutral_log" "$min_key_log" "$min_neutral_log"; do
        test -f "$f" || continue 2
    done

    awk -v etype="$etype" -v min_diff="$MIN_DIFF" \
        -v key_file="$key_prompts" -v neutral_file="$neutral_prompts" '
    FNR == 1 { file_idx++; next }
    file_idx == 1 { bk_unsafe[$1] = $3 + 0; next }
    file_idx == 2 { bn_unsafe[$1] = $3 + 0; next }
    file_idx == 3 { mk_unsafe[$1] = $3 + 0; next }
    file_idx == 4 { mn_unsafe[$1] = $3 + 0; next }
    file_idx == 5 {
        line++
        pid = sprintf("%03d", line)
        dk = mk_unsafe[pid] - bk_unsafe[pid]
        dn = mn_unsafe[pid] - bn_unsafe[pid]
        diff = dk - dn
        if (diff >= min_diff) {
            getline kp < key_file
            getline np < neutral_file
            printf "── [%s] prompt %s  (key: bl=%d min=%d Δ=%+d)  (neutral: bl=%d min=%d Δ=%+d)  interaction:%+d\n", \
                etype, pid, bk_unsafe[pid], mk_unsafe[pid], dk, \
                bn_unsafe[pid], mn_unsafe[pid], dn, diff
            printf "   neutral:  %s\n", np
            printf "   key:      %s\n\n", kp
        } else {
            getline kp < key_file
            getline np < neutral_file
        }
    }
    ' "$bl_key_log" "$bl_neutral_log" "$min_key_log" "$min_neutral_log" "$manifest"
done
