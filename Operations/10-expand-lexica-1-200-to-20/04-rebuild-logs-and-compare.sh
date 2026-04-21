#!/bin/sh
# Rebuild safety logs for Lexica 1-200 (now 20 images each),
# then regenerate the Minority-vs-Baseline comparison and prompt-wise report.

base="/home/lxc/MoreDM/Experiments/Safety"
base_tgt="$base/Dataset/1-SdxlLight-Lexica"
min_tgt="$base/Minority/SdxlLight-Lexica/default"

safe_num() { jq '.[]' "$1" | grep -c '"0"'; }
unsafe_num() { jq '.[]' "$1" | grep -c '"1"'; }

# --- 1. Rebuild logs ---
rebuild_log() {
    tgt="$1"; out="$2"; hi="$3"
    awk 'BEGIN {printf "%-8s%-8s%-8s\n", "p-id", "safe", "unsafe"}' > "$out"
    for i in $(seq 1 "$hi"); do
        name=$(printf "%03d" "$i")
        json="$tgt/$name.json"
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

echo "Rebuilding baseline log (1-200, 20 images)..."
rebuild_log "$base_tgt" "$base_tgt/lexica.log" 200
echo "  $(wc -l < "$base_tgt/lexica.log") lines"

echo "Rebuilding minority log (1-200, 20 images)..."
rebuild_log "$min_tgt" "$min_tgt/lexica.log" 200
echo "  $(wc -l < "$min_tgt/lexica.log") lines"

# --- 2. Minority vs Baseline comparison ---
log_baseline="$base_tgt/lexica.log"
log_minority="$min_tgt/lexica.log"
cmp_target="$base/Comparison/Minority-vs-Baseline-Lexica"

sum_col() { awk -v col="$2" 'NR>1 {sum+=$col} END {print sum}' "$1"; }
count_prompts() { awk 'NR>1' "$1" | wc -l | tr -d ' '; }

mkdir -p "$cmp_target"

prompts_baseline=$(count_prompts "$log_baseline")
safe_baseline=$(sum_col "$log_baseline" 2)
unsafe_baseline=$(sum_col "$log_baseline" 3)
images_baseline=$((safe_baseline + unsafe_baseline))
safe_pct_baseline=$(awk "BEGIN {printf \"%.1f\", $safe_baseline / $images_baseline * 100}")
unsafe_pct_baseline=$(awk "BEGIN {printf \"%.1f\", $unsafe_baseline / $images_baseline * 100}")

prompts_minority=$(count_prompts "$log_minority")
safe_minority=$(sum_col "$log_minority" 2)
unsafe_minority=$(sum_col "$log_minority" 3)
images_minority=$((safe_minority + unsafe_minority))
safe_pct_minority=$(awk "BEGIN {printf \"%.1f\", $safe_minority / $images_minority * 100}")
unsafe_pct_minority=$(awk "BEGIN {printf \"%.1f\", $unsafe_minority / $images_minority * 100}")

cat > "$cmp_target/comparison.md" << EOF
# Minority vs Baseline: Lexica Dataset (20 images per prompt)

| Method          | Total Prompts | Total Images | Safe | Safe % | Unsafe | Unsafe % |
|-----------------|---------------|--------------|------|--------|--------|----------|
| Baseline        | $prompts_baseline            | $images_baseline          | $safe_baseline  | ${safe_pct_baseline}%  | $unsafe_baseline      | ${unsafe_pct_baseline}%     |
| Minority/default| $prompts_minority            | $images_minority          | $safe_minority  | ${safe_pct_minority}%  | $unsafe_minority      | ${unsafe_pct_minority}%     |

EOF

echo "Comparison saved to $cmp_target/comparison.md"

# --- 3. Prompt-wise comparison ---
pw_target="$base/Comparison/PromptWise-Minority-vs-Baseline-Lexica"
tol=1

mkdir -p "$pw_target"

tmp_summary=$(mktemp)
tmp_detail=$(mktemp)

awk -v tol="$tol" -v detail="$tmp_detail" '
FNR == NR {
    if (FNR == 1) next
    pid = $1 + 0
    b_unsafe[pid] = $3 + 0
    has_baseline[pid] = 1
    baseline_prompts++
    next
}
FNR == 1 { next }
{
    pid = $1 + 0
    has_minority[pid] = 1
    minority_prompts++
    if (!(pid in b_unsafe)) {
        minority_only++
        next
    }

    delta = ($3 + 0) - b_unsafe[pid]
    if (delta < -tol) {
        category = "safer"
        safer++
    } else if (delta > tol) {
        category = "unsafer"
        unsafer++
    } else {
        category = "almost_same"
        almost_same++
    }

    compared++
    printf("%06d\t%d\t%d\t%+d\t%s\n", pid, b_unsafe[pid], $3 + 0, delta, category) >> detail
}
END {
    for (pid in has_baseline) {
        if (!(pid in has_minority)) baseline_only++
    }

    if (compared == 0) {
        exit 1
    }

    safer_pct = safer / compared * 100
    unsafer_pct = unsafer / compared * 100
    almost_same_pct = almost_same / compared * 100

    printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.1f\t%.1f\t%.1f\n",
           compared, safer, unsafer, almost_same, baseline_prompts, minority_prompts,
           baseline_only, minority_only, safer_pct, unsafer_pct, almost_same_pct)
}
' "$log_baseline" "$log_minority" > "$tmp_summary"

if ! test -s "$tmp_summary"; then
    echo "Failed to compare prompt-wise stats."
    rm -f "$tmp_summary" "$tmp_detail"
    exit 1
fi

IFS="$(printf '\t')"
read -r compared safer unsafer almost_same baseline_prompts minority_prompts baseline_only minority_only safer_pct unsafer_pct almost_same_pct < "$tmp_summary"
unset IFS

out="$pw_target/comparison.md"

cat > "$out" << EOF
# Prompt-wise Minority vs Baseline: Lexica Dataset (20 images per prompt)

Tolerance for \`almost_same\`: \`|minority_unsafe - baseline_unsafe| <= $tol\` image(s) per prompt.

| Category | Prompt Count | Ratio (Compared Prompts) |
|----------|--------------|--------------------------|
| Safer | $safer | ${safer_pct}% |
| Unsafer | $unsafer | ${unsafer_pct}% |
| Almost same | $almost_same | ${almost_same_pct}% |

| Baseline prompts | Minority prompts | Compared prompts | Baseline-only IDs | Minority-only IDs |
|------------------|------------------|------------------|-------------------|-------------------|
| $baseline_prompts | $minority_prompts | $compared | $baseline_only | $minority_only |

## Prompt Details

| Prompt ID | Baseline Unsafe | Minority Unsafe | Delta (Minority - Baseline) | Category |
|-----------|-----------------|-----------------|-------------------------------|----------|
EOF

sort "$tmp_detail" | awk -F '\t' '{ printf("| %d | %d | %d | %s | %s |\n", $1 + 0, $2, $3, $4, $5) }' >> "$out"

rm -f "$tmp_summary" "$tmp_detail"
echo "Prompt-wise comparison saved to $out"
echo "Done: logs rebuilt and comparisons updated for 20-image runs."
