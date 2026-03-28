#!/bin/sh
# Compare key-element vs neutral-element prompts for the injection experiment.
#
# For each element type, compares:
#   1. Aggregate unsafe rates: key vs neutral (for both Baseline and Minority)
#   2. Per-prompt delta: does the key element increase unsafety?
#   3. Interaction effect: is the key-element boost *larger* under Minority
#      than under Baseline?  (This would confirm that low-density amplifies
#      the effect of certain elements.)
#
# Output: Experiments/Injection/Comparison/{person,artist,mood}/report.md
#         Experiments/Injection/Comparison/summary.md

base="/home/lxc/MoreDM/Experiments/Injection"
safety_base="$base/Safety"
comparison_dir="$base/Comparison"

mkdir -p "$comparison_dir"

summary="$comparison_dir/summary.md"
cat > "$summary" << 'HEADER'
# Template Injection — Summary

Does injecting a "key" element (one that drove unsafety in attribution)
into neutral templates produce more unsafe images than a neutral element?

| Element Type | Method | Key Unsafe Rate | Neutral Unsafe Rate | Δ (key − neutral) | Prompts |
|--------------|--------|-----------------|---------------------|--------------------|---------|
HEADER

for etype in person artist mood; do
    log_base_key="$safety_base/Baseline/$etype/key.log"
    log_base_neutral="$safety_base/Baseline/$etype/neutral.log"
    log_min_key="$safety_base/Minority/$etype/key.log"
    log_min_neutral="$safety_base/Minority/$etype/neutral.log"

    for f in "$log_base_key" "$log_base_neutral" "$log_min_key" "$log_min_neutral"; do
        if ! test -f "$f"; then
            echo "Missing $f — skipping $etype"
            continue 2
        fi
    done

    out_dir="$comparison_dir/$etype"
    mkdir -p "$out_dir"
    report="$out_dir/report.md"

    # Per-element-type report
    cat > "$report" << EOF
# Injection Report: $etype

Compares prompts with **key** $etype elements vs **neutral** $etype elements.

## Aggregate Stats

EOF

    # Compute aggregate stats for all 4 combinations and prompt-wise comparison
    awk '
    FNR == 1 { file_idx++; next }   # skip header of each file

    file_idx == 1 { bk_unsafe[$1] = $3 + 0; bk_total[$1] = ($2 + 0) + ($3 + 0); next }
    file_idx == 2 { bn_unsafe[$1] = $3 + 0; bn_total[$1] = ($2 + 0) + ($3 + 0); next }
    file_idx == 3 { mk_unsafe[$1] = $3 + 0; mk_total[$1] = ($2 + 0) + ($3 + 0); next }
    file_idx == 4 { mn_unsafe[$1] = $3 + 0; mn_total[$1] = ($2 + 0) + ($3 + 0); next }

    END {
        # Aggregate
        for (p in bk_unsafe) { sum_bk += bk_unsafe[p]; tot_bk += bk_total[p] }
        for (p in bn_unsafe) { sum_bn += bn_unsafe[p]; tot_bn += bn_total[p] }
        for (p in mk_unsafe) { sum_mk += mk_unsafe[p]; tot_mk += mk_total[p] }
        for (p in mn_unsafe) { sum_mn += mn_unsafe[p]; tot_mn += mn_total[p] }

        bk_rate = (tot_bk > 0) ? sum_bk / tot_bk * 100 : 0
        bn_rate = (tot_bn > 0) ? sum_bn / tot_bn * 100 : 0
        mk_rate = (tot_mk > 0) ? sum_mk / tot_mk * 100 : 0
        mn_rate = (tot_mn > 0) ? sum_mn / tot_mn * 100 : 0

        printf("| Method   | Key Unsafe | Key Total | Key Rate | Neutral Unsafe | Neutral Total | Neutral Rate | Δ Rate |\n")
        printf("|----------|------------|-----------|----------|----------------|---------------|--------------|--------|\n")
        printf("| Baseline | %d | %d | %.1f%% | %d | %d | %.1f%% | %+.1f%% |\n",
               sum_bk, tot_bk, bk_rate, sum_bn, tot_bn, bn_rate, bk_rate - bn_rate)
        printf("| Minority | %d | %d | %.1f%% | %d | %d | %.1f%% | %+.1f%% |\n",
               sum_mk, tot_mk, mk_rate, sum_mn, tot_mn, mn_rate, mk_rate - mn_rate)

        printf("\n## Interaction Effect\n\n")
        base_boost = bk_rate - bn_rate
        min_boost  = mk_rate - mn_rate
        interaction = min_boost - base_boost
        printf("- Baseline key-element boost: %+.1f%%\n", base_boost)
        printf("- Minority key-element boost: %+.1f%%\n", min_boost)
        printf("- Interaction (Minority boost − Baseline boost): %+.1f%%\n", interaction)
        if (interaction > 0)
            printf("- ⚠ Low-density **amplifies** the effect of key elements.\n")
        else
            printf("- Key-element effect is **not amplified** by low-density.\n")

        # Prompt-wise comparison
        printf("\n## Prompt-wise Comparison\n\n")
        printf("| Prompt | BL-Key Unsafe | BL-Neutral Unsafe | BL Δ | Min-Key Unsafe | Min-Neutral Unsafe | Min Δ | Interaction |\n")
        printf("|--------|---------------|-------------------|------|----------------|---------------------|-------|-------------|\n")

        n = 0
        safer = 0; unsafer = 0; same = 0
        for (p in mk_unsafe) {
            n++
            bd = bk_unsafe[p] - bn_unsafe[p]
            md = mk_unsafe[p] - mn_unsafe[p]
            inter = md - bd
            printf("| %s | %d | %d | %+d | %d | %d | %+d | %+d |\n",
                   p, bk_unsafe[p], bn_unsafe[p], bd,
                   mk_unsafe[p], mn_unsafe[p], md, inter)
            if (inter > 1) unsafer++
            else if (inter < -1) safer++
            else same++
        }

        printf("\n**Prompt-wise interaction**: %d amplified, %d dampened, %d similar (|interaction| <= 1)\n",
               unsafer, safer, same)

        # Write summary lines to stdout after a marker
        printf("\n---SUMMARY---\n")
        printf("| Baseline | %.1f%% | %.1f%% | %+.1f%% | %d |\n", bk_rate, bn_rate, bk_rate - bn_rate, n)
        printf("| Minority | %.1f%% | %.1f%% | %+.1f%% | %d |\n", mk_rate, mn_rate, mk_rate - mn_rate, n)
    }
    ' "$log_base_key" "$log_base_neutral" "$log_min_key" "$log_min_neutral" > "${report}.tmp"

    # Split: everything before ---SUMMARY--- goes to report, after to summary
    sed '/^---SUMMARY---$/,$d' "${report}.tmp" >> "$report"

    # Extract summary lines
    summary_lines=$(sed -n '/^---SUMMARY---$/,$ p' "${report}.tmp" | tail -n +2)
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        printf '| %s %s\n' "$etype" "$line" >> "$summary"
    done << EOF
$summary_lines
EOF

    rm -f "${report}.tmp"
    echo "  $report"
done

echo ""
echo "Summary: $summary"
echo "Injection comparison complete."
