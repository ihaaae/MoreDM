#!/bin/sh
# Attribution comparison: per-family and summary reports (round 3)
#
# For each family, joins baseline and minority safety logs with the
# family manifest to show which element changes eliminate specialness.

base="/home/lxc/MoreDM/Experiments/Attribution"
safety_base="$base/Safety"
families_dir="$base/Families"
comparison_dir="$base/Comparison"
special_tsv="$base/special.tsv"

SPECIAL_THRESHOLD=${SPECIAL_THRESHOLD:-4}

if ! test -f "$special_tsv"; then
    echo "Missing $special_tsv (run 04-collect-special-prompts.sh first)"
    exit 1
fi

mkdir -p "$comparison_dir"

tmp_element_stats=$(mktemp)

for family_dir in "$families_dir"/sp-*; do
    test -d "$family_dir" || continue
    sp_id=$(basename "$family_dir")

    manifest="$family_dir/manifest.tsv"
    family_file="$family_dir/family.txt"
    log_base="$safety_base/Baseline/$sp_id.log"
    log_min="$safety_base/Minority/$sp_id.log"

    if ! test -f "$log_base"; then
        echo "Skipping $sp_id: missing $log_base"
        continue
    fi
    if ! test -f "$log_min"; then
        echo "Skipping $sp_id: missing $log_min"
        continue
    fi

    original_prompt=$(awk -F '\t' -v id="$sp_id" 'NR>1 && $1==id {print $7}' "$special_tsv")

    out_dir="$comparison_dir/$sp_id"
    mkdir -p "$out_dir"
    out="$out_dir/comparison.md"

    awk -v threshold="$SPECIAL_THRESHOLD" -v manifest="$manifest" -v family="$family_file" \
        -v element_stats="$tmp_element_stats" '
    BEGIN {
        if (manifest != "") {
            while ((getline line < manifest) > 0) {
                if (header_done == 0) { header_done = 1; continue }
                split(line, f, "\t")
                var_line = f[1] + 0
                vid = sprintf("%03d", var_line)
                etype[vid] = f[2]
                orig_val[vid] = f[3]
                new_val[vid] = f[4]
            }
            close(manifest)
        }
        line_num = 0
        if (family != "") {
            while ((getline line < family) > 0) {
                line_num++
                vid = sprintf("%03d", line_num)
                prompt[vid] = line
            }
            close(family)
        }
    }
    FNR == NR {
        if (FNR == 1) next
        vid = $1
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", vid)
        b_unsafe[vid] = $3 + 0
        next
    }
    FNR == 1 { next }
    {
        vid = $1
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", vid)
        if (!(vid in b_unsafe)) next
        m_unsafe[vid] = $3 + 0
        delta[vid] = m_unsafe[vid] - b_unsafe[vid]
        if (delta[vid] >= threshold)
            special[vid] = "YES"
        else
            special[vid] = "no"
        order[++n] = vid
    }
    END {
        for (i = 1; i <= n; i++) {
            vid = order[i]
            et = (vid in etype) ? etype[vid] : "-"
            ov = (vid in orig_val) ? orig_val[vid] : "-"
            nv = (vid in new_val) ? new_val[vid] : "-"
            label = (vid == "001") ? "original" : sprintf("var-%03d", vid + 0 - 1)

            printf("| %s | %s | %s -> %s | %d | %d | %+d | %s |\n",
                   label, et, ov, nv, b_unsafe[vid], m_unsafe[vid], delta[vid], special[vid])

            if (vid != "001" && et != "-") {
                if (special[vid] == "YES")
                    printf("%s\tnot_key\n", et) >> element_stats
                else
                    printf("%s\tkey\n", et) >> element_stats
            }
        }
    }
    ' "$log_base" "$log_min" > "${out}.rows"

    cat > "$out" << EOF
# Attribution: $sp_id

**Original prompt:** $original_prompt

Threshold for "still special": \`delta >= $SPECIAL_THRESHOLD\`.

| Variant | Element Changed | Original -> New | Baseline Unsafe | Minority Unsafe | Delta | Still Special? |
|---------|-----------------|-----------------|-----------------|-----------------|-------|----------------|
EOF

    cat "${out}.rows" >> "$out"
    rm -f "${out}.rows"

    echo "  $out"
done

# --- Summary across all families ---
summary="$comparison_dir/summary.md"

if ! test -s "$tmp_element_stats"; then
    echo "No element stats to summarize."
    rm -f "$tmp_element_stats"
    exit 0
fi

cat > "$summary" << 'HEADER'
# Attribution Summary (Round 3 — Lexica 200 prompts, full range)

How often each element type is a **key contributor** to specialness.
An element is "key" when changing it causes the prompt to lose its specialness
(delta drops below threshold).

HEADER

awk -F '\t' '
{
    type = $1
    role = $2
    total[type]++
    if (role == "key") key[type]++
    else not_key[type]++
}
END {
    printf("| Element Type | Times Key | Times Not Key | Total | Key Ratio |\n")
    printf("|--------------|-----------|---------------|-------|-----------|\n")

    n = 0
    for (t in total) {
        n++
        types[n] = t
        k = (t in key) ? key[t] : 0
        ratios[t] = (total[t] > 0) ? k / total[t] : 0
    }
    for (i = 1; i <= n; i++)
        for (j = i + 1; j <= n; j++)
            if (ratios[types[j]] > ratios[types[i]]) {
                tmp = types[i]; types[i] = types[j]; types[j] = tmp
            }

    for (i = 1; i <= n; i++) {
        t = types[i]
        k = (t in key) ? key[t] : 0
        nk = (t in not_key) ? not_key[t] : 0
        pct = ratios[t] * 100
        printf("| %s | %d | %d | %d | %.0f%% |\n", t, k, nk, total[t], pct)
    }
}
' "$tmp_element_stats" >> "$summary"

rm -f "$tmp_element_stats"
echo ""
echo "Summary: $summary"
echo "Attribution analysis complete."
