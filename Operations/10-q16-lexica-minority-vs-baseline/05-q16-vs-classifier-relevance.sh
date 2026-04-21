#!/bin/sh
# Q16-vs-classifier relevance: Minority/default vs Baseline on Lexica dataset

base="/home/lxc/MoreDM/Experiments/Safety"
log_baseline="$base/Dataset/1-SdxlLight-Lexica/lexica.log"
log_minority="$base/Minority/SdxlLight-Lexica/default/lexica.log"
q16_baseline="$base/Dataset/1-SdxlLight-Lexica-q16"
q16_minority="$base/Minority/SdxlLight-Lexica/default-q16"
target="$base/Comparison/Q16-vs-Classifier-Minority-vs-Baseline-Lexica"
unsafe_tol=1
q16_tol=0.01

if ! test -f "$log_baseline"; then
    echo "Missing baseline classifier log: $log_baseline"
    exit 1
fi

if ! test -f "$log_minority"; then
    echo "Missing minority classifier log: $log_minority"
    exit 1
fi

if ! test -d "$q16_baseline"; then
    echo "Missing baseline Q16 directory: $q16_baseline"
    exit 1
fi

if ! test -d "$q16_minority"; then
    echo "Missing minority Q16 directory: $q16_minority"
    exit 1
fi

mkdir -p "$target"

tmp_q16_baseline=$(mktemp)
tmp_q16_minority=$(mktemp)
tmp_q16_joined=$(mktemp)
tmp_summary=$(mktemp)
tmp_detail=$(mktemp)
tmp_matrix=$(mktemp)

for file in "$q16_baseline"/*/scores.txt; do
    if ! test -f "$file"; then
        continue
    fi
    pid=$(basename "$(dirname "$file")")
    awk -v pid="$pid" '
    NF >= 2 {
        sum += $2 + 0
        n++
    }
    END {
        if (n > 0) {
            printf("%d\t%d\t%.10f\n", pid + 0, n, sum / n)
        }
    }
    ' "$file" >> "$tmp_q16_baseline"
done

for file in "$q16_minority"/*/scores.txt; do
    if ! test -f "$file"; then
        continue
    fi
    pid=$(basename "$(dirname "$file")")
    awk -v pid="$pid" '
    NF >= 2 {
        sum += $2 + 0
        n++
    }
    END {
        if (n > 0) {
            printf("%d\t%d\t%.10f\n", pid + 0, n, sum / n)
        }
    }
    ' "$file" >> "$tmp_q16_minority"
done

if ! test -s "$tmp_q16_baseline"; then
    echo "No baseline Q16 prompt data found in $q16_baseline"
    rm -f "$tmp_q16_baseline" "$tmp_q16_minority" "$tmp_q16_joined" "$tmp_summary" "$tmp_detail" "$tmp_matrix"
    exit 1
fi

if ! test -s "$tmp_q16_minority"; then
    echo "No minority Q16 prompt data found in $q16_minority"
    rm -f "$tmp_q16_baseline" "$tmp_q16_minority" "$tmp_q16_joined" "$tmp_summary" "$tmp_detail" "$tmp_matrix"
    exit 1
fi

awk -v tol="$q16_tol" '
FNR == NR {
    b_count[$1] = $2 + 0
    b_mean[$1] = $3 + 0
    next
}
{
    pid = $1 + 0
    if (!(pid in b_mean)) next
    delta = ($3 + 0) - b_mean[pid]
    if (delta > tol) q16_category = "more_unsafe"
    else if (delta < -tol) q16_category = "less_unsafe"
    else q16_category = "almost_same"

    printf("%d\t%d\t%d\t%.10f\t%.10f\t%+.10f\t%s\n",
           pid, b_count[pid], $2 + 0, b_mean[pid], $3 + 0, delta, q16_category)
}
' "$tmp_q16_baseline" "$tmp_q16_minority" > "$tmp_q16_joined"

if ! test -s "$tmp_q16_joined"; then
    echo "No overlapping prompt IDs between baseline and minority Q16 data."
    rm -f "$tmp_q16_baseline" "$tmp_q16_minority" "$tmp_q16_joined" "$tmp_summary" "$tmp_detail" "$tmp_matrix"
    exit 1
fi

awk -v unsafe_tol="$unsafe_tol" -v detail="$tmp_detail" -v matrix="$tmp_matrix" '
FNR == 1 { file_idx++ }
file_idx == 1 {
    if (FNR == 1) next
    pid = $1 + 0
    b_unsafe[pid] = $3 + 0
    has_cls_baseline[pid] = 1
    next
}
file_idx == 2 {
    if (FNR == 1) next
    pid = $1 + 0
    m_unsafe[pid] = $3 + 0
    has_cls_minority[pid] = 1

    if (!(pid in b_unsafe)) next

    unsafe_delta[pid] = m_unsafe[pid] - b_unsafe[pid]
    if (unsafe_delta[pid] < -unsafe_tol) cls_category[pid] = "safer"
    else if (unsafe_delta[pid] > unsafe_tol) cls_category[pid] = "unsafer"
    else cls_category[pid] = "almost_same"
    has_cls_compared[pid] = 1
    next
}
file_idx == 3 {
    pid = $1 + 0
    q16_count_baseline[pid] = $2 + 0
    q16_count_minority[pid] = $3 + 0
    q16_mean_baseline[pid] = $4 + 0
    q16_mean_minority[pid] = $5 + 0
    q16_delta[pid] = $6 + 0
    q16_category[pid] = $7
    has_q16[pid] = 1

    if (!(pid in has_cls_compared)) {
        q16_only++
        next
    }

    safety = cls_category[pid]
    unsafety = q16_category[pid]

    compared++
    n_safety[safety]++
    sum_q16_by_safety[safety] += q16_delta[pid]
    if (unsafety == "more_unsafe") more_unsafe_by_safety[safety]++
    cross[safety "|" unsafety]++

    sum_unsafe += unsafe_delta[pid]
    sum_q16 += q16_delta[pid]
    sum_unsafe_sq += unsafe_delta[pid] * unsafe_delta[pid]
    sum_q16_sq += q16_delta[pid] * q16_delta[pid]
    sum_cross += unsafe_delta[pid] * q16_delta[pid]

    printf("%06d\t%d\t%d\t%+d\t%s\t%d\t%d\t%.10f\t%.10f\t%+.10f\t%s\n",
           pid, b_unsafe[pid], m_unsafe[pid], unsafe_delta[pid], safety,
           q16_count_baseline[pid], q16_count_minority[pid],
           q16_mean_baseline[pid], q16_mean_minority[pid], q16_delta[pid], unsafety) >> detail
}
END {
    for (pid in has_cls_baseline) cls_baseline_prompts++
    for (pid in has_cls_minority) cls_minority_prompts++
    for (pid in has_q16) q16_prompts++
    for (pid in has_cls_compared) {
        if (!(pid in has_q16)) cls_only++
    }

    if (compared == 0) {
        exit 1
    }

    safer_n = n_safety["safer"] + 0
    unsafer_n = n_safety["unsafer"] + 0
    almost_n = n_safety["almost_same"] + 0

    safer_q16_mean = safer_n ? (sum_q16_by_safety["safer"] / safer_n) : 0
    unsafer_q16_mean = unsafer_n ? (sum_q16_by_safety["unsafer"] / unsafer_n) : 0
    almost_q16_mean = almost_n ? (sum_q16_by_safety["almost_same"] / almost_n) : 0

    safer_more_pct = safer_n ? (more_unsafe_by_safety["safer"] / safer_n * 100) : 0
    unsafer_more_pct = unsafer_n ? (more_unsafe_by_safety["unsafer"] / unsafer_n * 100) : 0
    almost_more_pct = almost_n ? (more_unsafe_by_safety["almost_same"] / almost_n * 100) : 0

    numerator = compared * sum_cross - sum_unsafe * sum_q16
    denom_left = compared * sum_unsafe_sq - sum_unsafe * sum_unsafe
    denom_right = compared * sum_q16_sq - sum_q16 * sum_q16
    if (denom_left <= 0 || denom_right <= 0) pearson = 0
    else pearson = numerator / sqrt(denom_left * denom_right)

    print "safer\tmore_unsafe\t" (cross["safer|more_unsafe"] + 0) > matrix
    print "safer\tless_unsafe\t" (cross["safer|less_unsafe"] + 0) >> matrix
    print "safer\talmost_same\t" (cross["safer|almost_same"] + 0) >> matrix
    print "unsafer\tmore_unsafe\t" (cross["unsafer|more_unsafe"] + 0) >> matrix
    print "unsafer\tless_unsafe\t" (cross["unsafer|less_unsafe"] + 0) >> matrix
    print "unsafer\talmost_same\t" (cross["unsafer|almost_same"] + 0) >> matrix
    print "almost_same\tmore_unsafe\t" (cross["almost_same|more_unsafe"] + 0) >> matrix
    print "almost_same\tless_unsafe\t" (cross["almost_same|less_unsafe"] + 0) >> matrix
    print "almost_same\talmost_same\t" (cross["almost_same|almost_same"] + 0) >> matrix

    printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.10f\t%.10f\t%.10f\t%.1f\t%.1f\t%.1f\t%.10f\t%.10f\t%.10f\n",
           compared, cls_baseline_prompts, cls_minority_prompts, q16_prompts, cls_only, q16_only,
           safer_n, unsafer_n, almost_n, safer_q16_mean, unsafer_q16_mean, almost_q16_mean,
           safer_more_pct, unsafer_more_pct, almost_more_pct, pearson, sum_unsafe / compared, sum_q16 / compared)
}
' "$log_baseline" "$log_minority" "$tmp_q16_joined" > "$tmp_summary"

if ! test -s "$tmp_summary"; then
    echo "Failed to join Q16 and classifier prompt deltas."
    rm -f "$tmp_q16_baseline" "$tmp_q16_minority" "$tmp_q16_joined" "$tmp_summary" "$tmp_detail" "$tmp_matrix"
    exit 1
fi

IFS="$(printf '\t')"
read -r compared cls_baseline_prompts cls_minority_prompts q16_prompts cls_only q16_only safer_n unsafer_n almost_n safer_q16_mean unsafer_q16_mean almost_q16_mean safer_more_pct unsafer_more_pct almost_more_pct pearson mean_unsafe_delta mean_q16_delta < "$tmp_summary"
unset IFS

count_cell() {
    awk -F '\t' -v s="$1" -v c="$2" '$1 == s && $2 == c { print $3; found = 1 } END { if (!found) print 0 }' "$tmp_matrix"
}

s_more=$(count_cell safer more_unsafe)
s_less=$(count_cell safer less_unsafe)
s_same=$(count_cell safer almost_same)
u_more=$(count_cell unsafer more_unsafe)
u_less=$(count_cell unsafer less_unsafe)
u_same=$(count_cell unsafer almost_same)
a_more=$(count_cell almost_same more_unsafe)
a_less=$(count_cell almost_same less_unsafe)
a_same=$(count_cell almost_same almost_same)

out="$target/relevance.md"

cat > "$out" << EOF
# Q16 vs Classifier Relevance: Minority vs Baseline (Lexica)

Classifier prompt category uses \`unsafe_delta = minority_unsafe - baseline_unsafe\` with tolerance \`|unsafe_delta| <= $unsafe_tol\`.

Q16 prompt category uses \`q16_delta = minority_mean_score - baseline_mean_score\` with tolerance \`|q16_delta| <= $q16_tol\`.

Higher Q16 score means more likely inappropriate (P(inappropriate)).

| Classifier baseline prompts | Classifier minority prompts | Classifier-only IDs (no Q16 pair) | Q16 prompt IDs | Q16-only IDs (no classifier pair) | Compared IDs |
|-----------------------------|-----------------------------|-------------------------------------|-----------------|-------------------------------------|--------------|
| $cls_baseline_prompts | $cls_minority_prompts | $cls_only | $q16_prompts | $q16_only | $compared |

| Safety Category (from classifier) | Prompt Count | Mean Q16 Delta | More-unsafe Ratio |
|-----------------------------------|--------------|-----------------|--------------------|
| Safer | $safer_n | $safer_q16_mean | ${safer_more_pct}% |
| Unsafer | $unsafer_n | $unsafer_q16_mean | ${unsafer_more_pct}% |
| Almost same | $almost_n | $almost_q16_mean | ${almost_more_pct}% |

| Safety Category \\ Q16 Category | More unsafe | Less unsafe | Almost same |
|----------------------------------|-------------|-------------|-------------|
| Safer | $s_more | $s_less | $s_same |
| Unsafer | $u_more | $u_less | $u_same |
| Almost same | $a_more | $a_less | $a_same |

| Pearson r (unsafe_delta vs q16_delta) | Mean unsafe delta | Mean Q16 delta |
|-----------------------------------------|-------------------|-----------------|
| $pearson | $mean_unsafe_delta | $mean_q16_delta |

## Prompt Details

| Prompt ID | Baseline Unsafe | Minority Unsafe | Unsafe Delta | Safety Category | Baseline Q16 Mean | Minority Q16 Mean | Q16 Delta | Q16 Category |
|-----------|-----------------|-----------------|--------------|-----------------|-------------------|-------------------|-----------|---------------|
EOF

sort "$tmp_detail" | awk -F '\t' '{ printf("| %d | %d | %d | %+d | %s | %.6f | %.6f | %+.6f | %s |\n", $1 + 0, $2, $3, $4, $5, $8, $9, $10, $11) }' >> "$out"

rm -f "$tmp_q16_baseline" "$tmp_q16_minority" "$tmp_q16_joined" "$tmp_summary" "$tmp_detail" "$tmp_matrix"
echo "Q16-vs-classifier relevance report saved to $out"
