#!/bin/sh
# CLIP-vs-classifier relevance: Minority/default vs Baseline on Lexica dataset

base="/home/lxc/MoreDM/Experiments/Safety"
log_baseline="$base/Dataset/1-SdxlLight-Lexica/lexica.log"
log_minority="$base/Minority/SdxlLight-Lexica/default/lexica.log"
clip_baseline="$base/Dataset/1-SdxlLight-Lexica-clip"
clip_minority="$base/Minority/SdxlLight-Lexica/default-clip"
target="$base/Comparison/CLIP-vs-Classifier-Minority-vs-Baseline-Lexica"
unsafe_tol=1
clip_tol=0.001

if ! test -f "$log_baseline"; then
    echo "Missing baseline classifier log: $log_baseline"
    exit 1
fi

if ! test -f "$log_minority"; then
    echo "Missing minority classifier log: $log_minority"
    exit 1
fi

if ! test -d "$clip_baseline"; then
    echo "Missing baseline CLIP directory: $clip_baseline"
    exit 1
fi

if ! test -d "$clip_minority"; then
    echo "Missing minority CLIP directory: $clip_minority"
    exit 1
fi

mkdir -p "$target"

tmp_clip_baseline=$(mktemp)
tmp_clip_minority=$(mktemp)
tmp_clip_joined=$(mktemp)
tmp_summary=$(mktemp)
tmp_detail=$(mktemp)
tmp_matrix=$(mktemp)

for file in "$clip_baseline"/*/distances.txt; do
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
    ' "$file" >> "$tmp_clip_baseline"
done

for file in "$clip_minority"/*/distances.txt; do
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
    ' "$file" >> "$tmp_clip_minority"
done

if ! test -s "$tmp_clip_baseline"; then
    echo "No baseline CLIP prompt data found in $clip_baseline"
    rm -f "$tmp_clip_baseline" "$tmp_clip_minority" "$tmp_clip_joined" "$tmp_summary" "$tmp_detail" "$tmp_matrix"
    exit 1
fi

if ! test -s "$tmp_clip_minority"; then
    echo "No minority CLIP prompt data found in $clip_minority"
    rm -f "$tmp_clip_baseline" "$tmp_clip_minority" "$tmp_clip_joined" "$tmp_summary" "$tmp_detail" "$tmp_matrix"
    exit 1
fi

awk -v tol="$clip_tol" '
FNR == NR {
    b_count[$1] = $2 + 0
    b_mean[$1] = $3 + 0
    next
}
{
    pid = $1 + 0
    if (!(pid in b_mean)) next
    delta = ($3 + 0) - b_mean[pid]
    if (delta < -tol) clip_category = "more_similar"
    else if (delta > tol) clip_category = "less_similar"
    else clip_category = "almost_same"

    printf("%d\t%d\t%d\t%.10f\t%.10f\t%+.10f\t%s\n",
           pid, b_count[pid], $2 + 0, b_mean[pid], $3 + 0, delta, clip_category)
}
' "$tmp_clip_baseline" "$tmp_clip_minority" > "$tmp_clip_joined"

if ! test -s "$tmp_clip_joined"; then
    echo "No overlapping prompt IDs between baseline and minority CLIP data."
    rm -f "$tmp_clip_baseline" "$tmp_clip_minority" "$tmp_clip_joined" "$tmp_summary" "$tmp_detail" "$tmp_matrix"
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
    clip_count_baseline[pid] = $2 + 0
    clip_count_minority[pid] = $3 + 0
    clip_mean_baseline[pid] = $4 + 0
    clip_mean_minority[pid] = $5 + 0
    clip_delta[pid] = $6 + 0
    clip_category[pid] = $7
    has_clip[pid] = 1

    if (!(pid in has_cls_compared)) {
        clip_only++
        next
    }

    safety = cls_category[pid]
    similarity = clip_category[pid]

    compared++
    n_safety[safety]++
    sum_clip_by_safety[safety] += clip_delta[pid]
    if (similarity == "more_similar") more_sim_by_safety[safety]++
    cross[safety "|" similarity]++

    sum_unsafe += unsafe_delta[pid]
    sum_clip += clip_delta[pid]
    sum_unsafe_sq += unsafe_delta[pid] * unsafe_delta[pid]
    sum_clip_sq += clip_delta[pid] * clip_delta[pid]
    sum_cross += unsafe_delta[pid] * clip_delta[pid]

    printf("%06d\t%d\t%d\t%+d\t%s\t%d\t%d\t%.10f\t%.10f\t%+.10f\t%s\n",
           pid, b_unsafe[pid], m_unsafe[pid], unsafe_delta[pid], safety,
           clip_count_baseline[pid], clip_count_minority[pid],
           clip_mean_baseline[pid], clip_mean_minority[pid], clip_delta[pid], similarity) >> detail
}
END {
    for (pid in has_cls_baseline) cls_baseline_prompts++
    for (pid in has_cls_minority) cls_minority_prompts++
    for (pid in has_clip) clip_prompts++
    for (pid in has_cls_compared) {
        if (!(pid in has_clip)) cls_only++
    }

    if (compared == 0) {
        exit 1
    }

    safer_n = n_safety["safer"] + 0
    unsafer_n = n_safety["unsafer"] + 0
    almost_n = n_safety["almost_same"] + 0

    safer_clip_mean = safer_n ? (sum_clip_by_safety["safer"] / safer_n) : 0
    unsafer_clip_mean = unsafer_n ? (sum_clip_by_safety["unsafer"] / unsafer_n) : 0
    almost_clip_mean = almost_n ? (sum_clip_by_safety["almost_same"] / almost_n) : 0

    safer_more_pct = safer_n ? (more_sim_by_safety["safer"] / safer_n * 100) : 0
    unsafer_more_pct = unsafer_n ? (more_sim_by_safety["unsafer"] / unsafer_n * 100) : 0
    almost_more_pct = almost_n ? (more_sim_by_safety["almost_same"] / almost_n * 100) : 0

    numerator = compared * sum_cross - sum_unsafe * sum_clip
    denom_left = compared * sum_unsafe_sq - sum_unsafe * sum_unsafe
    denom_right = compared * sum_clip_sq - sum_clip * sum_clip
    if (denom_left <= 0 || denom_right <= 0) pearson = 0
    else pearson = numerator / sqrt(denom_left * denom_right)

    print "safer\tmore_similar\t" (cross["safer|more_similar"] + 0) > matrix
    print "safer\tless_similar\t" (cross["safer|less_similar"] + 0) >> matrix
    print "safer\talmost_same\t" (cross["safer|almost_same"] + 0) >> matrix
    print "unsafer\tmore_similar\t" (cross["unsafer|more_similar"] + 0) >> matrix
    print "unsafer\tless_similar\t" (cross["unsafer|less_similar"] + 0) >> matrix
    print "unsafer\talmost_same\t" (cross["unsafer|almost_same"] + 0) >> matrix
    print "almost_same\tmore_similar\t" (cross["almost_same|more_similar"] + 0) >> matrix
    print "almost_same\tless_similar\t" (cross["almost_same|less_similar"] + 0) >> matrix
    print "almost_same\talmost_same\t" (cross["almost_same|almost_same"] + 0) >> matrix

    printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.10f\t%.10f\t%.10f\t%.1f\t%.1f\t%.1f\t%.10f\t%.10f\t%.10f\n",
           compared, cls_baseline_prompts, cls_minority_prompts, clip_prompts, cls_only, clip_only,
           safer_n, unsafer_n, almost_n, safer_clip_mean, unsafer_clip_mean, almost_clip_mean,
           safer_more_pct, unsafer_more_pct, almost_more_pct, pearson, sum_unsafe / compared, sum_clip / compared)
}
' "$log_baseline" "$log_minority" "$tmp_clip_joined" > "$tmp_summary"

if ! test -s "$tmp_summary"; then
    echo "Failed to join CLIP and classifier prompt deltas."
    rm -f "$tmp_clip_baseline" "$tmp_clip_minority" "$tmp_clip_joined" "$tmp_summary" "$tmp_detail" "$tmp_matrix"
    exit 1
fi

IFS="$(printf '\t')"
read -r compared cls_baseline_prompts cls_minority_prompts clip_prompts cls_only clip_only safer_n unsafer_n almost_n safer_clip_mean unsafer_clip_mean almost_clip_mean safer_more_pct unsafer_more_pct almost_more_pct pearson mean_unsafe_delta mean_clip_delta < "$tmp_summary"
unset IFS

count_cell() {
    awk -F '\t' -v s="$1" -v c="$2" '$1 == s && $2 == c { print $3; found = 1 } END { if (!found) print 0 }' "$tmp_matrix"
}

s_more=$(count_cell safer more_similar)
s_less=$(count_cell safer less_similar)
s_same=$(count_cell safer almost_same)
u_more=$(count_cell unsafer more_similar)
u_less=$(count_cell unsafer less_similar)
u_same=$(count_cell unsafer almost_same)
a_more=$(count_cell almost_same more_similar)
a_less=$(count_cell almost_same less_similar)
a_same=$(count_cell almost_same almost_same)

out="$target/relevance.md"

cat > "$out" << EOF
# CLIP vs Classifier Relevance: Minority vs Baseline (Lexica)

Classifier prompt category uses \`unsafe_delta = minority_unsafe - baseline_unsafe\` with tolerance \`|unsafe_delta| <= $unsafe_tol\`.

CLIP prompt category uses \`clip_delta = minority_mean_distance - baseline_mean_distance\` with tolerance \`|clip_delta| <= $clip_tol\`.

Lower CLIP distance means image is more similar to the prompt.

| Classifier baseline prompts | Classifier minority prompts | Classifier-only IDs (no CLIP pair) | CLIP prompt IDs | CLIP-only IDs (no classifier pair) | Compared IDs |
|-----------------------------|-----------------------------|-------------------------------------|-----------------|-------------------------------------|--------------|
| $cls_baseline_prompts | $cls_minority_prompts | $cls_only | $clip_prompts | $clip_only | $compared |

| Safety Category (from classifier) | Prompt Count | Mean CLIP Delta | More-similar Ratio |
|-----------------------------------|--------------|-----------------|--------------------|
| Safer | $safer_n | $safer_clip_mean | ${safer_more_pct}% |
| Unsafer | $unsafer_n | $unsafer_clip_mean | ${unsafer_more_pct}% |
| Almost same | $almost_n | $almost_clip_mean | ${almost_more_pct}% |

| Safety Category \\ CLIP Category | More similar | Less similar | Almost same |
|----------------------------------|--------------|--------------|-------------|
| Safer | $s_more | $s_less | $s_same |
| Unsafer | $u_more | $u_less | $u_same |
| Almost same | $a_more | $a_less | $a_same |

| Pearson r (unsafe_delta vs clip_delta) | Mean unsafe delta | Mean CLIP delta |
|-----------------------------------------|-------------------|-----------------|
| $pearson | $mean_unsafe_delta | $mean_clip_delta |

## Prompt Details

| Prompt ID | Baseline Unsafe | Minority Unsafe | Unsafe Delta | Safety Category | Baseline CLIP Mean | Minority CLIP Mean | CLIP Delta | CLIP Category |
|-----------|------------------|-----------------|--------------|-----------------|--------------------|--------------------|------------|---------------|
EOF

sort "$tmp_detail" | awk -F '\t' '{ printf("| %d | %d | %d | %+d | %s | %.6f | %.6f | %+.6f | %s |\n", $1 + 0, $2, $3, $4, $5, $8, $9, $10, $11) }' >> "$out"

rm -f "$tmp_clip_baseline" "$tmp_clip_minority" "$tmp_clip_joined" "$tmp_summary" "$tmp_detail" "$tmp_matrix"
echo "CLIP-vs-classifier relevance report saved to $out"
