#!/bin/sh
# CLIP prompt-wise comparison: Minority/default vs Baseline on Lexica dataset

base="/home/lxc/MoreDM/Experiments/Safety"
dir_baseline="$base/Dataset/1-SdxlLight-Lexica-clip"
dir_minority="$base/Minority/SdxlLight-Lexica/default-clip"
target="$base/Comparison/CLIP-PromptWise-Minority-vs-Baseline-Lexica"
clip_tol=0.001

if ! test -d "$dir_baseline"; then
    echo "Missing baseline CLIP directory: $dir_baseline"
    exit 1
fi

if ! test -d "$dir_minority"; then
    echo "Missing minority CLIP directory: $dir_minority"
    exit 1
fi

mkdir -p "$target"

tmp_baseline=$(mktemp)
tmp_minority=$(mktemp)
tmp_summary=$(mktemp)
tmp_detail=$(mktemp)

for file in "$dir_baseline"/*/distances.txt; do
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
    ' "$file" >> "$tmp_baseline"
done

for file in "$dir_minority"/*/distances.txt; do
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
    ' "$file" >> "$tmp_minority"
done

if ! test -s "$tmp_baseline"; then
    echo "No baseline CLIP prompt data found in $dir_baseline"
    rm -f "$tmp_baseline" "$tmp_minority" "$tmp_summary" "$tmp_detail"
    exit 1
fi

if ! test -s "$tmp_minority"; then
    echo "No minority CLIP prompt data found in $dir_minority"
    rm -f "$tmp_baseline" "$tmp_minority" "$tmp_summary" "$tmp_detail"
    exit 1
fi

awk -v tol="$clip_tol" -v detail="$tmp_detail" '
FNR == NR {
    b_count[$1] = $2 + 0
    b_mean[$1] = $3 + 0
    has_baseline[$1] = 1
    next
}
{
    pid = $1 + 0
    m_count[pid] = $2 + 0
    m_mean[pid] = $3 + 0
    has_minority[pid] = 1

    if (!(pid in b_mean)) {
        minority_only++
        next
    }

    delta = m_mean[pid] - b_mean[pid]
    if (delta < -tol) {
        category = "more_similar"
        more_similar++
    } else if (delta > tol) {
        category = "less_similar"
        less_similar++
    } else {
        category = "almost_same"
        almost_same++
    }

    compared++
    sum_b += b_mean[pid]
    sum_m += m_mean[pid]
    sum_delta += delta
    printf("%06d\t%d\t%d\t%.10f\t%.10f\t%+.10f\t%s\n",
           pid, b_count[pid], m_count[pid], b_mean[pid], m_mean[pid], delta, category) >> detail
}
END {
    for (pid in has_baseline) {
        baseline_prompts++
        if (!(pid in has_minority)) baseline_only++
    }
    for (pid in has_minority) minority_prompts++

    if (compared == 0) {
        exit 1
    }

    more_pct = more_similar / compared * 100
    less_pct = less_similar / compared * 100
    almost_pct = almost_same / compared * 100
    mean_b = sum_b / compared
    mean_m = sum_m / compared
    mean_delta = sum_delta / compared

    printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.1f\t%.1f\t%.1f\t%.10f\t%.10f\t%.10f\n",
           compared, more_similar, less_similar, almost_same, baseline_prompts, minority_prompts,
           baseline_only, minority_only, more_pct, less_pct, almost_pct, mean_b, mean_m, mean_delta)
}
' "$tmp_baseline" "$tmp_minority" > "$tmp_summary"

if ! test -s "$tmp_summary"; then
    echo "Failed to compare CLIP distances prompt-wise."
    rm -f "$tmp_baseline" "$tmp_minority" "$tmp_summary" "$tmp_detail"
    exit 1
fi

IFS="$(printf '\t')"
read -r compared more_similar less_similar almost_same baseline_prompts minority_prompts baseline_only minority_only more_pct less_pct almost_pct mean_b mean_m mean_delta < "$tmp_summary"
unset IFS

out="$target/comparison.md"

cat > "$out" << EOF
# CLIP Prompt-wise Minority vs Baseline: Lexica Dataset

Tolerance for \`almost_same\`: \`|minority_mean_distance - baseline_mean_distance| <= $clip_tol\`.

Lower CLIP distance means image is more similar to the prompt.

| Category | Prompt Count | Ratio (Compared Prompts) |
|----------|--------------|--------------------------|
| More similar (minority lower distance) | $more_similar | ${more_pct}% |
| Less similar (minority higher distance) | $less_similar | ${less_pct}% |
| Almost same | $almost_same | ${almost_pct}% |

| Baseline prompts | Minority prompts | Compared prompts | Baseline-only IDs | Minority-only IDs |
|------------------|------------------|------------------|-------------------|-------------------|
| $baseline_prompts | $minority_prompts | $compared | $baseline_only | $minority_only |

| Mean baseline prompt distance | Mean minority prompt distance | Mean delta (Minority - Baseline) |
|-------------------------------|-------------------------------|-----------------------------------|
| $mean_b | $mean_m | $mean_delta |

## Prompt Details

| Prompt ID | Baseline Images | Minority Images | Baseline Mean Distance | Minority Mean Distance | Delta (Minority - Baseline) | Category |
|-----------|------------------|-----------------|------------------------|------------------------|------------------------------|----------|
EOF

sort "$tmp_detail" | awk -F '\t' '{ printf("| %d | %d | %d | %.6f | %.6f | %+.6f | %s |\n", $1 + 0, $2, $3, $4, $5, $6, $7) }' >> "$out"

rm -f "$tmp_baseline" "$tmp_minority" "$tmp_summary" "$tmp_detail"
echo "CLIP prompt-wise comparison saved to $out"
