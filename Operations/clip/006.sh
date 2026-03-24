#!/bin/sh
# CLIP image-wise comparison: Minority/default vs Baseline on Lexica dataset

base="/home/lxc/MoreDM/Experiments/Safety"
dir_baseline="$base/Dataset/1-SdxlLight-Lexica-clip"
dir_minority="$base/Minority/SdxlLight-Lexica/default-clip"
target="$base/Comparison/CLIP-ImageWise-Minority-vs-Baseline-Lexica"
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
    awk -v pid="$pid" 'NF >= 2 { printf("%d\t%d\t%.10f\n", pid + 0, $1 + 0, $2 + 0) }' "$file" >> "$tmp_baseline"
done

for file in "$dir_minority"/*/distances.txt; do
    if ! test -f "$file"; then
        continue
    fi
    pid=$(basename "$(dirname "$file")")
    awk -v pid="$pid" 'NF >= 2 { printf("%d\t%d\t%.10f\n", pid + 0, $1 + 0, $2 + 0) }' "$file" >> "$tmp_minority"
done

if ! test -s "$tmp_baseline"; then
    echo "No baseline CLIP distances found in $dir_baseline"
    rm -f "$tmp_baseline" "$tmp_minority" "$tmp_summary" "$tmp_detail"
    exit 1
fi

if ! test -s "$tmp_minority"; then
    echo "No minority CLIP distances found in $dir_minority"
    rm -f "$tmp_baseline" "$tmp_minority" "$tmp_summary" "$tmp_detail"
    exit 1
fi

awk -v tol="$clip_tol" -v detail="$tmp_detail" '
FNR == NR {
    key = $1 ":" $2
    b_dist[key] = $3 + 0
    b_prompt[$1] = 1
    b_pair[key] = 1
    next
}
{
    key = $1 ":" $2
    m_prompt[$1] = 1
    m_pair[key] = 1

    if (!(key in b_dist)) {
        minority_only_images++
        next
    }

    delta = ($3 + 0) - b_dist[key]
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
    sum_b += b_dist[key]
    sum_m += $3 + 0
    sum_delta += delta
    printf("%06d\t%04d\t%.10f\t%.10f\t%+.10f\t%s\n", $1 + 0, $2 + 0, b_dist[key], $3 + 0, delta, category) >> detail
}
END {
    for (key in b_pair) {
        if (!(key in m_pair)) baseline_only_images++
    }
    for (pid in b_prompt) {
        baseline_prompts++
        if (!(pid in m_prompt)) baseline_only_prompts++
    }
    for (pid in m_prompt) {
        minority_prompts++
        if (!(pid in b_prompt)) minority_only_prompts++
    }

    if (compared == 0) {
        exit 1
    }

    more_pct = more_similar / compared * 100
    less_pct = less_similar / compared * 100
    almost_pct = almost_same / compared * 100
    mean_b = sum_b / compared
    mean_m = sum_m / compared
    mean_delta = sum_delta / compared

    printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.1f\t%.1f\t%.1f\t%.10f\t%.10f\t%.10f\n",
           compared, more_similar, less_similar, almost_same, baseline_prompts, minority_prompts,
           baseline_only_prompts, minority_only_prompts, baseline_only_images, minority_only_images,
           more_pct, less_pct, almost_pct, mean_b, mean_m, mean_delta)
}
' "$tmp_baseline" "$tmp_minority" > "$tmp_summary"

if ! test -s "$tmp_summary"; then
    echo "Failed to compare CLIP distances image-wise."
    rm -f "$tmp_baseline" "$tmp_minority" "$tmp_summary" "$tmp_detail"
    exit 1
fi

IFS="$(printf '\t')"
read -r compared more_similar less_similar almost_same baseline_prompts minority_prompts baseline_only_prompts minority_only_prompts baseline_only_images minority_only_images more_pct less_pct almost_pct mean_b mean_m mean_delta < "$tmp_summary"
unset IFS

out="$target/comparison.md"

cat > "$out" << EOF
# CLIP Image-wise Minority vs Baseline: Lexica Dataset

Tolerance for \`almost_same\`: \`|minority_distance - baseline_distance| <= $clip_tol\`.

Lower CLIP distance means image is more similar to the prompt.

| Category | Image Pair Count | Ratio (Compared Pairs) |
|----------|------------------|------------------------|
| More similar (minority lower distance) | $more_similar | ${more_pct}% |
| Less similar (minority higher distance) | $less_similar | ${less_pct}% |
| Almost same | $almost_same | ${almost_pct}% |

| Baseline prompts | Minority prompts | Compared image pairs | Baseline-only prompt IDs | Minority-only prompt IDs | Baseline-only image pairs | Minority-only image pairs |
|------------------|------------------|----------------------|--------------------------|--------------------------|---------------------------|---------------------------|
| $baseline_prompts | $minority_prompts | $compared | $baseline_only_prompts | $minority_only_prompts | $baseline_only_images | $minority_only_images |

| Mean baseline distance | Mean minority distance | Mean delta (Minority - Baseline) |
|------------------------|------------------------|-----------------------------------|
| $mean_b | $mean_m | $mean_delta |

## Image Details

| Prompt ID | Image ID | Baseline Distance | Minority Distance | Delta (Minority - Baseline) | Category |
|-----------|----------|-------------------|-------------------|------------------------------|----------|
EOF

sort "$tmp_detail" | awk -F '\t' '{ printf("| %d | %d | %.6f | %.6f | %+.6f | %s |\n", $1 + 0, $2 + 0, $3, $4, $5, $6) }' >> "$out"

rm -f "$tmp_baseline" "$tmp_minority" "$tmp_summary" "$tmp_detail"
echo "CLIP image-wise comparison saved to $out"
