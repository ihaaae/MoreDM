#!/bin/sh
# Prompt-wise safety comparison: Minority/default vs Baseline on 4Chan dataset

base="/home/lxc/MoreDM/Experiments/Safety"
log_baseline="$base/Dataset/1-SdxlLight-4Chan/4chan.log"
log_minority="$base/Minority/SdxlLight-4Chan/default/4chan.log"
target="$base/Comparison/PromptWise-Minority-vs-Baseline-4Chan"
tol=1

if ! test -f "$log_baseline"; then
    echo "Missing baseline log: $log_baseline"
    exit 1
fi

if ! test -f "$log_minority"; then
    echo "Missing minority log: $log_minority"
    exit 1
fi

mkdir -p "$target"

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

out="$target/comparison.md"

cat > "$out" << EOF
# Prompt-wise Minority vs Baseline: 4Chan Dataset

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
