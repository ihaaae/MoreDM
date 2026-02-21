#!/bin/sh
# Safety comparison: Minority/default vs Baseline on Lexica dataset

base="/home/lxc/MoreDM/Experiments/Safety"
log_baseline="$base/Dataset/1-SdxlLight-Lexica/lexica.log"
log_minority="$base/Minority/SdxlLight-Lexica/default/lexica.log"
target="$base/Comparison/Minority-vs-Baseline-Lexica"

# Sum column from log file (skip header)
sum_col() { awk -v col="$2" 'NR>1 {sum+=$col} END {print sum}' "$1"; }

# Count prompts (lines minus header)
count_prompts() { awk 'NR>1' "$1" | wc -l | tr -d ' '; }

mkdir -p "$target"

# Calculate stats for Baseline
prompts_baseline=$(count_prompts "$log_baseline")
safe_baseline=$(sum_col "$log_baseline" 2)
unsafe_baseline=$(sum_col "$log_baseline" 3)
images_baseline=$((prompts_baseline * 10))
safe_pct_baseline=$(awk "BEGIN {printf \"%.1f\", $safe_baseline / $images_baseline * 100}")
unsafe_pct_baseline=$(awk "BEGIN {printf \"%.1f\", $unsafe_baseline / $images_baseline * 100}")

# Calculate stats for Minority/default
prompts_minority=$(count_prompts "$log_minority")
safe_minority=$(sum_col "$log_minority" 2)
unsafe_minority=$(sum_col "$log_minority" 3)
images_minority=$((prompts_minority * 10))
safe_pct_minority=$(awk "BEGIN {printf \"%.1f\", $safe_minority / $images_minority * 100}")
unsafe_pct_minority=$(awk "BEGIN {printf \"%.1f\", $unsafe_minority / $images_minority * 100}")

# Generate markdown table
cat > "$target/comparison.md" << EOF
# Minority vs Baseline: Lexica Dataset

| Method          | Total Prompts | Total Images | Safe | Safe % | Unsafe | Unsafe % |
|-----------------|---------------|--------------|------|--------|--------|----------|
| Baseline        | $prompts_baseline            | $images_baseline          | $safe_baseline  | ${safe_pct_baseline}%  | $unsafe_baseline      | ${unsafe_pct_baseline}%     |
| Minority/default| $prompts_minority            | $images_minority          | $safe_minority  | ${safe_pct_minority}%  | $unsafe_minority      | ${unsafe_pct_minority}%     |

EOF

echo "Comparison saved to $target/comparison.md"
