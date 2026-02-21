#!/bin/sh
# Safety comparison: 4Chan vs Lexica datasets

base="/home/lxc/MoreDM/Experiments/Safety/Dataset"
log_4chan="$base/SdxlLight-4Chan/4chan.log"
log_lexica="$base/SdxlLight-Lexica/lexica.log"
target="$base/SdxlLight-4Chan-Lexica"

# Sum column from log file (skip header)
sum_col() { awk -v col="$2" 'NR>1 {sum+=$col} END {print sum}' "$1"; }

# Count prompts (lines minus header)
count_prompts() { awk 'NR>1' "$1" | wc -l | tr -d ' '; }

mkdir -p "$target"

# Calculate stats for 4Chan
prompts_4chan=$(count_prompts "$log_4chan")
safe_4chan=$(sum_col "$log_4chan" 2)
unsafe_4chan=$(sum_col "$log_4chan" 3)
images_4chan=$((prompts_4chan * 10))
safe_pct_4chan=$(awk "BEGIN {printf \"%.1f\", $safe_4chan / $images_4chan * 100}")
unsafe_pct_4chan=$(awk "BEGIN {printf \"%.1f\", $unsafe_4chan / $images_4chan * 100}")

# Calculate stats for Lexica
prompts_lexica=$(count_prompts "$log_lexica")
safe_lexica=$(sum_col "$log_lexica" 2)
unsafe_lexica=$(sum_col "$log_lexica" 3)
images_lexica=$((prompts_lexica * 10))
safe_pct_lexica=$(awk "BEGIN {printf \"%.1f\", $safe_lexica / $images_lexica * 100}")
unsafe_pct_lexica=$(awk "BEGIN {printf \"%.1f\", $unsafe_lexica / $images_lexica * 100}")

# Generate markdown table
cat > "$target/comparison.md" << EOF
| Dataset | Total Prompts | Total Images | Safe | Safe % | Unsafe | Unsafe % |
|---------|---------------|--------------|------|--------|--------|----------|
| 4Chan   | $prompts_4chan            | $images_4chan          | $safe_4chan  | ${safe_pct_4chan}%  | $unsafe_4chan      | ${unsafe_pct_4chan}%     |
| Lexica  | $prompts_lexica            | $images_lexica          | $safe_lexica  | ${safe_pct_lexica}%  | $unsafe_lexica      | ${unsafe_pct_lexica}%     |

EOF

echo "Comparison saved to $target/comparison.md"
