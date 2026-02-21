#!/bin/sh
# Safety comparison: COCO vs Lexica datasets

base="/home/lxc/MoreDM/Experiments/Safety/Dataset"
log_coco="$base/SdxlLight-COCO/coco.log"
log_lexica="$base/SdxlLight-Lexica/lexica.log"
target="$base/SdxlLight-COCO-Lexica"

# Sum column from log file (skip header)
sum_col() { awk -v col="$2" 'NR>1 {sum+=$col} END {print sum}' "$1"; }

# Count prompts (lines minus header)
count_prompts() { awk 'NR>1' "$1" | wc -l | tr -d ' '; }

mkdir -p "$target"

# Calculate stats for COCO
prompts_coco=$(count_prompts "$log_coco")
safe_coco=$(sum_col "$log_coco" 2)
unsafe_coco=$(sum_col "$log_coco" 3)
images_coco=$((prompts_coco * 10))
safe_pct_coco=$(awk "BEGIN {printf \"%.1f\", $safe_coco / $images_coco * 100}")
unsafe_pct_coco=$(awk "BEGIN {printf \"%.1f\", $unsafe_coco / $images_coco * 100}")

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
| COCO    | $prompts_coco            | $images_coco          | $safe_coco  | ${safe_pct_coco}%  | $unsafe_coco      | ${unsafe_pct_coco}%     |
| Lexica  | $prompts_lexica            | $images_lexica          | $safe_lexica  | ${safe_pct_lexica}%  | $unsafe_lexica      | ${unsafe_pct_lexica}%     |

EOF

echo "Comparison saved to $target/comparison.md"
