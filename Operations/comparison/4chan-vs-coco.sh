#!/bin/sh
# Safety comparison: 4Chan vs COCO datasets

base="/home/lxc/MoreDM/Experiments/Safety/Dataset"
log_4chan="$base/SdxlLight-4Chan/4chan.log"
log_coco="$base/SdxlLight-COCO/coco.log"
target="$base/SdxlLight-4Chan-COCO/"

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

# Calculate stats for COCO
prompts_coco=$(count_prompts "$log_coco")
safe_coco=$(sum_col "$log_coco" 2)
unsafe_coco=$(sum_col "$log_coco" 3)
images_coco=$((prompts_coco * 10))
safe_pct_coco=$(awk "BEGIN {printf \"%.1f\", $safe_coco / $images_coco * 100}")
unsafe_pct_coco=$(awk "BEGIN {printf \"%.1f\", $unsafe_coco / $images_coco * 100}")

# Generate markdown table
cat > "$target/comparison.md" << EOF
| Dataset | Total Prompts | Total Images | Safe | Safe % | Unsafe | Unsafe % |
|---------|---------------|--------------|------|--------|--------|----------|
| 4Chan   | $prompts_4chan            | $images_4chan          | $safe_4chan  | ${safe_pct_4chan}%  | $unsafe_4chan      | ${unsafe_pct_4chan}%     |
| COCO    | $prompts_coco            | $images_coco          | $safe_coco  | ${safe_pct_coco}%  | $unsafe_coco      | ${unsafe_pct_coco}%     |

EOF

echo "Comparison saved to $target/comparison.md"
