#!/bin/sh
# Reorganize Operations/ and Experiments/ directory structure.
#
# Operations/:
#   attribution/   -> attrib-all-1-50/        (3 datasets × 50 prompts)
#   attribution2/  -> attrib-lexica-101-200/  (Lexica, skip 51-100; superseded)
#   attribution3/  -> attrib-lexica-1-200/    (Lexica, full 200; current)
#   015-018.sh     -> expand-lexica/          (expand Lexica from 50 to 200)
#   034.sh, 035.sh stay loose (dead — reference deleted Attribution2)
#
# Experiments/:
#   Delete Attribution/  (strict subset of Attribution3)
#   Rename Attribution3/ -> Attribution/

set -e

cd /home/lxc/MoreDM

echo "=== Operations/ renames ==="

# 1. Rename attribution dirs
# attribution and attribution2 are git-tracked; attribution3 is untracked
git mv Operations/attribution  Operations/attrib-all-1-50    2>/dev/null || true
git mv Operations/attribution2 Operations/attrib-lexica-101-200 2>/dev/null || true
mv Operations/attribution3 Operations/attrib-lexica-1-200
git add Operations/attrib-lexica-1-200

# 2. Move 015-018 into expand-lexica
mkdir -p Operations/expand-lexica
git mv Operations/015.sh Operations/expand-lexica/015.sh
git mv Operations/016.sh Operations/expand-lexica/016.sh
git mv Operations/017.sh Operations/expand-lexica/017.sh
git mv Operations/018.sh Operations/expand-lexica/018.sh

echo "=== Experiments/ restructure ==="

# 3. Delete Attribution/ (round 1 — strict subset of round 3)
git rm -r Experiments/Attribution

# 4. Rename Attribution3/ -> Attribution/
git mv Experiments/Attribution3 Experiments/Attribution

echo "=== Update script paths (Experiments/Attribution3 -> Experiments/Attribution) ==="

# attrib-lexica-1-200 scripts: update base= paths
for f in \
    Operations/attrib-lexica-1-200/028.sh \
    Operations/attrib-lexica-1-200/029.sh \
    Operations/attrib-lexica-1-200/030.sh \
    Operations/attrib-lexica-1-200/031.sh \
    Operations/attrib-lexica-1-200/032.sh \
    Operations/attrib-lexica-1-200/033.sh; do
    sed -i 's|Experiments/Attribution3|Experiments/Attribution|g' "$f"
done

# 028.sh: remove the now-pointless "rm -rf Attribution2" line
sed -i '/rm -rf.*Attribution2/d' Operations/attrib-lexica-1-200/028.sh

echo "=== Update .gitignore ==="

# The Attribution/ lines already exist; just remove stale Attribution2 lines
sed -i '/Attribution2/d' Experiments/.gitignore

echo "=== Done ==="
echo "Verify with: git status"
echo "Then update AGENTS.md manually to reflect the new layout."
