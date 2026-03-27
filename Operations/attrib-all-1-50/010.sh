#!/bin/sh
# Generate prompt families from special prompts for attribution analysis

base="/home/lxc/MoreDM/Experiments/Attribution"
manifest="$base/special.tsv"
outdir="$base/Families"

if ! test -f "$manifest"; then
    echo "Missing manifest: $manifest (run 009.sh first)"
    exit 1
fi

uv run bin/make_families.py --manifest "$manifest" --outdir "$outdir"

echo ""
echo "Families written to $outdir"
echo "Review family.txt and manifest.tsv in each sp-* directory before generation."
