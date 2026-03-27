#!/bin/sh
# Collect "special" prompts from Lexica — ALL 200 prompts (no exclusions).
# Replaces Attribution2 which skipped 51-100.
#
# Special := baseline_unsafe <= BASELINE_MAX_UNSAFE AND
#            (minority_unsafe - baseline_unsafe) >= MIN_DELTA
#
# Also erases old Attribution2 results before writing new Attribution3.

base="/home/lxc/MoreDM/Experiments/Safety"
datasets="/home/lxc/MoreDM/Datasets/unsafe-diffusion"
target="/home/lxc/MoreDM/Experiments/Attribution3"

BASELINE_MAX_UNSAFE=${BASELINE_MAX_UNSAFE:-3}
MIN_DELTA=${MIN_DELTA:-4}

echo "Cleaning old Attribution2 outputs..."
rm -rf /home/lxc/MoreDM/Experiments/Attribution2

mkdir -p "$target"

log_base="$base/Dataset/1-SdxlLight-Lexica/lexica.log"
log_min="$base/Minority/SdxlLight-Lexica/default/lexica.log"

if ! test -f "$log_base"; then
    echo "Missing $log_base"
    exit 1
fi
if ! test -f "$log_min"; then
    echo "Missing $log_min"
    exit 1
fi

tmp_hits=$(mktemp)

# Filter: join baseline and minority logs, use ALL 200 prompts
awk -v max_base="$BASELINE_MAX_UNSAFE" -v min_delta="$MIN_DELTA" '
FNR == NR {
    if (FNR == 1) next
    pid = $1 + 0
    b[pid] = $3 + 0
    next
}
FNR == 1 { next }
{
    pid = $1 + 0
    if (!(pid in b)) next
    m = $3 + 0
    d = m - b[pid]
    if (b[pid] <= max_base && d >= min_delta)
        printf("Lexica\t%d\t%d\t%d\t%d\n", pid, b[pid], m, d)
}
' "$log_base" "$log_min" > "$tmp_hits"

if ! test -s "$tmp_hits"; then
    echo "No special prompts found (BASELINE_MAX_UNSAFE=$BASELINE_MAX_UNSAFE, MIN_DELTA=$MIN_DELTA)"
    rm -f "$tmp_hits"
    exit 1
fi

sort -t "$(printf '\t')" -k5 -rn "$tmp_hits" > "${tmp_hits}.sorted"

tsv="$target/special.tsv"
txt="$target/special.txt"

printf 'sp_id\tdataset\tsrc_line\tbaseline_unsafe\tminority_unsafe\tdelta\tprompt\n' > "$tsv"
: > "$txt"

sp_idx=0
while IFS="$(printf '\t')" read -r dataset src_line b_unsafe m_unsafe delta; do
    sp_idx=$((sp_idx + 1))
    sp_id=$(printf "sp-%03d" "$sp_idx")

    prompt_text=$(sed -n "${src_line}p" "$datasets/Lexica.txt")

    printf '%s\t%s\t%d\t%d\t%d\t%d\t' "$sp_id" "$dataset" "$src_line" "$b_unsafe" "$m_unsafe" "$delta" >> "$tsv"
    printf '%s\n' "$prompt_text" >> "$tsv"
    printf '%s\n' "$prompt_text" >> "$txt"
done < "${tmp_hits}.sorted"

rm -f "$tmp_hits" "${tmp_hits}.sorted"

echo "Found $sp_idx special prompt(s) from 200 Lexica prompts (full range 1-200)"
echo "Manifest: $tsv"
echo "Prompts:  $txt"
