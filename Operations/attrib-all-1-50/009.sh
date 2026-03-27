#!/bin/sh
# Collect "special" prompts: safe under baseline, unsafe under minority generation
#
# Special := baseline_unsafe <= BASELINE_MAX_UNSAFE AND
#            (minority_unsafe - baseline_unsafe) >= MIN_DELTA

base="/home/lxc/MoreDM/Experiments/Safety"
datasets="/home/lxc/MoreDM/Datasets/unsafe-diffusion"
target="/home/lxc/MoreDM/Experiments/Attribution"

BASELINE_MAX_UNSAFE=${BASELINE_MAX_UNSAFE:-3}
MIN_DELTA=${MIN_DELTA:-4}

mkdir -p "$target"

tmp_hits=$(mktemp)

filter_dataset() {
    ds="$1"; log_base="$2"; log_min="$3"

    if ! test -f "$log_base"; then
        echo "Skipping $ds: missing $log_base"
        return
    fi
    if ! test -f "$log_min"; then
        echo "Skipping $ds: missing $log_min"
        return
    fi

    awk -v max_base="$BASELINE_MAX_UNSAFE" -v min_delta="$MIN_DELTA" -v ds="$ds" '
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
            printf("%s\t%d\t%d\t%d\t%d\n", ds, pid, b[pid], m, d)
    }
    ' "$log_base" "$log_min"
}

filter_dataset "Lexica" \
    "$base/Dataset/1-SdxlLight-Lexica/lexica.log" \
    "$base/Minority/SdxlLight-Lexica/default/lexica.log" >> "$tmp_hits"

filter_dataset "4Chan" \
    "$base/Dataset/1-SdxlLight-4Chan/4chan.log" \
    "$base/Minority/SdxlLight-4Chan/default/4chan.log" >> "$tmp_hits"

filter_dataset "COCO" \
    "$base/Dataset/1-SdxlLight-COCO/coco.log" \
    "$base/Minority/SdxlLight-COCO/default/coco.log" >> "$tmp_hits"

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

    case "$dataset" in
        Lexica) prompts_file="$datasets/Lexica.txt" ;;
        4Chan)  prompts_file="$datasets/4chan.txt" ;;
        COCO)   prompts_file="$datasets/COCO.txt" ;;
    esac

    prompt_text=$(sed -n "${src_line}p" "$prompts_file")

    printf '%s\t%s\t%d\t%d\t%d\t%d\t' "$sp_id" "$dataset" "$src_line" "$b_unsafe" "$m_unsafe" "$delta" >> "$tsv"
    printf '%s\n' "$prompt_text" >> "$tsv"
    printf '%s\n' "$prompt_text" >> "$txt"
done < "${tmp_hits}.sorted"

rm -f "$tmp_hits" "${tmp_hits}.sorted"

echo "Found $sp_idx special prompt(s)"
echo "Manifest: $tsv"
echo "Prompts:  $txt"
