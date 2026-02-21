src="/home/lxc/MoreDM/Experiments/Text2Image/Baseline/SdxlLight-Lexica"
prompts="/home/lxc/MoreDM/Datasets/unsafe-diffusion/Lexica.txt"
target="/home/lxc/MoreDM/Experiments/Safety/Dataset/1-SdxlLight-Lexica-clip"

pad3() { printf "%03d" "$1"; }
pad2() { printf "%02d" "$1"; }
log() { printf "[%s] %s\n" "$(date '+%F %T')" "$*"; }

total_prompts=50
images_per_prompt=10
total_images=$((total_prompts * images_per_prompt))
done_images=0

log "Starting CLIP distance run: prompts=${total_prompts}, images_per_prompt=${images_per_prompt}, total_images=${total_images}"

for i in $(seq 1 "$total_prompts"); do
    pid=$(pad3 "$i")
    prompt=$(sed -n "${i}p" "$prompts")
    mkdir -p "$target/$pid"
    log "Prompt ${pid} (${i}/${total_prompts})"
    for j in $(seq 1 "$images_per_prompt"); do
        iid=$(pad2 "$j")
        image="$src/$pid/$iid.png"
        distance=$(uv run metrics/CLIP/bin/clip_distance_binary.py --prompt="$prompt" --image="$image" | awk -F '\t' 'END{print $NF}')
        score=$(printf "%s %s\n" "$iid" "$distance")
        echo "$score" >> "$target/$pid/distances.txt"
        done_images=$((done_images + 1))
        log "  Image ${iid} (${j}/${images_per_prompt}) -> score=${distance} [${done_images}/${total_images}]"
    done
done

log "Completed CLIP distance run"
