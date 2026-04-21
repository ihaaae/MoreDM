#!/bin/bash
# Q16 unsafe score for minority/default SdxlLight-Lexica (4-GPU parallel)

src="/home/lxc/MoreDM/Experiments/Text2Image/Minority/SdxlLight-Lexica/default"
target="/home/lxc/MoreDM/Experiments/Safety/Minority/SdxlLight-Lexica/default-q16"

pad3() { printf "%03d" "$1"; }
pad2() { printf "%02d" "$1"; }
log() { printf "[%s] %s\n" "$(date '+%F %T')" "$*"; }

total_prompts=50
images_per_prompt=10

run_shard() {
    local gpu=$1 start=$2 end=$3
    for i in $(seq "$start" "$end"); do
        pid=$(pad3 "$i")
        mkdir -p "$target/$pid"
        for j in $(seq 1 "$images_per_prompt"); do
            iid=$(pad2 "$j")
            image="$src/$pid/$iid.png"
            if [ ! -f "$image" ]; then
                continue
            fi
            score=$(CUDA_VISIBLE_DEVICES=$gpu uv run /home/lxc/MoreDM/metrics/Q16/bin/q16_score.py --image="$image" | awk -F '\t' 'END{print $NF}')
            echo "$iid $score" >> "$target/$pid/scores.txt"
        done
        log "GPU${gpu} Prompt ${pid} done"
    done
}

log "Starting Q16 score run (minority): prompts=${total_prompts}, images_per_prompt=${images_per_prompt}"

shard=$((total_prompts / 4))
run_shard 0 1 $((shard))       &
run_shard 1 $((shard+1)) $((shard*2))   &
run_shard 2 $((shard*2+1)) $((shard*3)) &
run_shard 3 $((shard*3+1)) $total_prompts &
wait

log "Completed Q16 score run (minority)"
