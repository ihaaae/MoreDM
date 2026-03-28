#!/bin/sh
# Generate template-based prompts for element injection experiment.
#
# Creates paired prompt files (key vs neutral) for each element type
# (person, artist, mood) under Experiments/Injection/.
#
# Default: 50 prompt pairs per element type.

NUM_PROMPTS=${NUM_PROMPTS:-50}
SEED=${SEED:-42}
outdir="/home/lxc/MoreDM/Experiments/Injection"

echo "Building template prompts: $NUM_PROMPTS pairs per element type..."

uv run bin/make_templates.py \
    --outdir "$outdir" \
    --num-prompts "$NUM_PROMPTS" \
    --seed "$SEED"

echo "Template generation complete."
