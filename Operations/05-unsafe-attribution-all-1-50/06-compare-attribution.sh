#!/bin/sh
# Attribution comparison: per-family and summary reports.

SPECIAL_THRESHOLD=${SPECIAL_THRESHOLD:-4}

uv run lib/attribution.py compare \
    --base /home/lxc/MoreDM/Experiments/Attribution \
    --special-threshold "$SPECIAL_THRESHOLD"
