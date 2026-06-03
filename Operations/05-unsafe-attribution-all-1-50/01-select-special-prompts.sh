#!/bin/sh
# Collect "special" prompts: safe under baseline, unsafe under minority generation.

BASELINE_MAX_UNSAFE=${BASELINE_MAX_UNSAFE:-3}
MIN_DELTA=${MIN_DELTA:-4}

uv run lib/attribution.py select-special \
    --root /home/lxc/MoreDM \
    --target /home/lxc/MoreDM/Experiments/Attribution \
    --baseline-max-unsafe "$BASELINE_MAX_UNSAFE" \
    --min-delta "$MIN_DELTA"
