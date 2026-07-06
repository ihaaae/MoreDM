# AGENTS.md - Recipes

Shell scripts in `Recipes/` are experiment blueprints for building or
reproducing artifacts. Numbered subdirectories run sequentially, such as
`01-generation`, `02-evaluation`, and
`03-comparison`.

## Recipe Family Convention

The first recipe family is organized around the basic safety comparison:

- `01-generation/`
- `02-evaluation/`
- `03-comparison/`

This recipe family establishes the baseline object of study:

1. Generate images from the same prompt sets under Vanilla and MinorityPrompt.
2. Evaluate generated images with the multi-headed unsafe-diffusion classifier.
3. Compare safety rates across datasets and between generation strategies.

Use `Vanilla` for ordinary SDXL-Light generation and `Minority` for
low-density prompt optimization.

## Artifact Path Convention

- Vanilla generation: `Experiments/Text2Image/Vanilla/SdxlLight-<dataset>-<subset>/`
- Minority generation: `Experiments/Text2Image/Minority/SdxlLight-<dataset>-<subset>/default/`
- Vanilla safety: `Experiments/Safety/Dataset/SdxlLight-<subset>/`
- Minority safety: `Experiments/Safety/Minority/SdxlLight-<dataset>-<subset>/default/`

## Recipe Style

Recipe shell scripts should be thin experiment orchestration blueprints.
Shared mechanics should live under `lib/`:

- `lib/gen.py` handles generation.
- `lib/eval.py` handles safety evaluation and safety-log rebuilding.
- `lib/compare.py` handles dataset, strategy, and prompt-wise safety comparison.

New scripts should be placed directly in `Recipes/` with sequential names
such as `001.sh`, `002.sh`, and `003.sh`. Move them into numbered
subdirectories after the recipe family is established.

Once scripts live in a numbered subdirectory, prefer descriptive names over
generic numbers. Use lowercase hyphenated names with the pattern
`<verb>-<analysis-kind>-<scope-or-dataset>-<comparison>.sh` when practical.

Comparison scripts should make the granularity explicit:

- `compare-dataset-*`: aggregate whole-dataset safety summaries.
- `compare-strategy-*`: aggregate strategy summaries, such as Minority vs Vanilla.
- `compare-promptwise-*`: prompt-ID-level comparisons.

Read nearby scripts before adding new ones so GPU sharding, paths, logging, and
output naming remain consistent.
