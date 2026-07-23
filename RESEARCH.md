# UnsafeDistribution Research State

## Research question

This project tests whether low-density prompt optimization can change the
safety of text-to-image generation. The current comparison uses:

- `Vanilla`: ordinary text-to-image generation.
- `Minority`: MinorityPrompt low-density prompt optimization.

The immediate question is not whether MinorityPrompt is universally less safe.
It is whether there are model and dataset settings in which MinorityPrompt
produces more images that a safety detector labels unsafe.

## Current finding

Current evidence shows that such a setting exists. With SDXL-Lightning on the
unsafe-diffusion Lexica prompt subset, Minority generation increased the unsafe
rate measured by the multi-headed unsafe-diffusion classifier.

### Experimental scope

- Model: SDXL-Lightning.
- Prompt source: unsafe-diffusion.
- Main subset: Lexica prompts 1–200.
- Intended sampling: 20 images per prompt and strategy.
- Strategies: Vanilla and MinorityPrompt with the default project
  configuration.
- Safety metric: binary output from the multi-headed unsafe-diffusion
  classifier.
- Prompt-wise tolerance: results differing by at most one unsafe image are
  classified as almost the same.

### Aggregate Lexica result

| Strategy | Prompts | Images | Unsafe images | Unsafe rate |
|----------|--------:|-------:|--------------:|------------:|
| Vanilla | 200 | 3,990 | 1,190 | 29.8% |
| Minority | 200 | 4,000 | 1,464 | 36.6% |

The observed Minority-minus-Vanilla difference is `+6.8` percentage points.
The Vanilla result contains 3,990 rather than the intended 4,000 images. Prompt
146 contains only 10 evaluated images (`5` safe and `5` unsafe), so 10 intended
images are missing from that arm.

### Prompt-wise Lexica result

| Category | Prompts | Share |
|----------|--------:|------:|
| Safer | 32 | 16.0% |
| Unsafer | 66 | 33.0% |
| Almost the same | 102 | 51.0% |

The aggregate increase is therefore not uniform across prompts. One third of
the compared prompts became less safe under the prompt-wise rule, while half
were approximately unchanged.

### Dataset dependence

The smaller 50-prompt comparisons did not show a similarly large shift:

| Dataset | Vanilla unsafe rate | Minority unsafe rate | Difference |
|---------|--------------------:|---------------------:|-----------:|
| 4Chan | 6.2% | 6.4% | +0.2 percentage points |
| COCO | 0.0% | 0.8% | +0.8 percentage points |
| Lexica | 29.8% | 36.6% | +6.8 percentage points |

Lexica used 200 prompts and approximately 4,000 images per strategy; 4Chan and
COCO used 50 prompts and 500 images per strategy. These rows therefore compare
observed dataset-level results, not equally sized estimates.

## Supported conclusion

The current evidence supports this limited claim:

> In at least one tested model and dataset setting, MinorityPrompt generation
> increased detector-labeled unsafety relative to ordinary generation.

This result establishes a research direction rather than a general safety
property. It motivates testing which prompts, models, generation settings, and
safety metrics preserve or reverse the effect.

## Limitations

- The current finding centers on one model and one prompt subset.
- The safety outcome is a classifier measurement, not human ground truth.
- The main comparison uses one detector family.
- The Vanilla Lexica arm is missing 10 intended images from prompt 146.
- The results do not establish statistical significance or causal mechanism.
- The dataset comparisons use different sample sizes.
- The historical PNG corpus used to produce the safety logs is unavailable in
  the current workspace; generated images and raw detector outputs are not
  tracked in Git.

## Next questions

- Does the Lexica result reproduce across independent generation runs?
- Does the direction hold across additional text-to-image models?
- Does it hold under independent safety detectors and human evaluation?
- Which prompt properties concentrate the observed unsafe shift?
- How much of the effect depends on prompt-image alignment?
