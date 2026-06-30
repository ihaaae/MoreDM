# SD-family safety metric comparison

This report summarizes a multi-detector safety pass over existing SD-family
Lexica and Template runs. It keeps the readable research interpretation while
leaving generated JSON/CSV intermediates out of git.

## Provenance

### Scripts

Detector scoring wrappers:

```bash
bash Operations/19-multimetric-safety-sd-family/score-nudenet-sd-family.sh
bash Operations/19-multimetric-safety-sd-family/score-sd-safety-checker-sd-family.sh
```

Aggregation wrapper:

```bash
bash Operations/19-multimetric-safety-sd-family/summarize-safety-metrics-sd-family.sh
```

Reusable CLIs:

- `lib/nudenet_score.py`
- `lib/sd_safety_score.py`
- `lib/safety_aggregate.py`
- `lib/metric_utils.py`

### Source inputs

This report uses existing generated image and metric artifacts under:

- `Experiments/Safety/Vanilla/Sd15-Lexica/default/`
- `Experiments/Safety/Vanilla/Sd20-Lexica/default/`
- `Experiments/Safety/Vanilla/Sd3-Lexica/default/`
- `Experiments/Safety/Vanilla/Sd35-Lexica/default/`
- `Experiments/Safety/Minority/Sd15-Lexica/default/`
- `Experiments/Safety/Minority/Sd20-Lexica/default/`
- `Experiments/Safety/Vanilla/Sd15-Template/default/`
- `Experiments/Safety/Vanilla/Sd20-Template/default/`
- `Experiments/Safety/Vanilla/SdxlLight-Template/default/`
- `Experiments/Safety/Vanilla/Sd3-Template/default/`
- `Experiments/Safety/Vanilla/Sd35-Template/default/`

and corresponding CLIP directories such as
`Experiments/Safety/Vanilla/Sd15-Lexica/default-clip/`.

Generated detector outputs (`nudenet_scores.json`, `sd_safety_scores.json`) and
aggregation intermediates (`summary.json`, `comparison.json`, `imagewise.csv`,
`promptwise.csv`) are intentionally ignored by `Experiments/.gitignore`. They
can be regenerated with the scripts above.

## Scope

This pass uses existing images only. It does not include the unbuilt modern
300-prompt prompt slice because the external prompt-source inputs were not
present, and it does not include person-name control prompts.

The detector panel is:

| Detector | Role | Caveat |
|---|---|---|
| unsafe-diffusion multi-headed classifier | Historical binary unsafe baseline | Existing artifact is binary OR; head-level category outputs require additional scoring. |
| Q16 | Broad continuous inappropriate-content score | Not sexual-content-specific; thresholding should be reported separately from mean score. |
| NudeNet | Sexual/nudity detector | Default exposed-part threshold is `0.45`. |
| Stable Diffusion safety checker | Production-filter reference | Sexual-focused and not ground truth for all harms. |

CLIP values are treated as cosine similarity: higher means better prompt-image
alignment. The summaries below use each run's 25th percentile CLIP similarity
as the alignment cutoff, so alignment retention is 75% by construction.

## Lexica vanilla model family

| Run | Images | Unsafe-diffusion unsafe % | Mean Q16 | NudeNet unsafe % | SD safety unsafe % | Mean CLIP similarity | Unsafe-diffusion aligned unsafe % | NudeNet aligned unsafe % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla/Sd15-Lexica/default | 2000 | 32.75% | 0.4848 | 2.05% | 16.15% | 0.2474 | 40.53% | 2.60% |
| Vanilla/Sd20-Lexica/default | 2000 | 39.00% | 0.5005 | 3.90% | 11.45% | 0.2579 | 42.27% | 4.00% |
| Vanilla/Sd3-Lexica/default | 2000 | 44.15% | 0.4283 | 3.05% | 9.50% | 0.2620 | 49.07% | 2.93% |
| Vanilla/Sd35-Lexica/default | 2000 | 46.65% | 0.4687 | 3.05% | 8.45% | 0.2642 | 50.47% | 3.60% |

Under unsafe-diffusion, SD3 and SD3.5 remain higher-unsafe than SD1.5 and
SD2.0 on Lexica. Under NudeNet and the SD safety checker, the SD3-family result
is not uniformly worse; the conclusion is detector-dependent.

## Template vanilla model family

| Run | Images | Unsafe-diffusion unsafe % | Mean Q16 | NudeNet unsafe % | SD safety unsafe % | Mean CLIP similarity | Unsafe-diffusion aligned unsafe % | NudeNet aligned unsafe % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla/Sd15-Template/default | 300 | 41.33% | 0.5508 | 3.00% | 23.67% | 0.2440 | 54.22% | 4.00% |
| Vanilla/Sd20-Template/default | 300 | 73.67% | 0.6563 | 19.00% | 24.00% | 0.2605 | 77.78% | 19.56% |
| Vanilla/SdxlLight-Template/default | 300 | 54.67% | 0.6041 | 12.67% | 11.67% | 0.2492 | 56.00% | 12.44% |
| Vanilla/Sd3-Template/default | 300 | 71.33% | 0.5427 | 11.00% | 8.00% | 0.2601 | 73.33% | 8.44% |
| Vanilla/Sd35-Template/default | 300 | 77.67% | 0.6458 | 11.67% | 14.00% | 0.2640 | 76.89% | 7.56% |

On Template prompts, unsafe-diffusion remains high for SD2.0, SD3, and SD3.5.
NudeNet concentrates mostly on sexual prompts, while Q16 is highest on several
violent/disturbing Template groups.

## SD1.5 / SD2.0 MinorityPrompt vs Vanilla on Lexica

| Run | Images | Unsafe-diffusion unsafe % | Mean Q16 | NudeNet unsafe % | SD safety unsafe % | Mean CLIP similarity | Unsafe-diffusion aligned unsafe % | NudeNet aligned unsafe % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla/Sd15-Lexica/default | 2000 | 32.75% | 0.4848 | 2.05% | 16.15% | 0.2474 | 40.53% | 2.60% |
| Minority/Sd15-Lexica/default | 2000 | 11.40% | 0.3395 | 1.30% | 8.10% | 0.1504 | 13.20% | 1.60% |
| Vanilla/Sd20-Lexica/default | 2000 | 39.00% | 0.5005 | 3.90% | 11.45% | 0.2579 | 42.27% | 4.00% |
| Minority/Sd20-Lexica/default | 2000 | 10.55% | 0.3238 | 1.25% | 11.95% | 0.1509 | 13.40% | 1.27% |

MinorityPrompt is lower than Vanilla across unsafe-diffusion, Q16, NudeNet, and
usually the SD safety checker for SD1.5/SD2.0, but CLIP similarity also drops
substantially. The safety drop is therefore alignment-sensitive.

## Interpretation

- The SD3-family reversal is strongest for unsafe-diffusion and some
  Template/Q16 views, not universal across detectors.
- NudeNet and the SD safety checker should be treated as complementary views,
  not ground truth.
- Run-relative alignment filtering shows unsafe images are not confined to each
  run's lowest-alignment tail, but it does not prove cross-run alignment parity.
- A stronger MinorityPrompt alignment claim requires a shared comparison-level
  CLIP cutoff.

## Limitations and next steps

- Historical unsafe-diffusion `predictions.json` files are binary; head-level
  category analysis needs a head-level output artifact.
- CLIP similarity is only a prompt-fidelity proxy.
- High-alignment unsafe examples from generated `imagewise.csv` outputs should
  be visually inspected before use as qualitative examples.
- Build and run the modern prompt slice only after source datasets and license
  notes are verified.
