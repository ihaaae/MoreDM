# Metric Patch: All-Model Minority vs Vanilla (Lexica)

This comparison extends the SD1.5/SD2.0 Minority-vs-Vanilla table with SDXL-Lightning legacy results and explicit SD3/SD3.5 blocked rows. It does not invent SD3-family MinorityPrompt results: those images and generator support do not exist in the current repo.

| Model | Strategy | Status | n images | Unsafe-diffusion unsafe % | Mean Q16 | NudeNet unsafe % | SD safety unsafe % | Mean CLIP similarity | Unsafe-diffusion aligned unsafe % | Caveat |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| SD1.5 | Vanilla | complete | 2000 | 32.75% | 0.4848 | 2.05% | 16.15% | 0.2474 | 40.53% |  |
| SD1.5 | Minority | complete | 2000 | 11.40% | 0.3395 | 1.30% | 8.10% | 0.1504 | 13.20% |  |
| SD2.0 | Vanilla | complete | 2000 | 39.00% | 0.5005 | 3.90% | 11.45% | 0.2579 | 42.27% |  |
| SD2.0 | Minority | complete | 2000 | 10.55% | 0.3238 | 1.25% | 11.95% | 0.1509 | 13.40% |  |
| SDXL-Lightning | Vanilla | partial | 3990 | 29.82% | 0.3715 | n/a | n/a | 0.2440 | n/a | Original SDXL PNGs are absent; NudeNet and SD safety checker cannot be run without regeneration. Q16/CLIP cover only the older 50-prompt subset. |
| SDXL-Lightning | Minority | partial | 4000 | 36.60% | 0.3163 | n/a | n/a | 0.2561 | n/a | Original SDXL PNGs are absent; NudeNet and SD safety checker cannot be run without regeneration. Q16/CLIP cover only the older 50-prompt subset. |
| SD3 | Vanilla | complete | 2000 | 44.15% | 0.4283 | 3.05% | 9.50% | 0.2620 | 49.07% |  |
| SD3 | Minority | blocked | n/a | n/a | n/a | n/a | n/a | n/a | n/a | No SD3/SD3.5 MinorityPrompt generator exists in lib/gen.py, and no Minority artifacts exist under Experiments/Safety. Implementing this requires a new SD3-flow prompt-optimization solver, not just running existing scripts. |
| SD3.5 | Vanilla | complete | 2000 | 46.65% | 0.4687 | 3.05% | 8.45% | 0.2642 | 50.47% |  |
| SD3.5 | Minority | blocked | n/a | n/a | n/a | n/a | n/a | n/a | n/a | No SD3/SD3.5 MinorityPrompt generator exists in lib/gen.py, and no Minority artifacts exist under Experiments/Safety. Implementing this requires a new SD3-flow prompt-optimization solver, not just running existing scripts. |

## Interpretation

- SD1.5/SD2.0 MinorityPrompt remains lower than Vanilla across the detector panel, while CLIP similarity is much lower than Vanilla.
- SDXL-Lightning legacy results point in the opposite unsafe-diffusion direction: Minority is higher unsafe (`36.6%`) than Vanilla (`29.8%`), while Q16 is lower and CLIP similarity is slightly higher. This is only partial because PNGs are absent, preventing NudeNet/SD-safety scoring.
- SD3/SD3.5 cannot be included as MinorityPrompt experiments yet. The current code supports Vanilla SD3/SD3.5 only; adding MinorityPrompt for these models requires implementing and validating a new prompt-optimization solver for the SD3 pipeline.
