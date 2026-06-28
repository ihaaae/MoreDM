# Metric Patch: All-Model Minority vs Vanilla (Lexica)

This comparison now uses full detector-panel rows for SD1.5, SD2.0, and regenerated SDXL-Lightning Lexica Vanilla/Minority runs. It does not invent SD3-family MinorityPrompt results: SD3/SD3.5 Minority generation support is merged and smoke-tested, but full Lexica artifacts have not been generated or scored yet.

| Model | Strategy | Status | n images | Unsafe-diffusion unsafe % | Mean Q16 | NudeNet unsafe % | SD safety unsafe % | Mean CLIP similarity | Unsafe-diffusion aligned unsafe % | Caveat |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| SD1.5 | Vanilla | complete | 2000 | 32.75% | 0.4848 | 2.05% | 16.15% | 0.2474 | 40.53% |  |
| SD1.5 | Minority | complete | 2000 | 11.40% | 0.3395 | 1.30% | 8.10% | 0.1504 | 13.20% |  |
| SD2.0 | Vanilla | complete | 2000 | 39.00% | 0.5005 | 3.90% | 11.45% | 0.2579 | 42.27% |  |
| SD2.0 | Minority | complete | 2000 | 10.55% | 0.3238 | 1.25% | 11.95% | 0.1509 | 13.40% |  |
| SDXL-Lightning | Vanilla | complete-regenerated | 2000 | 28.55% | 0.4189 | 3.80% | 4.15% | 0.2428 | 34.00% | Original historical SDXL PNGs were absent, so this row uses a regenerated 200-prompt Lexica run scored with unsafe-diffusion, Q16, NudeNet, SD safety checker, and CLIP similarity. |
| SDXL-Lightning | Minority | complete-regenerated | 2000 | 37.20% | 0.3480 | 4.15% | 6.65% | 0.2543 | 42.80% | Original historical SDXL PNGs were absent, so this row uses a regenerated 200-prompt Lexica run scored with unsafe-diffusion, Q16, NudeNet, SD safety checker, and CLIP similarity. |
| SD3 | Vanilla | complete | 2000 | 44.15% | 0.4283 | 3.05% | 9.50% | 0.2620 | 49.07% |  |
| SD3 | Minority | not-run | n/a | n/a | n/a | n/a | n/a | n/a | n/a | SD3-family MinorityPrompt generator support is now merged and one-image smoke-tested, but full Lexica Minority artifacts have not been generated or scored yet. |
| SD3.5 | Vanilla | complete | 2000 | 46.65% | 0.4687 | 3.05% | 8.45% | 0.2642 | 50.47% |  |
| SD3.5 | Minority | not-run | n/a | n/a | n/a | n/a | n/a | n/a | n/a | SD3-family MinorityPrompt generator support is now merged and one-image smoke-tested, but full Lexica Minority artifacts have not been generated or scored yet. |

## Interpretation

- SD1.5/SD2.0 MinorityPrompt remains lower than Vanilla across unsafe-diffusion, Q16, NudeNet, and mostly SD safety checker, but this coincides with a large CLIP similarity collapse (`~0.150` vs `~0.247–0.258`).
- Regenerated SDXL-Lightning does **not** match the SD1.5/SD2.0 safening pattern: Minority is higher than Vanilla on unsafe-diffusion (`37.20%` vs `28.55%`), NudeNet (`4.15%` vs `3.80%`), and SD safety checker (`6.65%` vs `4.15%`), while Q16 is lower (`0.3480` vs `0.4189`) and CLIP similarity is slightly higher (`0.2543` vs `0.2428`).
- SD3/SD3.5 still cannot be included as full MinorityPrompt experiments yet. The SD3-family MinorityPrompt generator is now merged and one-image smoke-tested, but full Lexica Minority artifacts still need generation and scoring.
