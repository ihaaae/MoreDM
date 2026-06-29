# Metric Patch: All-Model Minority vs Vanilla (Lexica)

| Run | Images | Unsafe-diffusion unsafe % | Mean Q16 | NudeNet unsafe % | SD safety unsafe % | Mean CLIP similarity | Alignment retention % | Unsafe-diffusion aligned unsafe % | NudeNet aligned unsafe % | Metric disagreements |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla/Sd15-Lexica/default | 2000 | 32.75% | 0.4848 | 2.05% | 16.15% | 0.2474 | 75.00% | 40.53% | 2.60% | 990 |
| Minority/Sd15-Lexica/default | 2000 | 11.40% | 0.3395 | 1.30% | 8.10% | 0.1504 | 75.00% | 13.20% | 1.60% | 378 |
| Vanilla/Sd20-Lexica/default | 2000 | 39.00% | 0.5005 | 3.90% | 11.45% | 0.2579 | 75.00% | 42.27% | 4.00% | 878 |
| Minority/Sd20-Lexica/default | 2000 | 10.55% | 0.3238 | 1.25% | 11.95% | 0.1509 | 75.00% | 13.40% | 1.27% | 429 |
| Vanilla/SdxlLight-Lexica/regenerated-default | 2000 | 28.55% | 0.4189 | 3.80% | 4.15% | 0.2428 | 75.00% | 34.00% | 4.73% | 624 |
| Minority/SdxlLight-Lexica/regenerated-default | 2000 | 37.20% | 0.3480 | 4.15% | 6.65% | 0.2543 | 75.00% | 42.80% | 4.60% | 798 |
| Vanilla/Sd3-Lexica/default | 2000 | 44.15% | 0.4283 | 3.05% | 9.50% | 0.2620 | 75.00% | 49.07% | 2.93% | 961 |
| Minority/Sd3-Lexica/default | 2000 | 28.35% | 0.3754 | 1.60% | 28.15% | 0.2214 | 75.00% | 37.53% | 2.13% | 1054 |
| Vanilla/Sd35-Lexica/default | 2000 | 46.65% | 0.4687 | 3.05% | 8.45% | 0.2642 | 75.00% | 50.47% | 3.60% | 975 |
| Minority/Sd35-Lexica/default | 2000 | 18.80% | 0.3017 | 1.30% | 27.95% | 0.1897 | 75.00% | 24.87% | 1.73% | 865 |

## Interpretation

- SD1.5 and SD2.0 MinorityPrompt still look much safer than Vanilla across unsafe-diffusion, Q16, NudeNet, and mostly SD safety checker, but this coincides with severe CLIP similarity collapse (`~0.150` vs `~0.247-0.258`).
- Regenerated SDXL-Lightning remains the exception: Minority is higher than Vanilla under unsafe-diffusion, NudeNet, and the SD safety checker, while Q16 is lower and CLIP similarity is slightly higher.
- SD3 and SD3.5 MinorityPrompt now complete the all-model panel. They reduce unsafe-diffusion, Q16, and NudeNet relative to their Vanilla baselines, but the SD safety checker flips direction sharply upward (`SD3: 28.15%` vs `9.50%`; `SD3.5: 27.95%` vs `8.45%`).
- SD3-family MinorityPrompt also lowers CLIP similarity (`SD3: 0.2214` vs `0.2620`; `SD3.5: 0.1897` vs `0.2642`), so part of the apparent safety improvement remains alignment-sensitive.
- Under run-relative alignment thresholds, SD3/SD3.5 Minority aligned unsafe-diffusion rates remain lower than Vanilla, but a shared alignment cutoff is still needed for the strongest cross-strategy claim.
