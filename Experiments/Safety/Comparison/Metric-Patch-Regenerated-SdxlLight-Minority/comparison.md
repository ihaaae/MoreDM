# Regenerated SDXL-Lightning Lexica Minority vs Vanilla metric patch

This comparison is the retained report view for the regenerated
SDXL-Lightning Lexica runs. The raw generated images, detector outputs, CLIP
per-image scores, and CSV/JSON aggregation intermediates are intentionally not
ported with this report.

| Run | Images | Unsafe-diffusion unsafe % | Mean Q16 | NudeNet unsafe % | SD safety unsafe % | Mean CLIP similarity | Alignment retention % | Unsafe-diffusion aligned unsafe % | NudeNet aligned unsafe % | Metric disagreements |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla/SdxlLight-Lexica/regenerated-default | 2000 | 28.55% | 0.4189 | 3.80% | 4.15% | 0.2428 | 75.00% | 34.00% | 4.73% | 624 |
| Minority/SdxlLight-Lexica/regenerated-default | 2000 | 37.20% | 0.3480 | 4.15% | 6.65% | 0.2543 | 75.00% | 42.80% | 4.60% | 798 |

Regenerated SDXL-Lightning does **not** match the SD1.5/SD2.0 safening
pattern: MinorityPrompt is higher than Vanilla under unsafe-diffusion,
NudeNet, and the SD safety checker, while Q16 is lower and CLIP similarity is
slightly higher.
