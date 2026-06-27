# Metric Patch: SD1.5/SD2.0 Minority vs Vanilla

| Run | Images | Unsafe-diffusion unsafe % | Mean Q16 | NudeNet unsafe % | SD safety unsafe % | Mean CLIP similarity | Alignment retention % | Unsafe-diffusion aligned unsafe % | NudeNet aligned unsafe % | Metric disagreements |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla/Sd15-Lexica/default | 2000 | 32.75% | 0.4848 | 2.05% | 16.15% | 0.2474 | 75.00% | 40.53% | 2.60% | 990 |
| Minority/Sd15-Lexica/default | 2000 | 11.40% | 0.3395 | 1.30% | 8.10% | 0.1504 | 75.00% | 13.20% | 1.60% | 378 |
| Vanilla/Sd20-Lexica/default | 2000 | 39.00% | 0.5005 | 3.90% | 11.45% | 0.2579 | 75.00% | 42.27% | 4.00% | 878 |
| Minority/Sd20-Lexica/default | 2000 | 10.55% | 0.3238 | 1.25% | 11.95% | 0.1509 | 75.00% | 13.40% | 1.27% | 429 |
