# CLIP Prompt-wise Minority vs Baseline: Lexica Dataset

Tolerance for `almost_same`: `|minority_mean_distance - baseline_mean_distance| <= 0.001`.

Lower CLIP distance means image is more similar to the prompt.

| Category | Prompt Count | Ratio (Compared Prompts) |
|----------|--------------|--------------------------|
| More similar (minority lower distance) | 19 | 38.0% |
| Less similar (minority higher distance) | 30 | 60.0% |
| Almost same | 1 | 2.0% |

| Baseline prompts | Minority prompts | Compared prompts | Baseline-only IDs | Minority-only IDs |
|------------------|------------------|------------------|-------------------|-------------------|
| 50 | 50 | 50 | 0 | 0 |

| Mean baseline prompt distance | Mean minority prompt distance | Mean delta (Minority - Baseline) |
|-------------------------------|-------------------------------|-----------------------------------|
| 0.2439832680 | 0.2560949360 | 0.0121116680 |

## Prompt Details

| Prompt ID | Baseline Images | Minority Images | Baseline Mean Distance | Minority Mean Distance | Delta (Minority - Baseline) | Category |
|-----------|------------------|-----------------|------------------------|------------------------|------------------------------|----------|
| 1 | 10 | 10 | 0.302285 | 0.254351 | -0.047933 | more_similar |
| 2 | 10 | 10 | 0.292172 | 0.261667 | -0.030505 | more_similar |
| 3 | 10 | 10 | 0.215565 | 0.267642 | +0.052077 | less_similar |
| 4 | 10 | 10 | 0.258851 | 0.227806 | -0.031045 | more_similar |
| 5 | 10 | 10 | 0.177496 | 0.201218 | +0.023722 | less_similar |
| 6 | 10 | 10 | 0.208142 | 0.190205 | -0.017937 | more_similar |
| 7 | 10 | 10 | 0.217403 | 0.176310 | -0.041093 | more_similar |
| 8 | 10 | 10 | 0.238059 | 0.286159 | +0.048099 | less_similar |
| 9 | 10 | 10 | 0.211916 | 0.240687 | +0.028772 | less_similar |
| 10 | 10 | 10 | 0.228563 | 0.214557 | -0.014006 | more_similar |
| 11 | 10 | 10 | 0.224102 | 0.237268 | +0.013166 | less_similar |
| 12 | 10 | 10 | 0.131992 | 0.130700 | -0.001292 | more_similar |
| 13 | 10 | 10 | 0.310597 | 0.370409 | +0.059812 | less_similar |
| 14 | 10 | 10 | 0.229722 | 0.174006 | -0.055717 | more_similar |
| 15 | 10 | 10 | 0.256721 | 0.243637 | -0.013085 | more_similar |
| 16 | 10 | 10 | 0.213350 | 0.293829 | +0.080479 | less_similar |
| 17 | 10 | 10 | 0.236896 | 0.185658 | -0.051238 | more_similar |
| 18 | 10 | 10 | 0.290403 | 0.334744 | +0.044340 | less_similar |
| 19 | 10 | 10 | 0.224420 | 0.267815 | +0.043395 | less_similar |
| 20 | 10 | 10 | 0.246794 | 0.207291 | -0.039503 | more_similar |
| 21 | 10 | 10 | 0.260655 | 0.250649 | -0.010007 | more_similar |
| 22 | 10 | 10 | 0.239979 | 0.261822 | +0.021843 | less_similar |
| 23 | 10 | 10 | 0.246325 | 0.244896 | -0.001429 | more_similar |
| 24 | 10 | 10 | 0.242789 | 0.327780 | +0.084991 | less_similar |
| 25 | 10 | 10 | 0.257711 | 0.249535 | -0.008176 | more_similar |
| 26 | 10 | 10 | 0.237214 | 0.242335 | +0.005120 | less_similar |
| 27 | 10 | 10 | 0.264024 | 0.234419 | -0.029605 | more_similar |
| 28 | 10 | 10 | 0.217898 | 0.239436 | +0.021538 | less_similar |
| 29 | 10 | 10 | 0.266713 | 0.266428 | -0.000285 | almost_same |
| 30 | 10 | 10 | 0.202859 | 0.279893 | +0.077034 | less_similar |
| 31 | 10 | 10 | 0.257734 | 0.280174 | +0.022441 | less_similar |
| 32 | 10 | 10 | 0.255535 | 0.314928 | +0.059392 | less_similar |
| 33 | 10 | 10 | 0.288275 | 0.300499 | +0.012224 | less_similar |
| 34 | 10 | 10 | 0.221694 | 0.197708 | -0.023987 | more_similar |
| 35 | 10 | 10 | 0.266329 | 0.241083 | -0.025246 | more_similar |
| 36 | 10 | 10 | 0.246881 | 0.279849 | +0.032968 | less_similar |
| 37 | 10 | 10 | 0.276255 | 0.300997 | +0.024742 | less_similar |
| 38 | 10 | 10 | 0.200612 | 0.242340 | +0.041728 | less_similar |
| 39 | 10 | 10 | 0.270166 | 0.299993 | +0.029827 | less_similar |
| 40 | 10 | 10 | 0.268793 | 0.301942 | +0.033149 | less_similar |
| 41 | 10 | 10 | 0.241709 | 0.272324 | +0.030615 | less_similar |
| 42 | 10 | 10 | 0.255819 | 0.208143 | -0.047676 | more_similar |
| 43 | 10 | 10 | 0.287270 | 0.340464 | +0.053194 | less_similar |
| 44 | 10 | 10 | 0.264505 | 0.311601 | +0.047096 | less_similar |
| 45 | 10 | 10 | 0.273012 | 0.246897 | -0.026115 | more_similar |
| 46 | 10 | 10 | 0.242090 | 0.312426 | +0.070337 | less_similar |
| 47 | 10 | 10 | 0.211538 | 0.213081 | +0.001543 | less_similar |
| 48 | 10 | 10 | 0.167558 | 0.198591 | +0.031033 | less_similar |
| 49 | 10 | 10 | 0.275397 | 0.291403 | +0.016006 | less_similar |
| 50 | 10 | 10 | 0.276376 | 0.287153 | +0.010777 | less_similar |
