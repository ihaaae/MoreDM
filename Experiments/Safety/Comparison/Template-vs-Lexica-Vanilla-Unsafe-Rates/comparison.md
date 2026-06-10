# Vanilla Unsafe Rate: Template vs Lexica

This compares unsafe-diffusion binary unsafe rates for the five vanilla model baselines on Unsafe Diffusion Template and Lexica.

Protocol caveat: Template uses `30` prompts x `10` images for all five models. Lexica uses `200` prompts x `10` images for SD 1.5, SD 2.0, SD3 Medium, and SD3.5 Medium. The SDXL-Lightning Lexica row uses the earlier committed 50-prompt SDXL-Lightning Lexica baseline summary because this repo does not contain a newer `Vanilla/SdxlLight-Lexica/default/predictions.json` artifact.

| Model | Lexica unsafe | Template unsafe | Template - Lexica |
|---|---:|---:|---:|
| SD 1.5 | `655/2000` (`32.75%`) | `124/300` (`41.33%`) | `+8.58 pp` |
| SD 2.0 | `780/2000` (`39.00%`) | `221/300` (`73.67%`) | `+34.67 pp` |
| SDXL-Lightning | `268/1000` (`26.80%`) | `164/300` (`54.67%`) | `+27.87 pp` |
| SD3 Medium | `883/2000` (`44.15%`) | `214/300` (`71.33%`) | `+27.18 pp` |
| SD3.5 Medium | `933/2000` (`46.65%`) | `233/300` (`77.67%`) | `+31.02 pp` |

## Takeaways

- Template is more unsafe than Lexica for every model, as expected for a concentrated unsafe-template stress test.
- The gap is smallest for SD 1.5 (`+8.58 pp`) and largest for SD 2.0 (`+34.67 pp`).
- Among the newer/higher-capability baselines, SD3.5 Medium has the highest unsafe rate on both datasets: `46.65%` on Lexica and `77.67%` on Template.
- SDXL-Lightning has the lowest Lexica unsafe rate in this table, but that row is less directly comparable because it comes from the older 50-prompt Lexica SDXL baseline rather than the newer 200-prompt Lexica protocol.

## Sources

- Template all-model summary: `Experiments/Safety/Comparison/Template-Vanilla-Model-Baselines/comparison.md`
- Lexica SD 1.5 / SD 2.0 / SD3 / SD3.5 predictions:
  - `Experiments/Safety/Vanilla/Sd15-Lexica/default/predictions.json`
  - `Experiments/Safety/Vanilla/Sd20-Lexica/default/predictions.json`
  - `Experiments/Safety/Vanilla/Sd3-Lexica/default/predictions.json`
  - `Experiments/Safety/Vanilla/Sd35-Lexica/default/predictions.json`
- Lexica SDXL-Lightning baseline summary: `PROGRESS.md` original Lexica result (`26.8%` vanilla unsafe on `50 x 20` images) and `Experiments/Safety/Comparison/Minority-vs-Baseline-Lexica/comparison.md` aggregate baseline row.
