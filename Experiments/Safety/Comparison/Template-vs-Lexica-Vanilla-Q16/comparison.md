# Vanilla Q16 Score: Template vs Lexica

This compares mean Q16 `P(inappropriate)` scores for vanilla model baselines on Unsafe Diffusion Template and Lexica. Higher Q16 means more likely inappropriate.

Protocol caveat: Template uses `30` prompts x `10` images for all five models. Lexica Q16 artifacts are available in this repo for SD 1.5, SD 2.0, SD3 Medium, and SD3.5 Medium using `200` prompts x `10` images. This repo does not contain a matching SDXL-Lightning Lexica `q16_scores.json`, so SDXL-Lightning is listed only for Template and has no cross-dataset Q16 delta.

| Model | Lexica mean Q16 | Template mean Q16 | Template - Lexica |
|---|---:|---:|---:|
| SD 1.5 | `0.4848` | `0.5508` | `+0.0660` |
| SD 2.0 | `0.5005` | `0.6563` | `+0.1559` |
| SDXL-Lightning | n/a | `0.6041` | n/a |
| SD3 Medium | `0.4283` | `0.5427` | `+0.1143` |
| SD3.5 Medium | `0.4687` | `0.6458` | `+0.1770` |

## Takeaways

- Template has higher mean Q16 than Lexica for every model with both artifacts available.
- The Q16 increase is smallest for SD 1.5 (`+0.0660`) and largest for SD3.5 Medium (`+0.1770`).
- Q16 ranks the Lexica baselines differently from the binary unsafe-diffusion rate: SD3 Medium has the highest Lexica binary unsafe rate among SD 1.5 / SD 2.0 / SD3 before SD3.5 is added, but SD3 has the lowest mean Lexica Q16 in this comparison.
- SDXL-Lightning Template Q16 is `0.6041`, between SD 1.5 and SD 2.0/SD3.5 on Template, but the missing Lexica Q16 artifact prevents a valid Template-vs-Lexica Q16 delta.

## Sources

- Template all-model summary: `Experiments/Safety/Comparison/Template-Vanilla-Model-Baselines/comparison.md`
- Q16 score artifacts:
  - `Experiments/Safety/Vanilla/Sd15-Lexica/default/q16_scores.json`
  - `Experiments/Safety/Vanilla/Sd15-Template/default/q16_scores.json`
  - `Experiments/Safety/Vanilla/Sd20-Lexica/default/q16_scores.json`
  - `Experiments/Safety/Vanilla/Sd20-Template/default/q16_scores.json`
  - `Experiments/Safety/Vanilla/SdxlLight-Template/default/q16_scores.json`
  - `Experiments/Safety/Vanilla/Sd3-Lexica/default/q16_scores.json`
  - `Experiments/Safety/Vanilla/Sd3-Template/default/q16_scores.json`
  - `Experiments/Safety/Vanilla/Sd35-Lexica/default/q16_scores.json`
  - `Experiments/Safety/Vanilla/Sd35-Template/default/q16_scores.json`
