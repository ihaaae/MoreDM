# SD3 / SD3.5 Vanilla Template Comparison

Unsafe Diffusion Template run: `30` prompts x `10` images/model. CLIP values are ViT-L/14 (`openai`) text-image cosine similarity; higher is more prompt-aligned.

## Aggregate results

| Model / dataset | Unsafe-diffusion unsafe | Mean Q16 | Mean CLIP similarity |
|---|---:|---:|---:|
| SD3 Medium Template | `214/300` (`71.33%`) | `0.5427` | `0.2601` |
| SD3.5 Medium Template | `233/300` (`77.67%`) | `0.6458` | `0.2640` |
| SD3 Medium Lexica | `883/2000` (`44.15%`) | `0.4283` | `0.2620` |
| SD3.5 Medium Lexica | `933/2000` (`46.65%`) | `0.4687` | `0.2642` |

## Direct Template delta: SD3.5 - SD3

| Metric | Delta |
|---|---:|
| Unsafe images | `+19/300` |
| Unsafe percentage points | `+6.33pp` |
| Mean Q16 | `+0.1031` |
| Mean CLIP similarity | `+0.0039` |

## Template category breakdown

| Group | Prompt IDs | Model | Unsafe-diffusion unsafe | Mean Q16 | Mean CLIP similarity |
|---|---|---|---:|---:|---:|
| sexual_explicit | `001-005` | SD3 Medium | `42/50` (`84.0%`) | `0.1823` | `0.2422` |
| sexual_explicit | `001-005` | SD3.5 Medium | `47/50` (`94.0%`) | `0.3987` | `0.2436` |
| violent_disturbing | `006-017` | SD3 Medium | `95/120` (`79.2%`) | `0.8197` | `0.2531` |
| violent_disturbing | `006-017` | SD3.5 Medium | `108/120` (`90.0%`) | `0.9171` | `0.2561` |
| hateful_symbolic | `018-025` | SD3 Medium | `42/80` (`52.5%`) | `0.4710` | `0.2686` |
| hateful_symbolic | `018-025` | SD3.5 Medium | `43/80` (`53.8%`) | `0.5341` | `0.2741` |
| political_celebrity | `026-030` | SD3 Medium | `35/50` (`70.0%`) | `0.3529` | `0.2816` |
| political_celebrity | `026-030` | SD3.5 Medium | `35/50` (`70.0%`) | `0.4202` | `0.2873` |

## Artifact paths

- `Experiments/Safety/Vanilla/Sd3-Template/default/{predictions,q16_scores}.json`
- `Experiments/Safety/Vanilla/Sd3-Template/default-clip/*/distances.txt`
- `Experiments/Safety/Vanilla/Sd35-Template/default/{predictions,q16_scores}.json`
- `Experiments/Safety/Vanilla/Sd35-Template/default-clip/*/distances.txt`
