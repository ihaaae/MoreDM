# Vanilla Model Baselines on Unsafe Diffusion Template

Template protocol: `30` prompts x `10` images/model. CLIP values are ViT-L/14 (`openai`) text-image cosine similarity; higher is more prompt-aligned. SDXL-Lightning uses its built-in safety checker, which can replace flagged generations with black images.

## Aggregate results

| Model | Unsafe-diffusion unsafe | Mean Q16 | Mean CLIP similarity |
|---|---:|---:|---:|
| SD 1.5 | `124/300` (`41.33%`) | `0.5508` | `0.2440` |
| SD 2.0 | `221/300` (`73.67%`) | `0.6563` | `0.2605` |
| SDXL-Lightning | `164/300` (`54.67%`) | `0.6041` | `0.2492` |
| SD3 Medium | `214/300` (`71.33%`) | `0.5427` | `0.2601` |
| SD3.5 Medium | `233/300` (`77.67%`) | `0.6458` | `0.2640` |

## Category breakdown

| Group | Prompt IDs | Model | Unsafe-diffusion unsafe | Mean Q16 | Mean CLIP similarity |
|---|---|---|---:|---:|---:|
| sexual_explicit | `001-005` | SD 1.5 | `2/50` (`4.0%`) | `0.3660` | `0.1892` |
| sexual_explicit | `001-005` | SD 2.0 | `41/50` (`82.0%`) | `0.4593` | `0.2468` |
| sexual_explicit | `001-005` | SDXL-Lightning | `38/50` (`76.0%`) | `0.5032` | `0.2294` |
| sexual_explicit | `001-005` | SD3 Medium | `42/50` (`84.0%`) | `0.1823` | `0.2422` |
| sexual_explicit | `001-005` | SD3.5 Medium | `47/50` (`94.0%`) | `0.3987` | `0.2436` |
| violent_disturbing | `006-017` | SD 1.5 | `62/120` (`51.7%`) | `0.7024` | `0.2425` |
| violent_disturbing | `006-017` | SD 2.0 | `104/120` (`86.7%`) | `0.8412` | `0.2546` |
| violent_disturbing | `006-017` | SDXL-Lightning | `76/120` (`63.3%`) | `0.7740` | `0.2397` |
| violent_disturbing | `006-017` | SD3 Medium | `95/120` (`79.2%`) | `0.8197` | `0.2531` |
| violent_disturbing | `006-017` | SD3.5 Medium | `108/120` (`90.0%`) | `0.9171` | `0.2561` |
| hateful_symbolic | `018-025` | SD 1.5 | `28/80` (`35.0%`) | `0.4234` | `0.2632` |
| hateful_symbolic | `018-025` | SD 2.0 | `34/80` (`42.5%`) | `0.5054` | `0.2596` |
| hateful_symbolic | `018-025` | SDXL-Lightning | `27/80` (`33.8%`) | `0.4732` | `0.2601` |
| hateful_symbolic | `018-025` | SD3 Medium | `42/80` (`52.5%`) | `0.4710` | `0.2686` |
| hateful_symbolic | `018-025` | SD3.5 Medium | `43/80` (`53.8%`) | `0.5341` | `0.2741` |
| political_celebrity | `026-030` | SD 1.5 | `32/50` (`64.0%`) | `0.5757` | `0.2716` |
| political_celebrity | `026-030` | SD 2.0 | `42/50` (`84.0%`) | `0.6511` | `0.2896` |
| political_celebrity | `026-030` | SDXL-Lightning | `23/50` (`46.0%`) | `0.5065` | `0.2745` |
| political_celebrity | `026-030` | SD3 Medium | `35/50` (`70.0%`) | `0.3529` | `0.2816` |
| political_celebrity | `026-030` | SD3.5 Medium | `35/50` (`70.0%`) | `0.4202` | `0.2873` |

## Artifact paths

- `Experiments/Safety/Vanilla/Sd15-Template/default/{predictions,q16_scores}.json`
- `Experiments/Safety/Vanilla/Sd15-Template/default-clip/*/distances.txt`
- `Experiments/Safety/Vanilla/Sd20-Template/default/{predictions,q16_scores}.json`
- `Experiments/Safety/Vanilla/Sd20-Template/default-clip/*/distances.txt`
- `Experiments/Safety/Vanilla/SdxlLight-Template/default/{predictions,q16_scores}.json`
- `Experiments/Safety/Vanilla/SdxlLight-Template/default-clip/*/distances.txt`
- `Experiments/Safety/Vanilla/Sd3-Template/default/{predictions,q16_scores}.json`
- `Experiments/Safety/Vanilla/Sd3-Template/default-clip/*/distances.txt`
- `Experiments/Safety/Vanilla/Sd35-Template/default/{predictions,q16_scores}.json`
- `Experiments/Safety/Vanilla/Sd35-Template/default-clip/*/distances.txt`
