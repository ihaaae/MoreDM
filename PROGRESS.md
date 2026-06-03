# MoreDM Progress

This note tracks the research state of the project. It is intentionally not a
strict chronological engineering log. The goal is to keep the current research
picture clear: what was tested, what the evidence says, and what questions
remain.

## Core Question

The project asks whether low-density prompt optimization, implemented here as
MinorityPrompt, makes text-to-image generation less safe, and which prompt
elements drive that effect.

The most useful object of study is the **special prompt**: a prompt that is
mostly safe under Vanilla generation but becomes substantially more unsafe under
MinorityPrompt.

Current operational rule:

- `vanilla_unsafe <= 3`
- `minority_unsafe - vanilla_unsafe >= 4`

This threshold may change, but the concept is stable: special prompts identify
where low-density optimization creates the clearest safety shift.

## Operations 01-03: Basic Safety Workflow

These experiments establish the baseline comparison between Vanilla SDXL-Light
generation and MinorityPrompt generation.

The original 50-prompt workflow was later expanded on Lexica. That larger
Lexica run is now treated as part of the same 01-03 workflow rather than a
separate experiment branch: Lexica uses prompts `1-200`, and the robustness
expansion raises sampling to `20` images per prompt for both Vanilla and
Minority.

### 01 Generation

The first workflow generates images from the same prompt sets under two
strategies:

- `Vanilla`: ordinary SDXL-Light generation
- `Minority`: low-density prompt optimization

The naming matters because the research contrast is not "baseline" versus a
new model. It is ordinary generation versus a low-density generation strategy.

### 02 Evaluation

Generated images are evaluated with the multi-headed unsafe-diffusion
classifier. The classifier tracks unsafe labels such as sexual, violent,
disturbing, hateful, and political content.

This metric is useful for counting unsafe images, but it is thresholded and
classifier-dependent. Later Q16 work was added partly because this first metric
should not carry the whole safety claim alone.

### 03 Comparison

The first aggregate comparisons on 50 prompts per dataset found:

| Dataset | Vanilla Unsafe % | Minority Unsafe % | Shift   |
|---------|------------------|-------------------|---------|
| 4Chan   | 6.2%             | 6.4%              | +0.2pp  |
| COCO    | 0.0%             | 0.8%              | +0.8pp  |
| Lexica  | 26.8%            | 43.2%             | +16.4pp |

The main result is that the effect is not uniform. Lexica shows a clear
increase in unsafe generations under MinorityPrompt, while the first 4Chan and
COCO samples show little or no aggregate shift.

Prompt-wise Lexica comparison also showed that the effect is concentrated:

| Category    | Count | Ratio |
|-------------|-------|-------|
| Safer       | 3     | 6%    |
| Unsafer     | 12    | 24%   |
| Almost same | 35    | 70%   |

Most prompts do not change much. The research target is therefore the minority
of prompts that become much less safe.

The larger Lexica run matters scientifically because the first Lexica result
was strong but too small to support a stable element-level claim. More prompts
and more images per prompt make it easier to separate recurring special prompts
from one-off stochastic spikes.

### SD 1.x / 2.x MinorityPrompt Generation

The generation code also supports SD 1.5 and SD 2.0 base models:

- `sd15` and `sd20`: baseline Stable Diffusion generation.
- `min-sd15` and `min-sd20`: MinorityPrompt generation with the SD solver.
- SD 2.0 uses `sd2-community/stable-diffusion-2-base`, a public mirror that
  avoids authentication failures from the official gated model ID.
- SD 1.x/2.x MinorityPrompt uses `p_opt_iter=10`, `t_lo=0.9`, and
  `dynamic_pr=False`; the fixed-ratio timing avoids scheduler indexing
  failures seen with the SDXL-Lightning default config.
- Minority SD generation uses a unique placeholder token per prompt/image
  output so repeated multi-image calls do not collide with tokenizer state.

### SD 1.x / 2.x Paired Safety Comparison

Milestone 1 now has matched Vanilla and MinorityPrompt arms for SD 1.5 and SD
2.0 on Lexica: `200` prompts x `10` images per arm and model. The committed
artifacts are:

- `Experiments/Safety/Minority/Sd15-Lexica/default/{predictions,q16_scores}.json`
- `Experiments/Safety/Minority/Sd20-Lexica/default/{predictions,q16_scores}.json`
- `Experiments/Safety/Vanilla/Sd15-Lexica/default/{predictions,q16_scores}.json`
- `Experiments/Safety/Vanilla/Sd20-Lexica/default/{predictions,q16_scores}.json`

The producer script is `Operations/005-score-sd15-sd20-lexica.sh`, using
`lib/eval.py` for unsafe-diffusion labels and `lib/q16_batch.py` for Q16
scores.

Results on this Lexica run:

- SD 1.5 unsafe-diffusion: Minority `228/2000` (`11.40%`) vs Vanilla
  `655/2000` (`32.75%`).
- SD 1.5 Q16: Minority mean `0.3395`, Vanilla mean `0.4848`, mean delta
  `-0.1453`.
- SD 2.0 unsafe-diffusion: Minority `211/2000` (`10.55%`) vs Vanilla
  `780/2000` (`39.00%`).
- SD 2.0 Q16: Minority mean `0.3238`, Vanilla mean `0.5005`, mean delta
  `-0.1767`.

This direction differs from the prior SDXL-Lightning unsafe-diffusion result in
`modules/MoreDM`, where MinorityPrompt increased the measured unsafe rate on
Lexica. The model-scale/configuration flip is an open question; treat it as a
result to investigate rather than a resolved causal claim.

## Operation 04: CLIP Semantic Relatedness

The CLIP experiment tested a naive mechanism hypothesis: perhaps Vanilla
generation suppresses semantically faithful unsafe content, while low-density
generation escapes that suppression and becomes more faithful to the prompt.

The experiment did not support that hypothesis. On the 50-prompt Lexica
comparison:

- more similar under Minority: `19` prompts (`38.0%`)
- less similar under Minority: `30` prompts (`60.0%`)
- Pearson `r = 0.2334` between unsafe delta and CLIP delta

Because lower CLIP distance means stronger prompt-image alignment, Minority
generation was more often less semantically related, not more. This failed
result narrows the mechanism search: minority-induced unsafety does not appear
to be explained by a general increase in prompt faithfulness.

## Operations 05, 07, 08: Attribution

Attribution moved the project from aggregate safety rates to prompt elements.
Instead of only asking whether MinorityPrompt is more unsafe overall, these
experiments ask which parts of a prompt make the minority effect appear.

Current element categories:

- `person`
- `artist`
- `mood`
- `medium`
- `suggestive`

Across attribution rounds, famous human names have been the strongest recurring
candidate. Political and public figures such as Obama, Donald Trump, Hillary
Clinton, Joe Biden, and Mike Pence appear repeatedly in special prompts.

The round-3 Lexica attribution result is the strongest version of this finding:

- `18` special prompts collected
- `12/18` reproduced in the rerun pipeline
- `person` was key in `10/10` cases where it was tested

This does not prove that person names are the only special element class.
Artists, moods, and media can matter too. But `person` is currently the most
consistent and interpretable driver.

## Operation 09: Template Injection

Template injection tests whether suspected key elements can create the effect
when inserted into otherwise neutral prompts.

The strongest result is for person names:

- Vanilla key-element boost: `+38.4%`
- Minority key-element boost: `+52.6%`
- Minority-specific interaction: `+14.2%`

This is closer to causal evidence than post-hoc attribution. It suggests that
named-person elements do not merely appear in unsafe prompts; under low-density
generation, their effect is amplified.

Artist and mood injections did not show the same minority-specific
amplification pattern.

## Operation 10 Q16: Safety Scoring

Q16 was added as a continuous `P(inappropriate)` score. It gives a second safety
metric alongside the binary multi-headed classifier.

On Lexica, Q16 and the multi-headed classifier showed only weak positive
agreement:

- Pearson `r = 0.2109`
- mean classifier unsafe delta: `+1.50`
- mean Q16 delta: `-0.0552`

This means the safety impact is metric-dependent. That is not a failure of Q16;
it is a warning that future claims should distinguish robust safety shifts from
artifacts of a single classifier.

## Current Working Claims

The strongest current claims are:

1. Low-density generation can make a subset of prompts less safe.
2. The effect is prompt-dependent, not a uniform dataset-wide shift.
3. Named people, especially public or political figures, are the clearest
   recurring special-element class.
4. Template injection supports a real interaction between person-name elements
   and MinorityPrompt.
5. Safety conclusions depend on the metric, so binary classifier results should
   be checked against secondary metrics where possible.
6. The SD 1.x/2.x Lexica run shows MinorityPrompt reducing measured unsafe
   rates relative to Vanilla, unlike the earlier SDXL-Lightning
   unsafe-diffusion direction.

## Near-Term Direction

The next research work should focus on making the special-element claim more
robust:

- keep attribution centered on reproduced special prompts rather than one-off
  spikes
- expand controlled injection for person names and plausible confounders
- compare binary classifier and Q16 behavior on the same prompt families
- use more images per prompt and repeated reruns for paper-stage special prompts
- investigate why the SD 1.x/2.x paired Lexica runs reduce measured unsafety
  while the earlier SDXL-Lightning unsafe-diffusion run increased it
- avoid treating operation-folder cleanup as research progress unless it changes
  the experimental evidence
