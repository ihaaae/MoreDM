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

### SD3 / SD3.5 Vanilla Lexica Safety Baselines

The vanilla-only SD3-family baseline now covers SD3 Medium and SD3.5 Medium on
the same Lexica protocol used for SD 1.5 / SD 2.0: `200` prompts x `10` images.
SD3.5 generation reuses the already-cached SD3 text encoders/tokenizers and
downloads only SD3.5-specific model components.

Artifacts:

- `Experiments/Safety/Vanilla/Sd3-Lexica/default/{predictions,q16_scores}.json`
- `Experiments/Safety/Vanilla/Sd3-Lexica/default-clip/*/distances.txt`
- `Experiments/Safety/Vanilla/Sd35-Lexica/default/{predictions,q16_scores}.json`
- `Experiments/Safety/Vanilla/Sd35-Lexica/default-clip/*/distances.txt`

Vanilla Lexica comparison:

| Model | Unsafe-diffusion unsafe | Mean Q16 | Mean CLIP similarity |
|-------|--------------------------|----------|----------------------|
| SD 1.5 | `655/2000` (`32.75%`) | `0.4848` | `0.2474` |
| SD 2.0 | `780/2000` (`39.00%`) | `0.5005` | `0.2579` |
| SD3 Medium | `883/2000` (`44.15%`) | `0.4283` | `0.2620` |
| SD3.5 Medium | `933/2000` (`46.65%`) | `0.4687` | `0.2642` |

Direct SD3 -> SD3.5 change: SD3.5 is modestly less safe by both safety metrics
(`+50/2000` unsafe images, `+2.50` percentage points, and `+0.0404` mean Q16),
while CLIP similarity is only slightly higher (`+0.0022`). This suggests the
SD3.5 Medium upgrade improves prompt alignment at most marginally in this setup
and does not provide a vanilla safety improvement over SD3 Medium.

On this prompt set and classifier, the newer vanilla SD3-family models do not
lower the binary unsafe rate; SD3.5 Medium is highest by unsafe-diffusion while
still below SD 1.5 / SD 2.0 on mean Q16. The metric split reinforces the need to
report both binary unsafe labels and continuous Q16 rather than treating either
as the only safety signal.

### SD3 / SD3.5 Vanilla Template Safety Baselines

The same vanilla SD3-family protocol was run on Unsafe Diffusion's Template
prompt set: `30` prompts x `10` images per model. This dataset is much smaller
than Lexica but intentionally concentrated on unsafe prompt templates, so its
unsafe rates should be interpreted as stress-test rates rather than population
rates.

Artifacts:

- `Experiments/Safety/Vanilla/Sd3-Template/default/{predictions,q16_scores}.json`
- `Experiments/Safety/Vanilla/Sd3-Template/default-clip/*/distances.txt`
- `Experiments/Safety/Vanilla/Sd35-Template/default/{predictions,q16_scores}.json`
- `Experiments/Safety/Vanilla/Sd35-Template/default-clip/*/distances.txt`
- `Experiments/Safety/Comparison/Sd3-Sd35-Template/comparison.md`

Vanilla Template comparison:

| Model | Unsafe-diffusion unsafe | Mean Q16 | Mean CLIP similarity |
|-------|--------------------------|----------|----------------------|
| SD3 Medium | `214/300` (`71.33%`) | `0.5427` | `0.2601` |
| SD3.5 Medium | `233/300` (`77.67%`) | `0.6458` | `0.2640` |

Direct SD3 -> SD3.5 change on Template: SD3.5 is again less safe by both safety
metrics (`+19/300` unsafe images, `+6.33` percentage points, and `+0.1031` mean
Q16), while CLIP similarity is only slightly higher (`+0.0039`). The direction
matches Lexica but the magnitude is larger on this concentrated unsafe-template
stress test.

Category-wise, SD3.5 increases unsafe rates most on sexual prompts (`84.0% ->
94.0%`) and violent/disturbing prompts (`79.2% -> 90.0%`), is almost unchanged
on hateful/symbolic prompts (`52.5% -> 53.8%`), and is unchanged by binary rate
on the political/celebrity prompts (`70.0% -> 70.0%`) despite higher mean Q16.

### Vanilla Template Baselines Across SD 1.5 / 2.0 / SDXL / SD3

Extended the Template stress test to vanilla SD 1.5, SD 2.0, and SDXL-Lightning
with the same `30` prompts x `10` images protocol and the same metrics. SDXL
generation required restoring the local SDXL-Lightning 4-step UNet checkpoint;
generated images remain gitignored.

Artifacts:

- `Experiments/Safety/Vanilla/Sd15-Template/default/{predictions,q16_scores}.json`
- `Experiments/Safety/Vanilla/Sd15-Template/default-clip/*/distances.txt`
- `Experiments/Safety/Vanilla/Sd20-Template/default/{predictions,q16_scores}.json`
- `Experiments/Safety/Vanilla/Sd20-Template/default-clip/*/distances.txt`
- `Experiments/Safety/Vanilla/SdxlLight-Template/default/{predictions,q16_scores}.json`
- `Experiments/Safety/Vanilla/SdxlLight-Template/default-clip/*/distances.txt`
- `Experiments/Safety/Comparison/Template-Vanilla-Model-Baselines/comparison.md`

All-model Template comparison:

| Model | Unsafe-diffusion unsafe | Mean Q16 | Mean CLIP similarity |
|-------|--------------------------|----------|----------------------|
| SD 1.5 | `124/300` (`41.33%`) | `0.5508` | `0.2440` |
| SD 2.0 | `221/300` (`73.67%`) | `0.6563` | `0.2605` |
| SDXL-Lightning | `164/300` (`54.67%`) | `0.6041` | `0.2492` |
| SD3 Medium | `214/300` (`71.33%`) | `0.5427` | `0.2601` |
| SD3.5 Medium | `233/300` (`77.67%`) | `0.6458` | `0.2640` |

On Template, SD 1.5 is lowest by the binary unsafe-diffusion rate, while SD3.5
Medium is highest. SD 2.0 is close to SD3 by unsafe-diffusion and highest by
mean Q16. SDXL-Lightning sits between SD 1.5 and the SD2/SD3-family models;
interpret its score with care because its built-in safety checker can replace
flagged generations with black images before downstream scoring.

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

### SD 1.x / 2.x CLIP Alignment Comparison

Ported the CLIP pipeline to the paired SD 1.5 / SD 2.0 arms to ask whether the
SDXL alignment behaviour holds at smaller model scale. Producer scripts
`Operations/006-015-*.sh` score `Vanilla` and `Minority` arms with `ViT-L-14`
(`openai`) on Lexica (10 images/prompt) and write to `…/default-clip/`;
`lib/clip.py` was extended so the relevance step reads the flat
`predictions.json` directly. SD 1.5/2.0 cover all 200 prompts; SDXL uses the
committed 50-prompt run (`Comparison/CLIP-PromptWise-Minority-vs-Baseline-Lexica`),
since SDXL images are not in this repo.

The metric is CLIP text-image **cosine similarity** (higher = better
prompt-image alignment), the value `lib/clip.py` computes and stores — note the
code and the original Op04 report label it "distance," which inverts the
intuition. Typical matched ViT-L cosine similarity is ~`0.25`.

Vanilla alignment is essentially flat across model scale (prompts 1-50,
common with SDXL):

| Model | Vanilla cosine sim |
|-------|--------------------|
| SDXL-Lightning | 0.2440 |
| SD 1.5 | 0.2462 |
| SD 2.0 | 0.2571 |

The Minority effect, by contrast, flips with model scale (prompts 1-50,
`delta = minority - vanilla`):

| Model | Vanilla | Minority | Delta | Relative |
|-------|---------|----------|-------|----------|
| SDXL-Lightning | 0.2440 | 0.2561 | **+0.0121** | +5% |
| SD 1.5 | 0.2462 | 0.1445 | **-0.1017** | -41% |
| SD 2.0 | 0.2571 | 0.1501 | **-0.1070** | -42% |

Full 200-prompt SD means agree: SD 1.5 `0.2474 -> 0.1504`, SD 2.0
`0.2579 -> 0.1509`.

- On SDXL-Lightning, MinorityPrompt leaves alignment roughly intact (a small
  +5% increase); images stay prompt-faithful.
- On SD 1.5 and SD 2.0, MinorityPrompt **collapses alignment by ~40%**; images
  become much less faithful to the prompt.
- This most likely explains the safety-direction flip in
  [SD 1.x / 2.x Paired Safety Comparison](#sd-1x--2x-paired-safety-comparison):
  on the smaller models the `p_opt_iter=10, t_lo=0.9` optimisation appears to
  push samples off-prompt / off-distribution rather than toward faithful-but-rare
  content, so the measured unsafe-rate drop (e.g. SD 1.5 `32.75% -> 11.40%`) may
  be a degradation artifact rather than genuine safening. On SDXL the procedure
  stays on-prompt and unsafe content is preserved.
- Open caveat: SDXL is 50 prompts vs 200 for SD 1.5/2.0, and the ~40% drop is
  large enough to warrant eyeballing a few SD 1.5/2.0 Minority images to confirm
  visual degradation rather than a scoring artifact.

**Known inconsistency — flagged, NOT fixed (needs a cleaner experiment).** The
stored CLIP value is cosine **similarity** (higher = better prompt match;
confirmed empirically: an image vs its own prompt scores ~`0.27-0.29`, vs
unrelated prompts ~`0.04-0.11`). But `lib/clip.py` and the original Op04 report
label it "distance" and read it as lower = better. Consequently the original
50-prompt SDXL prose ("Minority more often *less* semantically related") and the
table above (SDXL Minority `+0.012`, i.e. marginally *more* aligned) interpret
the **same numbers in opposite directions**. Both are intentionally left
unchanged for now. A clean redo should: (1) rename the metric to "similarity"
end-to-end so the sign is unambiguous, and (2) re-run the SDXL vanilla-vs-Minority
comparison at the full 200 prompts to match SD 1.5/2.0. On the current 50-prompt
data, SDXL Minority is the more prompt-aligned arm (`0.2561` vs `0.2440`;
30/50 prompts favour Minority), but the margin is small. The 200-prompt SDXL
re-run is **currently blocked**: only the SDXL `predictions.json` are in this
repo — the SDXL PNGs are absent (originals lived under `/home/lxc/MoreDM`), so
SDXL images would have to be regenerated first.

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

## Operation 19: Benchmark Metric Patch

The benchmark patch added NudeNet and the Stable Diffusion safety checker to the
existing unsafe-diffusion + Q16 panel, then aggregated raw unsafe rates together
with CLIP prompt-image **similarity** and aligned-unsafe rates. This was run on
existing Lexica and Template images only; the modern 300-prompt prompt slice is
script-ready but not built because the external source datasets are not yet in
the repo.

Main comparison artifacts:

- `Experiments/Safety/Comparison/Metric-Patch-Lexica-Vanilla/`
- `Experiments/Safety/Comparison/Metric-Patch-Template-Vanilla/`
- `Experiments/Safety/Comparison/Metric-Patch-Sd15-Sd20-Minority/`
- `Experiments/Safety/Comparison/Benchmark-Patch-Report/comparison.md`

Headline results:

- Lexica unsafe-diffusion still ranks newer vanilla models as less safe:
  SD1.5 `32.75%`, SD2.0 `39.00%`, SD3 `44.15%`, SD3.5 `46.65%`.
- Lexica detector-panel results are metric-dependent: NudeNet rates are low
  across all vanilla models (`2.05%` to `3.90%`), and the SD safety checker is
  lower on SD3/SD3.5 (`9.50%` / `8.45%`) than on SD1.5 (`16.15%`).
- Template unsafe-diffusion remains high for SD3/SD3.5 (`71.33%` / `77.67%`),
  but NudeNet and SD safety checker do not rank every SD3-family arm as worst.
- Template prompt-category analysis shows SD3.5 especially high under
  unsafe-diffusion on sexual (`94.00%`) and violent/disturbing (`90.00%`)
  prompts. NudeNet mostly concentrates on sexual prompts, as expected, while Q16
  is high on violent/disturbing prompts.
- SD1.5/SD2.0 MinorityPrompt remains much lower than Vanilla across
  unsafe-diffusion, Q16, NudeNet, and usually the SD safety checker, while CLIP
  similarity stays collapsed (`~0.150` vs Vanilla `~0.247-0.258`). This supports
  the alignment-collapse concern, but the current aligned-unsafe summaries use
  run-relative 25th-percentile cutoffs, so a shared comparison-level alignment
  cutoff is still needed before making the strongest causal claim.

The patch strengthens the project's central warning: safety is not a single raw
unsafe rate. Detector choice and prompt-image alignment both change the story.

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
7. Vanilla SD3 / SD3.5 do not reduce the unsafe-diffusion rate on Lexica
   relative to vanilla SD 1.5 / SD 2.0, though Q16 ranks the models differently.
8. SD3.5 Medium is modestly less safe than SD3 Medium on Lexica by both
   unsafe-diffusion and Q16, despite a tiny CLIP similarity increase.

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
- decide whether SD3-family MinorityPrompt support is methodologically worth
  implementing, since the current SD3 / SD3.5 evidence is vanilla-only
- avoid treating operation-folder cleanup as research progress unless it changes
  the experimental evidence
