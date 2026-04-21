# AGENTS.md - MoreDM Codebase

This project explores **how Minority Generation affects Unsafe Generation** in text-to-image models.

When I ask you to write a script, you should only generate them but not run them.
I will run the script myself after reviewing them.

## Build & Run
- Package manager: `uv` (use `uv run` to execute scripts)
- Install: `uv sync`
- Generation: `uv run python bin/gen.py`
- Safety eval: `uv run python metrics/unsafe-diffusion/inference.py`
- No test suite configured

## Directory Structure
- `bin/`: Core scripts (`gen.py` for image generation, `make_families.py` for attribution, `make_templates.py` for template injection)
- `minority/`: Minority generation techniques (`MinorityPrompt/`)
- `t2ls/`: Text-to-image models (sd3m, sd3.5m, sd3.5t)
- `metrics/`: Evaluation code
  - `unsafe-diffusion/`: Safety classifier using CLIP
  - `CLIP/`: CLIP-based evaluation
- `memory-ds/`: Dataset processing code
- `Operations/`: Shell scripts for running experiments (see below)
- `analyse/`: Analysis outputs and visualizations
- `Datasets/`: External datasets (4chan.txt, COCO, Lexica prompts)
- `Experiments/`: Experiment results storage (see below)
- `Models/`: Pre-downloaded model cache

## Operations Scripts
Shell scripts in `Operations/` orchestrate the experimental workflow.

**New scripts**: Place directly in `Operations/` with sequential names (`001.sh`, `002.sh`, ...) for sequential execution. Move to subdirectories after workflow is established.

### generation/
Scripts that run `bin/gen.py` to create images.
- `baseline-sdxllight-4chan.sh`: SdxlLight-4Chan baseline (50 prompts)
- `baseline-sdxllight-lexica.sh`: SdxlLight-Lexica baseline (50 prompts)
- `minority-sdxllight-4chan.sh`: SdxlLight-4Chan minority/default (50 prompts)

### evaluation/
Scripts that run `metrics/unsafe-diffusion/inference.py` for safety classification.
- `baseline-sdxllight-4chan.sh`: SdxlLight-4Chan baseline
- `baseline-sdxllight-coco.sh`: SdxlLight-COCO baseline
- `baseline-sdxllight-lexica.sh`: SdxlLight-Lexica baseline
- `minority-sdxllight-4chan.sh`: SdxlLight-4Chan minority/default
- `minority-sdxllight-coco.sh`: SdxlLight-COCO minority/default
- `minority-sdxllight-lexica.sh`: SdxlLight-Lexica minority/default

### comparison/
Scripts that compare safety stats (markdown reports).
For comparison tasks, check script input/output paths first, then logs in `Experiments/Safety/`, then final markdown reports.

Cross-dataset comparisons:
- `4chan-vs-coco.sh`: 4Chan vs COCO
- `4chan-vs-lexica.sh`: 4Chan vs Lexica
- `coco-vs-lexica.sh`: COCO vs Lexica

Minority vs Baseline comparisons:
- `minority-vs-baseline-4chan.sh`: Minority/default vs Baseline on 4Chan
- `minority-vs-baseline-coco.sh`: Minority/default vs Baseline on COCO
- `minority-vs-baseline-lexica.sh`: Minority/default vs Baseline on Lexica

Prompt-wise Minority vs Baseline comparisons:
- `001.sh`: Prompt-wise safer/unsafer/almost-same on 4Chan
- `002.sh`: Prompt-wise safer/unsafer/almost-same on COCO
- `003.sh`: Prompt-wise safer/unsafer/almost-same on Lexica
- Rule: `delta = minority_unsafe - baseline_unsafe`; default `almost_same` tolerance is `|delta| <= 1`
- Prompt IDs may be 2-digit or 3-digit strings across datasets; compare numerically by prompt id

### 04-clip-lexica-minority-vs-baseline/
CLIP distance analysis on Lexica (run sequentially: 01 -> 02 -> 03 -> 04 -> 05):
- `01-clip-distance-baseline.sh`: CLIP distance eval for baseline SdxlLight-Lexica
- `02-clip-distance-minority.sh`: CLIP distance eval for minority SdxlLight-Lexica
- `03-compare-imagewise.sh`: CLIP image-wise comparison: Minority vs Baseline on Lexica
- `04-compare-promptwise.sh`: CLIP prompt-wise comparison: Minority vs Baseline on Lexica
- `05-clip-vs-classifier-relevance.sh`: CLIP-vs-classifier relevance: Minority vs Baseline on Lexica

### 05-unsafe-attribution-all-1-50/
Attribution round 1 — all 3 datasets × 50 prompts (superseded by 08-attrib-lexica-1-200):
- `01-select-special-prompts.sh` -> `06-compare-attribution.sh`: Collect specials, build families, generate, evaluate, compare
  - Found 5 special prompts (all from Lexica). Person names = primary driver.

### 06-lexica-baseline-51-200-vs-low-density-101-200/
Lexica baseline vs low-density experiment for expanded ranges and safety comparison prep.
- `01-generate-baseline.sh`: Baseline generation Lexica 51-200 (4-GPU parallel)
- `02-generate-low-density.sh`: Low-density generation Lexica 101-200 (4-GPU parallel; 1-100 already existed)
- `03-evaluate-safety.sh`: Safety evaluation for both ranges (4-GPU parallel)
- `04-prepare-comparison-logs.sh`: Rebuild safety logs to cover 1-200 for baseline and low-density

### 07-attrib-lexica/
Attribution round 2 — Lexica only, prompts 1-50 + 101-200 (superseded by 08-attrib-lexica-1-200):
- `01-collect-special-prompts.sh`: Collect special prompts from Lexica (150 usable)
- `02-generate-families.sh`: Generate prompt families from specials
- `03-generate-baseline.sh`: Baseline image generation (4-GPU parallel)
- `04-generate-minority.sh`: Minority image generation (4-GPU parallel)
- `05-evaluate-safety.sh`: Safety evaluation (4-GPU parallel)
- `06-compare-attribution.sh`: Per-family + summary attribution reports

### 08-attrib-lexica-1-200/
Attribution round 3 — Lexica only, full 200 prompts (current):
- `01-regenerate-minority-51-100.sh`: Regenerate stale minority images for Lexica 51-100
- `02-evaluate-safety-minority-51-100.sh`: Safety eval for regenerated minority 51-100
- `03-rebuild-safety-logs.sh`: Rebuild safety logs for full 1-200 range
- `04-collect-special-prompts.sh`: Collect special prompts (all 200 Lexica)
  - Env: `BASELINE_MAX_UNSAFE` (default 3), `MIN_DELTA` (default 4)
  - Output: `Experiments/Attribution/special.tsv`, `special.txt`
- `05-generate-families.sh`: Generate prompt families via `bin/make_families.py`
  - Output: `Experiments/Attribution/Families/sp-NNN/{family.txt, manifest.tsv}`
- `06-generate-baseline.sh`: Baseline image generation (4-GPU parallel)
- `07-generate-minority.sh`: Minority image generation (4-GPU parallel)
- `08-evaluate-safety.sh`: Safety evaluation (4-GPU parallel)
- `09-compare-attribution.sh`: Attribution comparison reports (per-family + summary)
  - Env: `SPECIAL_THRESHOLD` (default 4)
  - Output: `Experiments/Attribution/Comparison/{sp-NNN/comparison.md, summary.md}`

### 09-template-injection/
Template injection experiment — inject key elements into neutral templates (50 pairs × 3 element types):
- `01-build-templates.sh`: Generate prompt pairs via `bin/make_templates.py` (Person, Artist, Mood)
- `02-generate-baseline.sh`: Baseline image generation (4-GPU parallel)
- `03-generate-minority.sh`: Minority image generation (4-GPU parallel)
- `04-evaluate-safety.sh`: Safety evaluation (4-GPU parallel)
- `05-compare-injection.sh`: Aggregate comparison reports with interaction effect
- `06-show-examples.sh`: Filter prompt pairs with high interaction (≥4)
  - Finding: Person names show +14.2% interaction; artists/moods show no minority-specific amplification.

## Model Cache
Pre-downloaded models for offline use:

- `Models/clip/hub/`: CLIP model cache (ViT-L-14 with OpenAI weights)
  - Used by `metrics/unsafe-diffusion/` for safety classification
  - `MHSafetyClassifier` in `train.py` accepts `cache_dir` parameter
  - `inference.py` uses `CLIP_CACHE_DIR` constant pointing to this location
  - To download/update: `HF_HOME=Models/clip python -c "import open_clip; open_clip.create_model_and_transforms('ViT-L-14', 'openai')"`

- `metrics/unsafe-diffusion/checkpoints/multi-headed/`: Pre-trained safety classifier heads
  - Contains `sexual.pt`, `violent.pt`, `disturbing.pt`, `hateful.pt`, `political.pt`

## Experiments Layout

### Text2Image/
Image generation results:
- `Baseline/<Model>-<Dataset>/`: Baseline generations (no minority techniques)
- `Minority/<Model>-<Dataset>/<config>/`: Minority generations with configs:
  - `default`: 1 opt token, init_word=handsome
  - `num_opt_tokens/<N>`: Variable token count
  - `init_word/<word>`: Different init words
- Each prompt folder contains ~10 generated images

### Safety/
Safety evaluation results:

- `Dataset/`: Single-dataset and cross-dataset evaluations
  - `1-SdxlLight-<Dataset>/`: Single dataset evals (4Chan, COCO, Lexica)
  - `2-SdxlLight-<A>-<B>/`: Cross-dataset comparisons
- `Minority/`: Minority technique safety evals
  - `SdxlLight-<Dataset>/`: Per-dataset minority evals
- `Comparison/`: Minority vs Baseline comparisons
  - `Minority-vs-Baseline-<Dataset>/`: Side-by-side comparison reports
  - `PromptWise-Minority-vs-Baseline-<Dataset>/`: Prompt-wise safer/unsafer/almost-same reports
  - Prompt counts may differ between baseline and minority runs for a dataset (for example Lexica)

### Attribution/
Prompt element attribution analysis (round 3 — Lexica, 200 prompts, 18 specials):
- `special.tsv`: Manifest of "special" prompts (sp_id, dataset, src_line, baseline/minority unsafe, delta, prompt)
- `special.txt`: Special prompt texts (one per line, order matches special.tsv)
- `Families/sp-NNN/`: Per-prompt family directories
  - `family.txt`: Line 1 = original prompt, lines 2+ = single-element variants
  - `manifest.tsv`: What was changed per variant (var_line, element_type, original, replacement)
- `Text2Image/{Baseline,Minority}/sp-NNN/NNN/`: Generated images per variant (10 each)
- `Safety/{Baseline,Minority}/sp-NNN.log`: Safety logs per family (v-id, safe, unsafe)
- `Comparison/sp-NNN/comparison.md`: Per-family attribution table
- `Comparison/summary.md`: Cross-family element-type attribution summary

## GPU Parallelism
This machine has **4x NVIDIA H100**. GPU-parallel execution is the default:
- Generation and evaluation scripts MUST split work across 4 GPUs using `CUDA_VISIBLE_DEVICES=N ... &` and `wait`.
- Split ranges as evenly as possible across 4 shards.

## Commit Practices
- Each separate experiment gets its own commit.
- Commit includes: the Operations script(s) AND all outputs (safety json, logs, reports), EXCEPT generated images (which are gitignored).
- Commit message: short summary of what the experiment does.

## Code Style
- Python 3.10+, type hints encouraged
- Use existing patterns from neighboring files
- Shell scripts in `Operations/` are organized into numbered subdirectories (`04-clip-lexica-minority-vs-baseline/`, `05-unsafe-attribution-all-1-50/`, etc.) plus `generation/`, `evaluation/`, `comparison/` for early experiments

## Experiment History
Chronological record of experiments conducted (matching git history):

1. **Initial setup** (`477eb2b`): AGENTS.md, Operations scripts, .gitignore
2. **Environment** (`e36f848`): pyproject.toml, uv.lock, .python-version
3. **Baseline + Minority generation & safety eval** (`0249893`..`5b363ee`): Ran gen.py for SdxlLight on 4Chan/COCO/Lexica (50 prompts each), baseline + minority/default; safety classification with unsafe-diffusion
4. **Cross-dataset & Minority-vs-Baseline comparisons** (`5b363ee`): Safety stat comparisons, prompt-wise reports (Operations/comparison/)
5. **CLIP distance analysis** (`e174647`..`114eacf`): CLIP distance eval on Lexica, image-wise and prompt-wise comparison, CLIP-vs-classifier relevance (Operations/clip/004-008)
6. **Attribution pipeline** (`c7c39a9`): Collected 5 special prompts (delta>=4, baseline_unsafe<=3, all from Lexica), generated prompt families, ran attribution analysis. Finding: named persons are primary driver (100% key ratio)
7. **Expand Lexica to 200 prompts** (`b843bda`): Baseline gen 51-200, minority gen 101-200, safety eval, log rebuild — to increase the pool for finding more special prompts
8. **Attribution round 2** (`9ba77db`): Lexica-only, 150 prompts (1-50 + 101-200, excluding 51-100). 14 special prompts found, 4/14 reproduced. Person and medium remain top key elements.
9. **Template injection** (`2cecc8a`): Injected key elements (person, artist, mood) into neutral templates (50 pairs each). Person names show +14.2% interaction effect; artists/moods show no minority-specific amplification.

## File Organization Style
One standard for file organization is the number of folders/files under one folder shouldn't exceed 10.
Only reorg when I ask you to do so.
