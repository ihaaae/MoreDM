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
- `bin/`: Core scripts (`gen.py` for image generation, `make_families.py` for attribution)
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

### clip/
CLIP distance analysis on Lexica (run sequentially: 004 -> 005 -> 006 -> 007 -> 008):
- `004.sh`: CLIP distance eval for baseline SdxlLight-Lexica
- `005.sh`: CLIP distance eval for minority SdxlLight-Lexica
- `006.sh`: CLIP image-wise comparison: Minority vs Baseline on Lexica
- `007.sh`: CLIP prompt-wise comparison: Minority vs Baseline on Lexica
- `008.sh`: CLIP-vs-classifier relevance: Minority vs Baseline on Lexica

### attribution/
Prompt element attribution pipeline (run sequentially: 009 -> 010 -> 011 -> 012 -> 013 -> 014):
- `009.sh`: Collect "special" prompts (safe baseline, unsafe minority) across all datasets
  - Env: `BASELINE_MAX_UNSAFE` (default 3), `MIN_DELTA` (default 4)
  - Output: `Experiments/Attribution/special.tsv`, `special.txt`
- `010.sh`: Generate prompt families via `bin/make_families.py`
  - Output: `Experiments/Attribution/Families/sp-NNN/{family.txt, manifest.tsv}`
- `011.sh`: Baseline image generation for all family variants
- `012.sh`: Minority image generation for all family variants
- `013.sh`: Safety evaluation for all family images
- `014.sh`: Attribution comparison reports (per-family + summary)
  - Env: `SPECIAL_THRESHOLD` (default 4)
  - Output: `Experiments/Attribution/Comparison/{sp-NNN/comparison.md, summary.md}`

### attribution2/
Attribution round 2 pipeline — Lexica only, 150 prompts (1-50 + 101-200, excluding 51-100):
- `019.sh`: Collect special prompts from Lexica only (excludes 51-100)
  - Env: `BASELINE_MAX_UNSAFE` (default 3), `MIN_DELTA` (default 4)
  - Output: `Experiments/Attribution2/special.tsv`, `special.txt`
- `020.sh`: Generate prompt families via `bin/make_families.py`
  - Output: `Experiments/Attribution2/Families/sp-NNN/{family.txt, manifest.tsv}`
- `021.sh`: Baseline image generation for all family variants (4-GPU parallel)
- `022.sh`: Minority image generation for all family variants (4-GPU parallel)
- `023.sh`: Safety evaluation for all family images (4-GPU parallel)
- `024.sh`: Attribution comparison reports (per-family + summary)
  - Env: `SPECIAL_THRESHOLD` (default 4)
  - Output: `Experiments/Attribution2/Comparison/{sp-NNN/comparison.md, summary.md}`

### Expand Lexica to 200 prompts (015 -> 016 -> 017 -> 018)
Expand Lexica baseline+minority from 50 to 200 prompts to find more special prompts for attribution.
- `015.sh`: Baseline generation Lexica 51-200 (4-GPU parallel)
- `016.sh`: Minority generation Lexica 101-200 (4-GPU parallel; 1-100 already existed)
- `017.sh`: Safety evaluation for both expanded ranges (4-GPU parallel)
- `018.sh`: Rebuild safety logs to cover 1-200 for both baseline and minority

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
Prompt element attribution analysis:
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
- Shell scripts in `Operations/` are organized into `generation/`, `evaluation/`, `comparison/` subdirectories

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

## File Organization Style
One standard for file organization is the number of folders/files under one folder shouldn't exceed 10.
Only reorg when I ask you to do so.
