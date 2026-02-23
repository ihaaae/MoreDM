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
- `bin/`: Core scripts (`gen.py` for image generation)
- `minority/`: Minority generation techniques (`MinorityPrompt/`)
- `t2ls/`: Text-to-image models (sd3m, sd3.5m, sd3.5t)
- `metrics/`: Evaluation code
  - `unsafe-diffusion/`: Safety classifier using CLIP
  - `CLIP/`: CLIP-based evaluation
- `memory-ds/`: Dataset processing code
- `Operations/`: Shell scripts for running experiments (see below)
- `analyse/`: Analysis outputs and visualizations
- `Datasets/`: External prompt datasets (git-ignored; stored locally). In particular `Datasets/unsafe-diffusion/` contains 4 prompt TXT files: `4chan.txt`, `COCO.txt`, `Lexica.txt`, `Template.txt`
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

## Code Style
- Python 3.10+, type hints encouraged
- Use existing patterns from neighboring files
- Shell scripts in `Operations/` are organized into `generation/`, `evaluation/`, `comparison/` subdirectories

## File Organization Style
One standard for file organization is the number of folders/files under one folder shouldn't exceed 10.
Only reorg when I ask you to do so.
