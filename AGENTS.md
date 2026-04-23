# AGENTS.md - MoreDM Codebase

This project studies whether **low-density prompt optimization (MinorityPrompt)** makes text-to-image generation less safe, and which prompt elements drive that effect. See `RESEARCH.md` for full findings and hypotheses.

When I ask you to write a script, you should only generate them but not run them.
I will run the script myself after reviewing them.

## Build & Run
- Package manager: `uv` (use `uv run` to execute scripts)
- No test suite configured

## GPU Parallelism
This machine has **4x NVIDIA H100**. GPU-parallel execution is the default:
- Generation and evaluation scripts MUST split work across 4 GPUs using `CUDA_VISIBLE_DEVICES=N ... &` and `wait`.
- Split ranges as evenly as possible across 4 shards.

## Operations Scripts
Shell scripts in `Operations/` orchestrate experiments. Numbered subdirectories run sequentially (01 → 02 → ...).

**New scripts**: Place directly in `Operations/` with sequential names (`001.sh`, `002.sh`, ...). Move to subdirectories after workflow is established.

Key conventions visible in existing scripts — read them before writing new ones.

## Path Notes
- `Experiments/Attribution/` holds the latest attribution data (round 3, Lexica 200 prompts). Earlier rounds (1 and 2) have been removed.
- `Experiments/Injection/` holds template injection results (from `Operations/09-template-injection/`).

## Commit Practices
- Each separate experiment gets its own commit.
- Commit includes: the Operations script(s) AND all outputs (safety json, logs, reports), EXCEPT generated images (which are gitignored).
- Commit message: short summary of what the experiment does.

## File Organization Style
The number of folders/files under one folder shouldn't exceed 10.
Only reorg when I ask you to do so.

## Experiment History
Chronological record (see git log for commit hashes):

1. **Initial setup**: AGENTS.md, Operations scripts, .gitignore
2. **Environment**: pyproject.toml, uv.lock, .python-version
3. **Baseline + Minority generation & safety eval**: SdxlLight on 4Chan/COCO/Lexica (50 prompts each)
4. **Cross-dataset & Minority-vs-Baseline comparisons**: Safety stat comparisons, prompt-wise reports
5. **CLIP distance analysis**: Lexica image-wise/prompt-wise comparison, CLIP-vs-classifier relevance
6. **Attribution round 1**: 5 special prompts (all Lexica). Person names = primary driver (100% key ratio)
7. **Expand Lexica to 200 prompts**: Baseline gen 51-200, minority gen 101-200
8. **Attribution round 2**: 14 special prompts, 4/14 reproduced. Person and medium top key elements.
9. **Template injection**: Person names +14.2% interaction; artists/moods no minority-specific amplification.
10. **Q16 safety scoring**: Weak positive correlation (r=0.211) with multi-headed classifier; safety impact metric-dependent.
