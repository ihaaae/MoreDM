# AGENTS.md - MoreDM Codebase

This project studies whether **low-density prompt optimization (MinorityPrompt)** makes text-to-image generation less safe, and which prompt elements drive that effect. See `RESEARCH.md` for full findings and hypotheses.

When I ask you to write a script, you should only generate them but not run them.
I will run the script myself after reviewing them.

## Build & Run
- Package manager: `uv` (use `uv run` to execute scripts)
- No test suite configured

## GPU Parallelism
This machine has **4x NVIDIA H100**. GPU-parallel execution is the default:
- Operations shell scripts for generation/evaluation/scoring MUST split work across 4 GPUs using `CUDA_VISIBLE_DEVICES=N uv run ... &` and `wait`.
- Python CLIs should expose ranges/options and let Operations shell wrappers shard work.
- Split ranges as evenly as possible across 4 shards.
- If a GPU is known-bad for the current machine/session, document that fact in
  the script comment and shard across the remaining healthy GPUs instead of
  forcing a failing 4-GPU run. Otherwise prefer all four GPUs.

## Operations Scripts
Shell scripts in `Operations/` orchestrate experiments. Numbered subdirectories run sequentially (01 → 02 → ...).

Agents working on files under `Operations/` MUST read `Operations/AGENTS.md` first and follow its local workflow conventions.

**New scripts**: Place directly in `Operations/` with sequential names (`001.sh`, `002.sh`, ...). Move to subdirectories after workflow is established.

**Established scripts**: Once scripts live in a numbered subdirectory, prefer descriptive names over generic numbers. Use lowercase hyphenated names with the pattern `<verb>-<analysis-kind>-<scope-or-dataset>-<comparison>.sh` when practical. Comparison scripts should make the granularity explicit:
- `compare-dataset-*`: aggregate whole-dataset safety summaries.
- `compare-strategy-*`: aggregate strategy summaries, such as Minority vs Vanilla.
- `compare-promptwise-*`: prompt-ID-level comparisons.

Key conventions visible in existing scripts — read them before writing new ones.

## External Repos (Git Submodules)
- Prefer git submodules to introduce external research repos.
- Place submodules under `modules/` unless a different top-level role is
  explicitly needed.
- Use git commands (e.g., `git submodule status`, `cat .gitmodules`) to discover existing submodules and their details.

## Path Notes
- `Experiments/Attribution/` holds the latest attribution data (round 3, Lexica 200 prompts). Earlier rounds (1 and 2) have been removed.
- `Experiments/Injection/` holds template injection results (from `Operations/09-template-injection/`).

## Commit Practices
- Each separate experiment gets its own commit.
- Commit includes: the Operations script(s) AND all outputs (safety json, logs, reports), EXCEPT generated images (which are gitignored).
- Commit message format:

  ```
  [scope](type): short summary of what the change does

  commit body explaining the what/why in more detail

  optional footer
  ```

  - First line: `[scope](type): subject`. The `(type)` is optional — `[scope]: subject` is also fine.
  - `scope`: the area touched, e.g. `experiment`, `Operations`, `lib`, `env`, `docs`, `refactor`.
  - `type`: conventional-commits style, e.g. `feat`, `chore`.
  - Body: explain what the change does and why; wrap as needed and use blank lines to separate paragraphs.
  - Footer: optional (e.g. `Co-Authored-By:` trailers).

## Progress Notes
- `PROGRESS.md` should reflect research progress, not strictly mirror chronological engineering work.
- Organize `PROGRESS.md` around experiments and research findings, not workflow conventions or script naming rules.
- Early in an investigation, it is acceptable to log small operational details because the larger structure may not yet be clear.
- As the research picture improves, condense earlier detailed logs into higher-level findings, hypotheses, and decisions, even when that violates strict time order.
- Git history is an engineering record, but progress notes may reorganize events by research logic. For example, a section about `Operations/01-generation`, `02-evaluation`, and `03-comparison` does not need to discuss the refactor that created those folders unless the refactor matters scientifically.

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
11. **SD 1.x/2.x MinorityPrompt generation**: `lib/gen.py` supports SD 1.5
    and SD 2.0 baseline/minority models. SD 2.0 uses the public
    `sd2-community/stable-diffusion-2-base` mirror, and SD 1.x/2.x
    MinorityPrompt uses fixed-ratio prompt optimization to avoid scheduler
    indexing failures.
12. **SD 1.x/2.x paired safety comparison**: Vanilla and MinorityPrompt arms
    have unsafe-diffusion and Q16 artifacts for SD 1.5 / SD 2.0 Lexica
    (`200 x 10` images per arm). The measured direction reduces unsafe rates
    under MinorityPrompt for SD 1.x/2.x, unlike the earlier SDXL-Lightning
    unsafe-diffusion result.
