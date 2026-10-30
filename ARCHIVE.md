# Evidence archive and model identity

## Model names are not interchangeable

| Research name | CLI keys | Historical path names |
| --- | --- | --- |
| SDXL-base | `sdxl`, `min-sdxl` | explicit base-model snapshot required |
| SDXL-Lightning | `sdxl-light`, `min-sdxl-light` | `SdxlLight`, `sdxl-lightning` |
| SD3 Medium | `sd3`, `min-sd3` | `Sd3` |
| SD3.5 Medium | `sd35`, `min-sd35` | `Sd35` |

Bare "SDXL", "old SDXL" and "original SDXL" are insufficient model identities.
SDXL-Lightning findings remain active, including RESEARCH.md and the standalone
`Metric-Patch-Regenerated-SdxlLight-*` reports. Do not transfer their measured
effects or safe prompt sets to SDXL-base or the SD3 family.

## Archive policy

Archive date: 2026-10-30. Source: Amp thread
https://ampcode.com/threads/T-01a17849-eece-7341-8bd9-979fe38ebd3a .

This is non-destructive quarantine, not deletion or rewriting of observations.
Moved files retain their original relative paths below
`Archive/2026-10-invalid-evidence/`; SHA256SUMS verifies the original bytes.
`.ignore` hides that tree from default searches, not from Git history.

The external H100 audit root also has a `.ignore` excluding all `sources/`
trees and `.py`/`.sh` snapshots from ignore-aware searches. This includes valid
frozen code: a snapshot is provenance, not the active implementation. Result
JSON and reports remain discoverable. Its `AGENTS.md` states the same boundary.
These are discovery controls, not access controls: explicit file reads and
tools that disregard ignore files can still access the preserved bytes.

Large result trees and frozen source snapshots remain **archived in place** to
preserve absolute-path manifests, hashes and symlinks. Their archive status
applies to descendants, copied raw outputs and downstream conclusions, even
when a summary does not carry its own banner. They may be used to diagnose
the historical failure, not as evidence for corrected TailProbe efficacy.
Mixed comparisons must label legacy arms; valid native controls are not erased.

## Invalid legacy SD3-family implementation

The historical single-CLIP-L velocity-deviation objective at MinorityPrompt
`0a48703a6024dd3fa9da44b7e2726c577e076993` is not the intended Eq6 method.
Frozen `latent_sd3.py` SHA-256:
`b0933b5fbf8d38531ed2dbb8868a6fc47bc67ce89c7e06ccdca92383fb29c9b3`.
Its copy is archived as `lib/legacy_sd3.py` under the archive root.

Moved out of active entry points:
- `Recipes/20-minority-sd3-sd35-lexica/` (generation and scoring recipes).
- `Recipes/028.sh` and the original `lib/sd3_ablation.py` diagnostic driver.
- `Recipes/039.sh` (legacy initializer experiment).
- `Experiments/Safety/Comparison/Metric-Patch-Minority-Sd3-Lexica/summary.md`
  and the corresponding `Sd35` summary.

`lib/sd3_ablation.py` now retains only the unchanged image-noise metric for
existing readers; its CLI cannot generate. Current generic initializer helpers
remain because the Eq6 experiments also use them; they are not proof that the
old objective is valid. Their legacy selection entry point is disabled and
their loader rejects the known legacy revision or solver hash before loading
GPU dependencies.

Archived in place under `Experiments/Safety/Minority/`:
`Sd3-Lexica/default`, `Sd35-Lexica/default` and their derived score/CLIP outputs.
The SD3-family **Minority** rows and interpretations in mixed historic reports
are retired. Native SD3/SD3.5 rows are not invalidated by the optimizer defect.

Archived in place under the external audit root
`/mnt/raid10/luke/UnsafeDistribution/TailProbe-Audit/`:
- `sd3-ablation-001`, `sd3-ablation-smoke-001`, `sd3-solver-smoke-001`.
- `flow-init-selection-001`, `flow-init-h100-001`, `flow-init-a800-001`,
  `flow-init-smoke-h100-001`, `flow-init-smoke-a800-001`,
  `flow-init-comparison-001`.
- `selected-ad-reuse-001` (reselected legacy optimized images remain legacy).
- `flow-legacy-replay-001` (historical failure reproduction only).
- Legacy D arms in `flow-eq6-comparison-001` and
  `flow-eq6-jev-comparison-001`; do not discard their separate corrected arms.

Current submodule `e6ed45c50778357e64b2c8e975e75c3419e457e7` has solver SHA
`5ff3d088ffd4c671e7a64ea2cd75b116267340bcd34637416274635fed91a787`.
It is **not** the invalid legacy implementation. Eq6 changes also include
encoder/numerical changes; old-versus-new results are not objective-only ablations.

## Cohort mismatch: restricted evidence, not a false detector measurement

Safe membership is specific to target checkpoint, native sampler, classifier
and selection rule. A prompt selected as safe on SDXL-Lightning is not thereby
safe on SDXL-base, SD3 or SD3.5. All inherited-safe claims without target-specific
screening are withdrawn, including downstream conclusions built on those claims.

Moved: `Recipes/038.sh` and `lib/flow_start_audit.py`, which explicitly reuse
eight Lightning diagnostic prompts across the SD3 family. External
`flow-start-001` and `flow-start-smoke-{h100,a800}-001` are archived in place as
cross-model exploratory diagnostics, **not target-safe-cohort evaluations**.
Their observed schedule sensitivity is not itself fabricated or invalid.

Do not over-quarantine:
- `Recipes/033.sh` uses outcome-independent stride-selected prompts;
  `034.sh`/`035.sh` use the same cohort with the corrected solver (verified SHA
  above). These are exploratory diagnostics, not a transferred safe cohort.
- `prompt-screen-001` separately screens SDXL-base, SD3 and SD3.5 with each
  classifier. Preserve it and its native source images.
- `prompt-screen-delta3-001` selects a relative drop, not target safety. Preserve
  it only as relative-drop analysis; `Recipes/032.sh` is not a safe-cohort result.
- Lightning-specific `selection-001`, schedule/factor/first-step experiments
  and `Recipes/022.sh`–`026.sh`, `036.sh`–`037.sh` remain Lightning evidence.

## Stale narrative

`PROGRESS.md`, `HANDOFF.md` and `log.md` moved into the archive. Their instructions
and conclusions are historical, not current project state. Linear is the task
tracker; RESEARCH.md is the active research entry point. Mixed-model reports
must retain a warning adjacent to any invalid legacy SD3-family interpretation.

## Boundaries of this cleanup

This checkout and the H100 audit-root archive index were reviewed. Paper source
sections `05_experiments.tex`, `06_conclusion.tex` and `07_appendix.tex` now
identify Lightning explicitly and retire legacy SD3-family efficacy claims.
Existing compiled paper/HTML previews were not rebuilt and remain stale;
they must not be used as current evidence. The main/appendix Lightning
SD-checker numbers still need provenance reconciliation before publication.

No A800 checkout, Git history or release artifact was rewritten. Do not assume
an unreviewed remote copy is clean; apply these provenance rules before reuse.
An exhaustive inventory of remote and secondary copies is a follow-up audit,
not a completed claim of this cleanup.
