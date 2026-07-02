# AGENTS.md - UnsafeDistribution

## Versioning
- Use semantic version tags for research milestones.
- Treat annotated tags as the durable progress record: summarize what changed,
  why the version matters, and where external artifacts are stored.
- Re-establish package versions explicitly before relying on historical
  package metadata from legacy branches.

## Workflow
- Use Linear for handoff, issue tracking, and current project state.
- Keep `AGENTS.md` focused on durable repository guidance.

## Build & Run
- Package manager: `uv`.
- Use `uv run` for Python entry points once the environment files are restored.
- No test suite is currently configured.
- When asked to write a script, generate it for review and wait for explicit
  permission before running it.

## GPU Parallelism
This machine has 4x NVIDIA H100. GPU-parallel execution is the default:
- Operations shell scripts for generation, evaluation, and scoring should split
  work across 4 GPUs with `CUDA_VISIBLE_DEVICES=N uv run ... &` and `wait`.
- Python CLIs should expose range and sharding options; shell wrappers should
  orchestrate GPU shards.
- Split ranges as evenly as possible across 4 shards.
- If a GPU is known bad for the current session, document that fact in the
  script comment and shard across the remaining healthy GPUs.

## Operations Scripts
- Shell scripts in `Operations/` orchestrate experiments.
- Agents working under `Operations/` must read `Operations/AGENTS.md` first
  after that file is restored.
- New scripts should start directly under `Operations/` with sequential names
  such as `001.sh`, `002.sh`, and move into numbered subdirectories only after
  the workflow is established.
- Established scripts should prefer descriptive lowercase hyphenated names.

## External Repos
- Prefer Git submodules for external research repos.
- Place reusable external code under `modules/` unless a different top-level
  role is intentionally documented.
- Use Git commands such as `git submodule status` and `cat .gitmodules` to
  discover submodules and their pinned revisions.

## Commit Practices
- Commit only after the user has reviewed the exact staged changes and the
  commit message draft.
- Split conceptual work into separate commits. Combine repo policy, runtime
  dependencies, library code, workflow scripts, data fixtures, and docs only
  when the user explicitly asks for a squash.
- Commit message format:

  ```text
  [scope](type): short summary

  Body explaining what changed and why.
  ```

- Common scopes: `[repo]`, `[env]`, `[modules]`, `[lib]`, `[Operations]`,
  `[data]`, `[docs]`, `[experiment]`.
- Use `(chore)` for policy, configuration, dependency, and maintenance changes;
  `(feat)` for new runnable behavior; `(fix)` for behavior corrections; and
  `(docs)` for documentation-only changes.

## Repository Layout
- `bin/` contains runnable command-line entry points and small scripts.
- `lib/` contains reusable library modules used by scripts and workflows.
- `modules/` contains Git submodule dependencies.
- `Operations/` contains experiment and workflow scripts intended to be run.
- `Experiments/` contains experiment results. Use `.gitignore` and the
  reviewed task scope to determine which result files are tracked.
- `Datasets/`, `MyPaper/`, and `Papers/` are unresolved top-level areas.
  Establish their layout and tracking rules in the relevant task before adding
  or reorganizing them.
