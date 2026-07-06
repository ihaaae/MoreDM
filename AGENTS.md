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
- Recipe shell scripts for generation, evaluation, and scoring should split
  work across 4 GPUs with `CUDA_VISIBLE_DEVICES=N uv run ... &` and `wait`.
- Python CLIs should expose range and sharding options; shell wrappers should
  orchestrate GPU shards.
- Split ranges as evenly as possible across 4 shards.
- If a GPU is known bad for the current session, document that fact in the
  script comment and shard across the remaining healthy GPUs.

## Recipes And Experiments
- `Recipes/` contains experiment blueprints: shell scripts and orchestration
  recipes that describe how to build or reproduce experiment artifacts.
- `Experiments/` contains built experiment artifacts and result views. Use
  `.gitignore` and the reviewed task scope to determine which experiment
  artifacts are tracked.
- Agents working under `Recipes/` must read `Recipes/AGENTS.md` first
  after that file is restored.
- New recipe scripts should start directly under `Recipes/` with sequential names
  such as `001.sh`, `002.sh`, and move into numbered subdirectories only after
  the recipe family is established.
- Established recipe scripts should prefer descriptive lowercase hyphenated names.
- If a legacy `Operations/` directory appears while refactoring old history,
  treat it as pending-renamed `Recipes/` content unless the reviewed task says
  otherwise.

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

- Common scopes: `[repo]`, `[env]`, `[modules]`, `[lib]`, `[bin]`,
  `[Recipes]`, `[Experiments]`, and `[docs]`.
- Use `(chore)` for policy, configuration, dependency, and maintenance changes;
  `(feat)` for new runnable behavior; `(fix)` for behavior corrections; and
  `(docs)` for documentation-only changes.
- Prefer these scopes for new work:
  - `[repo]` for repository policy, ignore rules, and agent guidance.
  - `[env]` for Python/runtime/dependency metadata.
  - `[modules]` for submodules and external code pointers.
  - `[lib]` for reusable Python modules.
  - `[bin]` for command-line entry points and thin wrappers.
  - `[Recipes]` for experiment blueprints/scripts.
  - `[Experiments]` for built experiment artifacts and result views.
  - `[docs]` for general documentation outside result reports.
- Use `(refactor)` for behavior-preserving restructuring when that distinction
  is clearer than `(feat)` or `(chore)`.

## Repository Layout
- `bin/` contains runnable command-line entry points and small scripts.
- `lib/` contains reusable library modules used by scripts and recipes.
- `modules/` contains Git submodule dependencies.
- `Recipes/` contains experiment blueprints/scripts intended to be run.
- `Experiments/` contains built experiment artifacts and result views. Use
  `.gitignore` and the reviewed task scope to determine which experiment files
  are tracked.
- `Datasets/` is external and ignored; do not track prompt datasets or dataset
  fixtures in this repository.
- `MyPaper/` and `Papers/` are unresolved top-level areas.
  Establish their layout and tracking rules in the relevant task before adding
  or reorganizing them.
