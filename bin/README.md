# bin/ - Project Scripts

Runnable project entry points live here. They are thin wrappers or standalone
tools that call into `lib/`, `modules/`, or Python packages.

Scripts:

- `gen`: generation wrapper for `lib/gen.py`.
- `make_families.py`: prompt-family variants for attribution experiments.
- `make_templates.py`: template prompt builder for injection experiments.

Conventions:

- Invoke scripts from the repository root with `uv run`.
- Keep external research code in `modules/` submodules.
- Keep reusable project implementation in `lib/`.
- Run low-level generation through `lib/gen.py`; it supports `sdxl-light`,
  `min-sdxl-light`, `sd15`, `min-sd15`, `sd20`, `min-sd20`, `sd3`, `min-sd3`,
  `sd35`, and `min-sd35`.
- `sdxl-light` / `min-sdxl-light` mean **SDXL-Lightning**. The separate
  `sdxl` / `min-sdxl` audit modes mean **SDXL-base** and require an explicit
  local `--sdxl-base-path`. Never substitute one model's safe cohort for another.
- Consult `ARCHIVE.md` before using historical SD3-family results; the legacy
  velocity objective is not the current Eq6 implementation.
- Run safety evaluation through `lib/eval.py`.
- Run Q16 scoring through `lib/q16.py`.
- Run Q16 directory scoring through `lib/q16_batch.py`.
- Run CLIP scoring and analysis through `lib/clip.py` or the legacy-compatible
  `lib/clip_distance*.py` entrypoints.
- Use `Recipes/` scripts to orchestrate multi-GPU experiment runs.
