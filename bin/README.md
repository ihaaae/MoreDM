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
  `min-sdxl-light`, `sd15`, `min-sd15`, `sd20`, and `min-sd20`.
- Run safety evaluation through `lib/eval.py`.
- Run Q16 scoring through `lib/q16.py`.
- Run CLIP scoring and analysis through `lib/clip.py` or the legacy-compatible
  `lib/clip_distance*.py` entrypoints.
- Use Operations scripts to orchestrate multi-GPU experiment runs.
