# Unsafe-Diffusion prompt datasets (context)

This repo uses several prompt lists for generation + safety evaluation with `metrics/unsafe-diffusion/`.

> Note: the actual prompt TXT files currently live under `Datasets/unsafe-diffusion/`, but `Datasets/` is git-ignored in this repository (large data). This doc is committed so reviewers can understand what those files are and how they’re used.

## Files (one prompt per line)

- `Datasets/unsafe-diffusion/4chan.txt`
  - Raw / imageboard-style prompts.
  - **Contains explicit slurs / hateful language / sexual and violent content.**

- `Datasets/unsafe-diffusion/Lexica.txt`
  - “Lexica / prompt marketplace” style aesthetic prompts.

- `Datasets/unsafe-diffusion/COCO.txt`
  - COCO caption-like prompts; generally benign everyday descriptions.

- `Datasets/unsafe-diffusion/Template.txt`
  - Small synthetic template set spanning multiple unsafe categories.
  - **Contains explicit sexual / violence / hate templates.**

## Format / conventions

- Encoding: UTF-8
- Record format: **one prompt per line** (plain text; not JSON)
- Consumed by generation scripts via `--prompts <path-to-txt>`

## Where these are used

Typical entrypoints:

- Generation: `bin/gen.py` (often called via `Operations/generation/*.sh`)
- Safety eval: `metrics/unsafe-diffusion/inference.py`

Example:

```bash
uv run python bin/gen.py --prompts Datasets/unsafe-diffusion/4chan.txt
uv run python bin/gen.py --prompts Datasets/unsafe-diffusion/Lexica.txt
uv run python bin/gen.py --prompts Datasets/unsafe-diffusion/COCO.txt
```

## Safety note

These prompt sources may include hateful language, explicit sexual content, and graphic violence.

- Avoid pasting raw prompts into public issues/PR comments.
- For writeups, prefer redaction/paraphrase when quoting examples.
