"""Build a small benchmark-patch prompt slice from local source files."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

CATEGORY_MAP = {
    "sexual": "sexual",
    "nudity": "sexual",
    "violence": "violent_disturbing",
    "violent": "violent_disturbing",
    "disturbing": "violent_disturbing",
    "hate": "hateful_harassment",
    "harassment": "hateful_harassment",
    "political": "political_public_figure",
    "public_figure": "political_public_figure",
    "illegal": "illegal",
    "copyright": "copyright_character",
    "character": "copyright_character",
    "benign": "benign_control",
    "safe": "benign_control",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the 300-prompt benchmark patch slice.")
    parser.add_argument("--i2p", type=Path, required=True)
    parser.add_argument("--t2i-risky", type=Path, required=True)
    parser.add_argument("--p4d", type=Path, required=True)
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--n-i2p", type=int, default=100)
    parser.add_argument("--n-risky", type=int, default=100)
    parser.add_argument("--n-adversarial", type=int, default=50)
    parser.add_argument("--n-benign", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_category(value: Any, default: str = "other") -> str:
    if value is None:
        return default
    text = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    for key, normalized in CATEGORY_MAP.items():
        if key in text:
            return normalized
    return default


def read_text_prompts(path: Path, source: str, default_category: str) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for index, line in enumerate(f, start=1):
            prompt = line.strip()
            if not prompt:
                continue
            rows.append({
                "prompt": prompt,
                "source_dataset": source,
                "source_id": str(index),
                "source_category": None,
                "normalized_category": default_category,
                "license_access_note": "Preserved from user-provided local source path; verify source license before publication.",
            })
    return rows


def read_structured(path: Path, source: str, default_category: str) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".txt", ".prompts"}:
        return read_text_prompts(path, source, default_category)
    if path.suffix.lower() == ".jsonl":
        raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload if isinstance(payload, list) else payload.get("data", payload.get("prompts", []))
    elif path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as f:
            raw = list(csv.DictReader(f))
    else:
        raise ValueError(f"Unsupported source format: {path}")

    rows = []
    for index, item in enumerate(raw, start=1):
        if isinstance(item, str):
            prompt = item
            category = None
            source_id = index
        else:
            prompt = item.get("prompt") or item.get("text") or item.get("caption")
            category = item.get("category") or item.get("label") or item.get("harm_category") or item.get("source_category")
            source_id = item.get("id") or item.get("source_id") or index
        if not prompt:
            continue
        rows.append({
            "prompt": str(prompt).strip(),
            "source_dataset": source,
            "source_id": str(source_id),
            "source_category": category,
            "normalized_category": normalize_category(category, default_category),
            "license_access_note": "Preserved from user-provided local source path; verify source license before publication.",
        })
    return rows


def sample(rng: random.Random, rows: list[dict[str, Any]], n: int, label: str) -> list[dict[str, Any]]:
    if len(rows) < n:
        raise ValueError(f"Not enough {label} prompts: need {n}, found {len(rows)}")
    return rng.sample(rows, n)


def main() -> None:
    args = parse_args()
    for path in (args.i2p, args.t2i_risky, args.p4d, args.coco):
        if not path.is_file():
            raise SystemExit(f"Missing source file: {path}")

    rng = random.Random(args.seed)
    selected = []
    selected.extend(sample(rng, read_structured(args.i2p, "I2P", "other"), args.n_i2p, "I2P"))
    selected.extend(sample(rng, read_structured(args.t2i_risky, "T2I-RiskyPrompt", "other"), args.n_risky, "T2I-RiskyPrompt"))
    selected.extend(sample(rng, read_structured(args.p4d, "P4D/adversarial", "sexual"), args.n_adversarial, "adversarial"))
    selected.extend(sample(rng, read_structured(args.coco, "COCO/benign", "benign_control"), args.n_benign, "benign"))

    args.target.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    with args.target.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(row["prompt"] + "\n")
    with args.metadata.open("w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "created_by": "lib/build_prompt_slice.py",
                "seed": args.seed,
                "caveat": "I2P labels are approximate and should not be treated as ground-truth unsafe categories without validation.",
                "normalized_categories": sorted(set(CATEGORY_MAP.values()) | {"other"}),
            },
            "prompts": [{"benchmark_id": f"{i:03d}", **row} for i, row in enumerate(selected, start=1)],
        }, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
