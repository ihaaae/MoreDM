"""Shared helpers for benchmark-patch safety metrics."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

IMAGE_RE = re.compile(r"(?P<prompt>\d{3})[/\\](?P<image>\d{2}\.png)$")


def collect_images(image_root: Path) -> list[Path]:
    """Return generated PNGs under prompt folders, excluding record artifacts."""
    return sorted(
        path
        for path in image_root.rglob("*.png")
        if "record" not in path.relative_to(image_root).parts
    )


def canonical_image_id(path: str | Path, image_root: Path | None = None) -> str:
    """Normalize old/new metric keys to `001/01.png`.

    Existing artifacts mix absolute-ish paths, run-relative paths, and CLIP rows
    split across `default-clip/001/distances.txt`. The benchmark patch joins on
    this canonical identifier.
    """
    p = Path(path)
    if image_root is not None:
        try:
            return p.relative_to(image_root).as_posix()
        except ValueError:
            pass
    match = IMAGE_RE.search(str(path).replace("\\", "/"))
    if match:
        return f"{match.group('prompt')}/{match.group('image')}"
    return p.as_posix()


def prompt_id(image_id: str) -> str:
    return image_id.split("/", 1)[0]


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def load_unsafe_diffusion(path: Path) -> dict[str, bool]:
    if path.is_dir():
        result: dict[str, bool] = {}
        for json_path in sorted(path.glob("*.json")):
            result.update(load_unsafe_diffusion(json_path))
        return result

    raw = read_json(path)
    if isinstance(raw, dict) and "images" in raw:
        raw = raw["images"]
    result: dict[str, bool] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            unsafe = bool(value.get("unsafe"))
        else:
            unsafe = str(value) == "1"
        result[canonical_image_id(key)] = unsafe
    return result


def load_q16(path: Path) -> dict[str, float | None]:
    if path.is_dir():
        result: dict[str, float | None] = {}
        for scores_path in sorted(path.glob("*/scores.txt")):
            pid = scores_path.parent.name
            with scores_path.open(encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    image_name = parts[0]
                    if not image_name.endswith(".png"):
                        image_name = f"{int(image_name):02d}.png"
                    result[f"{pid}/{image_name}"] = float(parts[1])
        if result:
            return result
        return {}

    raw = read_json(path)
    if isinstance(raw, dict) and "images" in raw:
        raw = raw["images"]
    result: dict[str, float | None] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            value = value.get("score")
        result[canonical_image_id(key)] = None if value is None else float(value)
    return result


def load_boolean_detector(path: Path) -> dict[str, bool | None]:
    raw = read_json(path)
    images = raw.get("images", raw) if isinstance(raw, dict) else {}
    result: dict[str, bool | None] = {}
    for key, value in images.items():
        if value is None:
            result[canonical_image_id(key)] = None
        elif isinstance(value, dict):
            result[canonical_image_id(key)] = bool(value.get("unsafe"))
        else:
            result[canonical_image_id(key)] = bool(value)
    return result


def load_clip_scores(clip_dir: Path) -> dict[str, float]:
    """Load CLIP cosine similarity from `*/distances.txt` legacy folders."""
    result: dict[str, float] = {}
    if not clip_dir.exists():
        return result
    for score_file in sorted(clip_dir.glob("*/distances.txt")):
        pid = score_file.parent.name
        with score_file.open(encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                image_name = parts[0]
                if not image_name.endswith(".png"):
                    image_name = f"{int(image_name):02d}.png"
                result[f"{pid}/{image_name}"] = float(parts[1])
    return result


def mean(values: Iterable[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1 - frac) + ordered[high] * frac
