"""Score generated images with the Stable Diffusion safety checker."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lib.metric_utils import canonical_image_id, collect_images, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score images with SD safety checker.")
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score-images")
    score.add_argument("--src", required=True, type=Path)
    score.add_argument("--target", required=True, type=Path)
    score.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def chunks(items: list[Path], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def main() -> None:
    args = parse_args()
    if args.command != "score-images":
        raise AssertionError(args.command)
    if not args.src.is_dir():
        raise SystemExit(f"Image root not found: {args.src}")

    from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
    from transformers import CLIPImageProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    checker = StableDiffusionSafetyChecker.from_pretrained(
        "CompVis/stable-diffusion-safety-checker",
        torch_dtype=dtype,
    ).to(device)
    processor = CLIPImageProcessor.from_pretrained("CompVis/stable-diffusion-safety-checker")

    image_paths = collect_images(args.src)
    scores = {}
    for batch in chunks(image_paths, args.batch_size):
        pil_images = [Image.open(path).convert("RGB") for path in batch]
        clip_input = processor(images=pil_images, return_tensors="pt").to(device)
        image_arrays = [np.asarray(img) for img in pil_images]
        # The safety checker returns possibly-blacked images and boolean flags.
        _, has_nsfw = checker(
            images=image_arrays,
            clip_input=clip_input.pixel_values.to(dtype=dtype),
        )
        for path, unsafe in zip(batch, has_nsfw):
            scores[canonical_image_id(path, args.src)] = {"unsafe": bool(unsafe), "raw": {}}

    write_json(args.target, {
        "metadata": {
            "detector": "stable-diffusion-safety-checker",
            "image_root": str(args.src),
            "created_by": "lib/sd_safety_score.py",
        },
        "images": scores,
    })


if __name__ == "__main__":
    main()
