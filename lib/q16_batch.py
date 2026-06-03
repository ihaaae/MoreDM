"""Batch score a directory of images with Q16 and write q16_scores.json."""

import argparse
import json
import sys
from pathlib import Path

from lib.q16 import score_image


def collect_images(images_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in images_dir.rglob("*.png")
        if "record" not in path.relative_to(images_dir).parts
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Q16 batch score a directory of images.")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write q16_scores.json.")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)

    if not images_dir.is_dir():
        print(f"Images directory not found: {images_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = collect_images(images_dir)
    scores = {}
    for index, image_path in enumerate(image_files, start=1):
        key = str(image_path.relative_to(images_dir))
        try:
            scores[key] = score_image(str(image_path))
        except Exception as exc:
            print(f"Error scoring {image_path}: {exc}", file=sys.stderr)
            scores[key] = None
        if index % 100 == 0:
            print(f"Scored {index}/{len(image_files)}", file=sys.stderr)

    output_file = output_dir / "q16_scores.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    print(f"Wrote {len(scores)} scores to {output_file}")


if __name__ == "__main__":
    main()
