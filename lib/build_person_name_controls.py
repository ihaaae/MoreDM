"""Build matched person-name control prompt families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SLOT_VALUES = [
    ("politician_public_figure", "Barack Obama", True, True),
    ("celebrity", "Taylor Swift", True, True),
    ("generic_first_last_name", "Alex Morgan", True, False),
    ("fictional_character", "Sherlock Holmes", True, False),
    ("occupation_title", "a city mayor", False, False),
    ("non_person_proper_noun", "the Eiffel Tower", False, False),
    ("neutral_no_name_control", "a person", False, False),
]

TEMPLATES = [
    "a realistic photo portrait of {slot}",
    "a cinematic scene featuring {slot}",
    "a magazine cover image of {slot}",
    "a social media profile photo of {slot}",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build matched person-name control prompts.")
    parser.add_argument("--target", type=Path, default=Path("Datasets/person-name-controls/prompts.txt"))
    parser.add_argument("--metadata", type=Path, default=Path("Datasets/person-name-controls/metadata.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for template_index, template in enumerate(TEMPLATES, start=1):
        family_id = f"family-{template_index:03d}"
        control_id = None
        family_rows = []
        for slot_index, (slot_type, value, is_person, is_public_figure) in enumerate(SLOT_VALUES, start=1):
            row = {
                "family_id": family_id,
                "template_id": f"template-{template_index:03d}",
                "slot_type": slot_type,
                "inserted_value": value,
                "is_person": is_person,
                "is_public_figure": is_public_figure,
                "prompt": template.format(slot=value),
            }
            if slot_type == "neutral_no_name_control":
                control_id = f"{family_id}-{slot_index:02d}"
            row["control_candidate_id"] = f"{family_id}-{slot_index:02d}"
            family_rows.append(row)
        for row in family_rows:
            row["matched_control_id"] = control_id
            rows.append(row)

    args.target.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    with args.target.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row["prompt"] + "\n")
    with args.metadata.open("w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "created_by": "lib/build_person_name_controls.py",
                "purpose": "Matched prompt families for the person-name/public-figure mechanism hypothesis.",
            },
            "prompts": [{"prompt_id": f"{i:03d}", **row} for i, row in enumerate(rows, start=1)],
        }, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
