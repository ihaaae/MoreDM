#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SafetyRow:
    pid: int
    safe: int
    unsafe: int


@dataclass
class SpecialHit:
    dataset: str
    src_line: int
    baseline_unsafe: int
    minority_unsafe: int
    delta: int


def read_safety_log(path: Path) -> dict[int, SafetyRow]:
    if not path.is_file():
        raise FileNotFoundError(path)

    rows: dict[int, SafetyRow] = {}
    with path.open(encoding="utf-8") as f:
        next(f, None)
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                pid = int(parts[0])
                rows[pid] = SafetyRow(pid=pid, safe=int(parts[1]), unsafe=int(parts[2]))
    return rows


def read_prompt(path: Path, line_number: int) -> str:
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if idx == line_number:
                return line.rstrip("\n")
    raise IndexError(f"Line {line_number} not found in {path}")


def collect_hits(dataset: str, baseline_log: Path, minority_log: Path, max_base: int, min_delta: int) -> list[SpecialHit]:
    if not baseline_log.is_file():
        print(f"Skipping {dataset}: missing {baseline_log}")
        return []
    if not minority_log.is_file():
        print(f"Skipping {dataset}: missing {minority_log}")
        return []

    baseline = read_safety_log(baseline_log)
    minority = read_safety_log(minority_log)
    hits: list[SpecialHit] = []
    for pid in sorted(set(baseline) & set(minority)):
        b_unsafe = baseline[pid].unsafe
        m_unsafe = minority[pid].unsafe
        delta = m_unsafe - b_unsafe
        if b_unsafe <= max_base and delta >= min_delta:
            hits.append(SpecialHit(dataset, pid, b_unsafe, m_unsafe, delta))
    return hits


def cmd_select_special(args: argparse.Namespace) -> None:
    root = Path(args.root)
    safety_base = root / "Experiments" / "Safety"
    datasets = root / "Datasets" / "unsafe-diffusion"
    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)

    hits: list[SpecialHit] = []
    hits.extend(
        collect_hits(
            "Lexica",
            safety_base / "Dataset" / "1-SdxlLight-Lexica" / "lexica.log",
            safety_base / "Minority" / "SdxlLight-Lexica" / "default" / "lexica.log",
            args.baseline_max_unsafe,
            args.min_delta,
        )
    )
    hits.extend(
        collect_hits(
            "4Chan",
            safety_base / "Dataset" / "1-SdxlLight-4Chan" / "4chan.log",
            safety_base / "Minority" / "SdxlLight-4Chan" / "default" / "4chan.log",
            args.baseline_max_unsafe,
            args.min_delta,
        )
    )
    hits.extend(
        collect_hits(
            "COCO",
            safety_base / "Dataset" / "1-SdxlLight-COCO" / "coco.log",
            safety_base / "Minority" / "SdxlLight-COCO" / "default" / "coco.log",
            args.baseline_max_unsafe,
            args.min_delta,
        )
    )

    if not hits:
        raise RuntimeError(
            "No special prompts found "
            f"(BASELINE_MAX_UNSAFE={args.baseline_max_unsafe}, MIN_DELTA={args.min_delta})"
        )

    hits.sort(key=lambda hit: hit.delta, reverse=True)
    prompt_files = {
        "Lexica": datasets / "Lexica.txt",
        "4Chan": datasets / "4Chan.txt",
        "COCO": datasets / "COCO.txt",
    }
    tsv = target / "special.tsv"
    txt = target / "special.txt"

    with tsv.open("w", encoding="utf-8") as tsv_f, txt.open("w", encoding="utf-8") as txt_f:
        tsv_f.write("sp_id\tdataset\tsrc_line\tbaseline_unsafe\tminority_unsafe\tdelta\tprompt\n")
        for idx, hit in enumerate(hits, start=1):
            sp_id = f"sp-{idx:03d}"
            prompt_text = read_prompt(prompt_files[hit.dataset], hit.src_line)
            tsv_f.write(
                f"{sp_id}\t{hit.dataset}\t{hit.src_line}\t{hit.baseline_unsafe}\t"
                f"{hit.minority_unsafe}\t{hit.delta}\t{prompt_text}\n"
            )
            txt_f.write(f"{prompt_text}\n")

    print(f"Found {len(hits)} special prompt(s)")
    print(f"Manifest: {tsv}")
    print(f"Prompts:  {txt}")


def read_special_prompts(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path} (run 01-select-special-prompts.sh first)")

    prompts: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        next(f, None)
        for line in f:
            parts = line.rstrip("\n").split("\t", 6)
            if len(parts) == 7:
                prompts[parts[0]] = parts[6]
    return prompts


def read_manifest(path: Path) -> dict[str, tuple[str, str, str]]:
    rows: dict[str, tuple[str, str, str]] = {}
    if not path.is_file():
        return rows

    with path.open(encoding="utf-8") as f:
        next(f, None)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                vid = f"{int(parts[0]):03d}"
                rows[vid] = (parts[1], parts[2], parts[3])
    return rows


def read_family(path: Path) -> dict[str, str]:
    prompts: dict[str, str] = {}
    if not path.is_file():
        return prompts

    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            prompts[f"{idx:03d}"] = line.rstrip("\n")
    return prompts


def cmd_compare(args: argparse.Namespace) -> None:
    base = Path(args.base)
    safety_base = base / "Safety"
    families_dir = base / "Families"
    comparison_dir = base / "Comparison"
    special_prompts = read_special_prompts(base / "special.tsv")
    comparison_dir.mkdir(parents=True, exist_ok=True)

    element_stats: list[tuple[str, str]] = []

    for family_dir in sorted(families_dir.glob("sp-*")):
        if not family_dir.is_dir():
            continue
        sp_id = family_dir.name
        log_base = safety_base / "Baseline" / f"{sp_id}.log"
        log_min = safety_base / "Minority" / f"{sp_id}.log"

        if not log_base.is_file():
            print(f"Skipping {sp_id}: missing {log_base}")
            continue
        if not log_min.is_file():
            print(f"Skipping {sp_id}: missing {log_min}")
            continue

        baseline = read_safety_log(log_base)
        minority = read_safety_log(log_min)
        manifest = read_manifest(family_dir / "manifest.tsv")
        _family_prompts = read_family(family_dir / "family.txt")

        rows = []
        for vid_int in sorted(set(baseline) & set(minority)):
            vid = f"{vid_int:03d}"
            b_unsafe = baseline[vid_int].unsafe
            m_unsafe = minority[vid_int].unsafe
            delta = m_unsafe - b_unsafe
            still_special = "YES" if delta >= args.special_threshold else "no"
            etype, original, new = manifest.get(vid, ("-", "-", "-"))
            label = "original" if vid == "001" else f"var-{vid_int - 1:03d}"
            rows.append((label, etype, original, new, b_unsafe, m_unsafe, delta, still_special, vid))

            if vid != "001" and etype != "-":
                element_stats.append((etype, "not_key" if still_special == "YES" else "key"))

        out_dir = comparison_dir / sp_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "comparison.md"
        with out.open("w", encoding="utf-8") as f:
            f.write(f"# Attribution: {sp_id}\n\n")
            f.write(f"**Original prompt:** {special_prompts.get(sp_id, '')}\n\n")
            f.write(f"Threshold for \"still special\": `delta >= {args.special_threshold}`.\n\n")
            f.write("| Variant | Element Changed | Original -> New | Baseline Unsafe | Minority Unsafe | Delta | Still Special? |\n")
            f.write("|---------|-----------------|-----------------|-----------------|-----------------|-------|----------------|\n")
            for label, etype, original, new, b_unsafe, m_unsafe, delta, still_special, _vid in rows:
                f.write(
                    f"| {label} | {etype} | {original} -> {new} | {b_unsafe} | "
                    f"{m_unsafe} | {delta:+d} | {still_special} |\n"
                )
        print(f"  {out}")

    summary = comparison_dir / "summary.md"
    if not element_stats:
        print("No element stats to summarize.")
        return

    totals: dict[str, int] = {}
    keys: dict[str, int] = {}
    not_keys: dict[str, int] = {}
    for element_type, role in element_stats:
        totals[element_type] = totals.get(element_type, 0) + 1
        if role == "key":
            keys[element_type] = keys.get(element_type, 0) + 1
        else:
            not_keys[element_type] = not_keys.get(element_type, 0) + 1

    sorted_types = sorted(totals, key=lambda item: keys.get(item, 0) / totals[item], reverse=True)
    with summary.open("w", encoding="utf-8") as f:
        f.write("# Attribution Summary\n\n")
        f.write("How often each element type is a **key contributor** to specialness.\n")
        f.write("An element is \"key\" when changing it causes the prompt to lose its specialness\n")
        f.write("(delta drops below threshold).\n\n")
        f.write("| Element Type | Times Key | Times Not Key | Total | Key Ratio |\n")
        f.write("|--------------|-----------|---------------|-------|-----------|\n")
        for element_type in sorted_types:
            key_count = keys.get(element_type, 0)
            not_key_count = not_keys.get(element_type, 0)
            total = totals[element_type]
            f.write(f"| {element_type} | {key_count} | {not_key_count} | {total} | {key_count / total * 100:.0f}% |\n")

    print("")
    print(f"Summary: {summary}")
    print("Attribution analysis complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attribution helpers for MoreDM experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select-special", help="Select special prompts from safety logs.")
    select.add_argument("--root", default="/home/lxc/MoreDM")
    select.add_argument("--target", default="/home/lxc/MoreDM/Experiments/Attribution")
    select.add_argument("--baseline-max-unsafe", type=int, default=3)
    select.add_argument("--min-delta", type=int, default=4)
    select.set_defaults(func=cmd_select_special)

    compare = subparsers.add_parser("compare", help="Write attribution family reports.")
    compare.add_argument("--base", default="/home/lxc/MoreDM/Experiments/Attribution")
    compare.add_argument("--special-threshold", type=int, default=4)
    compare.set_defaults(func=cmd_compare)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
