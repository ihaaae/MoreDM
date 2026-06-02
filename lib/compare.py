import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SafetyRow:
    pid: int
    safe: int
    unsafe: int


def parse_args():
    parser = argparse.ArgumentParser(description="Compare MoreDM safety logs.")
    parser.add_argument("--root", default="/home/lxc/MoreDM")

    subparsers = parser.add_subparsers(dest="mode", required=True)

    dataset = subparsers.add_parser("dataset", help="Compare two Vanilla dataset logs.")
    dataset.add_argument("--left", required=True)
    dataset.add_argument("--right", required=True)

    strategy = subparsers.add_parser("strategy", help="Compare Vanilla and Minority aggregate rates.")
    strategy.add_argument("--dataset", required=True)
    strategy.add_argument("--subset", required=True)

    promptwise = subparsers.add_parser("promptwise", help="Compare Vanilla and Minority prompt-wise unsafe counts.")
    promptwise.add_argument("--dataset", required=True)
    promptwise.add_argument("--subset", required=True)
    promptwise.add_argument("--tolerance", type=int, default=1)

    return parser.parse_args()


def log_name(subset):
    return f"{subset.lower()}.log"


def dataset_log(base, subset):
    return base / "Dataset" / f"SdxlLight-{subset}" / log_name(subset)


def minority_log(base, dataset, subset):
    return base / "Minority" / f"SdxlLight-{dataset}-{subset}" / "default" / log_name(subset)


def read_log(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing log: {path}")

    rows = []
    with path.open(encoding="utf-8") as f:
        next(f, None)
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                rows.append(SafetyRow(pid=int(parts[0]), safe=int(parts[1]), unsafe=int(parts[2])))
    return rows


def summarize(rows):
    prompts = len(rows)
    safe = sum(row.safe for row in rows)
    unsafe = sum(row.unsafe for row in rows)
    images = safe + unsafe
    safe_pct = safe / images * 100 if images else 0.0
    unsafe_pct = unsafe / images * 100 if images else 0.0
    return {
        "prompts": prompts,
        "images": images,
        "safe": safe,
        "unsafe": unsafe,
        "safe_pct": safe_pct,
        "unsafe_pct": unsafe_pct,
    }


def write_aggregate(out, title, label_left, rows_left, label_right, rows_right):
    left = summarize(rows_left)
    right = summarize(rows_right)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        if title:
            f.write(f"# {title}\n\n")
        f.write("| Dataset | Total Prompts | Total Images | Safe | Safe % | Unsafe | Unsafe % |\n")
        f.write("|---------|---------------|--------------|------|--------|--------|----------|\n")
        for label, stats in ((label_left, left), (label_right, right)):
            f.write(
                f"| {label} | {stats['prompts']} | {stats['images']} | "
                f"{stats['safe']} | {stats['safe_pct']:.1f}% | "
                f"{stats['unsafe']} | {stats['unsafe_pct']:.1f}% |\n"
            )


def compare_dataset(args):
    base = Path(args.root) / "Experiments" / "Safety"
    left_log = dataset_log(base, args.left)
    right_log = dataset_log(base, args.right)
    out = base / "Dataset" / f"SdxlLight-{args.left}-{args.right}" / "comparison.md"
    write_aggregate(out, "", args.left, read_log(left_log), args.right, read_log(right_log))
    print(f"Comparison saved to {out}")


def compare_strategy(args):
    base = Path(args.root) / "Experiments" / "Safety"
    vanilla = read_log(dataset_log(base, args.subset))
    minority = read_log(minority_log(base, args.dataset, args.subset))
    out = base / "Comparison" / f"Minority-vs-Vanilla-{args.subset}" / "comparison.md"
    write_aggregate(
        out,
        f"Minority vs Vanilla: {args.subset} Dataset",
        "Vanilla",
        vanilla,
        "Minority/default",
        minority,
    )
    print(f"Comparison saved to {out}")


def compare_promptwise(args):
    base = Path(args.root) / "Experiments" / "Safety"
    vanilla_rows = read_log(dataset_log(base, args.subset))
    minority_rows = read_log(minority_log(base, args.dataset, args.subset))
    vanilla_by_pid = {row.pid: row for row in vanilla_rows}
    minority_by_pid = {row.pid: row for row in minority_rows}
    compared_pids = sorted(set(vanilla_by_pid) & set(minority_by_pid))

    if not compared_pids:
        raise RuntimeError("No overlapping prompt IDs to compare.")

    details = []
    counts = {"safer": 0, "unsafer": 0, "almost_same": 0}
    for pid in compared_pids:
        vanilla_unsafe = vanilla_by_pid[pid].unsafe
        minority_unsafe = minority_by_pid[pid].unsafe
        delta = minority_unsafe - vanilla_unsafe
        if delta < -args.tolerance:
            category = "safer"
        elif delta > args.tolerance:
            category = "unsafer"
        else:
            category = "almost_same"
        counts[category] += 1
        details.append((pid, vanilla_unsafe, minority_unsafe, delta, category))

    compared = len(compared_pids)
    vanilla_only = len(set(vanilla_by_pid) - set(minority_by_pid))
    minority_only = len(set(minority_by_pid) - set(vanilla_by_pid))
    out = base / "Comparison" / f"PromptWise-Minority-vs-Vanilla-{args.subset}" / "comparison.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        f.write(f"# Prompt-wise Minority vs Vanilla: {args.subset} Dataset\n\n")
        f.write(
            "Tolerance for `almost_same`: "
            f"`|minority_unsafe - vanilla_unsafe| <= {args.tolerance}` image(s) per prompt.\n\n"
        )
        f.write("| Category | Prompt Count | Ratio (Compared Prompts) |\n")
        f.write("|----------|--------------|--------------------------|\n")
        for category in ("safer", "unsafer", "almost_same"):
            ratio = counts[category] / compared * 100
            f.write(f"| {category} | {counts[category]} | {ratio:.1f}% |\n")
        f.write("\n")
        f.write("| Vanilla prompts | Minority prompts | Compared prompts | Vanilla-only IDs | Minority-only IDs |\n")
        f.write("|-----------------|------------------|------------------|------------------|-------------------|\n")
        f.write(f"| {len(vanilla_rows)} | {len(minority_rows)} | {compared} | {vanilla_only} | {minority_only} |\n\n")
        f.write("## Prompt Details\n\n")
        f.write("| Prompt ID | Vanilla Unsafe | Minority Unsafe | Delta (Minority - Vanilla) | Category |\n")
        f.write("|-----------|----------------|-----------------|----------------------------|----------|\n")
        for pid, vanilla_unsafe, minority_unsafe, delta, category in details:
            f.write(f"| {pid} | {vanilla_unsafe} | {minority_unsafe} | {delta:+d} | {category} |\n")

    print(f"Prompt-wise comparison saved to {out}")


def main():
    args = parse_args()
    if args.mode == "dataset":
        compare_dataset(args)
    elif args.mode == "strategy":
        compare_strategy(args)
    elif args.mode == "promptwise":
        compare_promptwise(args)


if __name__ == "__main__":
    main()
