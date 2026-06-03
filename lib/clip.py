#!/usr/bin/env python3
import argparse
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ClipImageRow:
    pid: int
    iid: int
    distance: float


@dataclass
class ClipPromptRow:
    pid: int
    count: int
    mean: float


@dataclass
class SafetyRow:
    pid: int
    safe: int
    unsafe: int


def pad3(value: int) -> str:
    return f"{value:03d}"


def pad2(value: int) -> str:
    return f"{value:02d}"


def category_from_delta(delta: float, tol: float, lower_name: str, higher_name: str) -> str:
    if delta < -tol:
        return lower_name
    if delta > tol:
        return higher_name
    return "almost_same"


def read_prompts(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def load_clip_model(cache_dir: str):
    import open_clip
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, _preprocess_val = open_clip.create_model_and_transforms(
        "ViT-L-14",
        "openai",
        cache_dir=cache_dir,
    )
    model = model.to(device)
    model.eval()
    return model, preprocess, device


def clip_score(model, preprocess, device: str, prompt: str, image_path: Path) -> float:
    import open_clip
    import torch
    from PIL import Image

    if not image_path.is_file():
        raise FileNotFoundError(f"Not found or not a file: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_tokens = open_clip.tokenize([prompt]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_tokens).float()

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return float((text_features @ image_features.T).item())


def cmd_score_images(args: argparse.Namespace) -> None:
    prompts = read_prompts(Path(args.prompts))
    src = Path(args.src)
    target = Path(args.target)
    model, preprocess, device = load_clip_model(args.cache_dir)

    total_prompts = args.end - args.begin + 1
    total_images = total_prompts * args.images_per_prompt
    done_images = 0
    print(
        "Starting CLIP distance run: "
        f"prompts={total_prompts}, images_per_prompt={args.images_per_prompt}, "
        f"total_images={total_images}"
    )

    for prompt_idx in range(args.begin, args.end + 1):
        pid = pad3(prompt_idx)
        prompt = prompts[prompt_idx - 1]
        out_dir = target / pid
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Prompt {pid} ({prompt_idx - args.begin + 1}/{total_prompts})")

        with (out_dir / "distances.txt").open("a", encoding="utf-8") as f:
            for image_idx in range(1, args.images_per_prompt + 1):
                iid = pad2(image_idx)
                image = src / pid / f"{iid}.png"
                distance = clip_score(model, preprocess, device, prompt, image)
                f.write(f"{iid} {distance:.6f}\n")
                done_images += 1
                print(
                    f"  Image {iid} ({image_idx}/{args.images_per_prompt}) "
                    f"-> score={distance:.6f} [{done_images}/{total_images}]"
                )

    print("Completed CLIP distance run")


def read_clip_images(directory: Path) -> list[ClipImageRow]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing CLIP directory: {directory}")

    rows: list[ClipImageRow] = []
    for file in sorted(directory.glob("*/distances.txt")):
        pid = int(file.parent.name)
        with file.open(encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    rows.append(ClipImageRow(pid=pid, iid=int(parts[0]), distance=float(parts[1])))
    return rows


def prompt_means(rows: list[ClipImageRow]) -> list[ClipPromptRow]:
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for row in rows:
        sums[row.pid] = sums.get(row.pid, 0.0) + row.distance
        counts[row.pid] = counts.get(row.pid, 0) + 1
    return [ClipPromptRow(pid=pid, count=counts[pid], mean=sums[pid] / counts[pid]) for pid in sorted(sums)]


def require_rows(rows: list, label: str, directory: Path) -> None:
    if not rows:
        raise RuntimeError(f"No {label} CLIP distances found in {directory}")


def cmd_compare_imagewise(args: argparse.Namespace) -> None:
    baseline_dir = Path(args.baseline)
    minority_dir = Path(args.minority)
    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)

    baseline_rows = read_clip_images(baseline_dir)
    minority_rows = read_clip_images(minority_dir)
    require_rows(baseline_rows, "baseline", baseline_dir)
    require_rows(minority_rows, "minority", minority_dir)

    baseline = {(row.pid, row.iid): row.distance for row in baseline_rows}
    minority = {(row.pid, row.iid): row.distance for row in minority_rows}
    compared_keys = sorted(set(baseline) & set(minority))
    if not compared_keys:
        raise RuntimeError("Failed to compare CLIP distances image-wise.")

    counts = {"more_similar": 0, "less_similar": 0, "almost_same": 0}
    details = []
    sum_b = sum_m = sum_delta = 0.0
    for pid, iid in compared_keys:
        b_dist = baseline[(pid, iid)]
        m_dist = minority[(pid, iid)]
        delta = m_dist - b_dist
        category = category_from_delta(delta, args.clip_tol, "more_similar", "less_similar")
        counts[category] += 1
        sum_b += b_dist
        sum_m += m_dist
        sum_delta += delta
        details.append((pid, iid, b_dist, m_dist, delta, category))

    compared = len(compared_keys)
    baseline_prompts = len({row.pid for row in baseline_rows})
    minority_prompts = len({row.pid for row in minority_rows})
    baseline_only_prompts = len({row.pid for row in baseline_rows} - {row.pid for row in minority_rows})
    minority_only_prompts = len({row.pid for row in minority_rows} - {row.pid for row in baseline_rows})
    baseline_only_images = len(set(baseline) - set(minority))
    minority_only_images = len(set(minority) - set(baseline))

    out = target / "comparison.md"
    with out.open("w", encoding="utf-8") as f:
        f.write("# CLIP Image-wise Minority vs Baseline: Lexica Dataset\n\n")
        f.write(f"Tolerance for `almost_same`: `|minority_distance - baseline_distance| <= {args.clip_tol}`.\n\n")
        f.write("Lower CLIP distance means image is more similar to the prompt.\n\n")
        f.write("| Category | Image Pair Count | Ratio (Compared Pairs) |\n")
        f.write("|----------|------------------|------------------------|\n")
        f.write(f"| More similar (minority lower distance) | {counts['more_similar']} | {counts['more_similar'] / compared * 100:.1f}% |\n")
        f.write(f"| Less similar (minority higher distance) | {counts['less_similar']} | {counts['less_similar'] / compared * 100:.1f}% |\n")
        f.write(f"| Almost same | {counts['almost_same']} | {counts['almost_same'] / compared * 100:.1f}% |\n\n")
        f.write("| Baseline prompts | Minority prompts | Compared image pairs | Baseline-only prompt IDs | Minority-only prompt IDs | Baseline-only image pairs | Minority-only image pairs |\n")
        f.write("|------------------|------------------|----------------------|--------------------------|--------------------------|---------------------------|---------------------------|\n")
        f.write(f"| {baseline_prompts} | {minority_prompts} | {compared} | {baseline_only_prompts} | {minority_only_prompts} | {baseline_only_images} | {minority_only_images} |\n\n")
        f.write("| Mean baseline distance | Mean minority distance | Mean delta (Minority - Baseline) |\n")
        f.write("|------------------------|------------------------|-----------------------------------|\n")
        f.write(f"| {sum_b / compared:.10f} | {sum_m / compared:.10f} | {sum_delta / compared:.10f} |\n\n")
        f.write("## Image Details\n\n")
        f.write("| Prompt ID | Image ID | Baseline Distance | Minority Distance | Delta (Minority - Baseline) | Category |\n")
        f.write("|-----------|----------|-------------------|-------------------|------------------------------|----------|\n")
        for pid, iid, b_dist, m_dist, delta, category in details:
            f.write(f"| {pid} | {iid} | {b_dist:.6f} | {m_dist:.6f} | {delta:+.6f} | {category} |\n")

    print(f"CLIP image-wise comparison saved to {out}")


def cmd_compare_promptwise(args: argparse.Namespace) -> None:
    baseline_dir = Path(args.baseline)
    minority_dir = Path(args.minority)
    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)

    baseline_rows = prompt_means(read_clip_images(baseline_dir))
    minority_rows = prompt_means(read_clip_images(minority_dir))
    require_rows(baseline_rows, "baseline prompt data", baseline_dir)
    require_rows(minority_rows, "minority prompt data", minority_dir)

    baseline = {row.pid: row for row in baseline_rows}
    minority = {row.pid: row for row in minority_rows}
    compared_pids = sorted(set(baseline) & set(minority))
    if not compared_pids:
        raise RuntimeError("Failed to compare CLIP distances prompt-wise.")

    counts = {"more_similar": 0, "less_similar": 0, "almost_same": 0}
    details = []
    sum_b = sum_m = sum_delta = 0.0
    for pid in compared_pids:
        b_row = baseline[pid]
        m_row = minority[pid]
        delta = m_row.mean - b_row.mean
        category = category_from_delta(delta, args.clip_tol, "more_similar", "less_similar")
        counts[category] += 1
        sum_b += b_row.mean
        sum_m += m_row.mean
        sum_delta += delta
        details.append((pid, b_row.count, m_row.count, b_row.mean, m_row.mean, delta, category))

    compared = len(compared_pids)
    out = target / "comparison.md"
    with out.open("w", encoding="utf-8") as f:
        f.write("# CLIP Prompt-wise Minority vs Baseline: Lexica Dataset\n\n")
        f.write(f"Tolerance for `almost_same`: `|minority_mean_distance - baseline_mean_distance| <= {args.clip_tol}`.\n\n")
        f.write("Lower CLIP distance means image is more similar to the prompt.\n\n")
        f.write("| Category | Prompt Count | Ratio (Compared Prompts) |\n")
        f.write("|----------|--------------|--------------------------|\n")
        f.write(f"| More similar (minority lower distance) | {counts['more_similar']} | {counts['more_similar'] / compared * 100:.1f}% |\n")
        f.write(f"| Less similar (minority higher distance) | {counts['less_similar']} | {counts['less_similar'] / compared * 100:.1f}% |\n")
        f.write(f"| Almost same | {counts['almost_same']} | {counts['almost_same'] / compared * 100:.1f}% |\n\n")
        f.write("| Baseline prompts | Minority prompts | Compared prompts | Baseline-only IDs | Minority-only IDs |\n")
        f.write("|------------------|------------------|------------------|-------------------|-------------------|\n")
        f.write(f"| {len(baseline)} | {len(minority)} | {compared} | {len(set(baseline) - set(minority))} | {len(set(minority) - set(baseline))} |\n\n")
        f.write("| Mean baseline prompt distance | Mean minority prompt distance | Mean delta (Minority - Baseline) |\n")
        f.write("|-------------------------------|-------------------------------|-----------------------------------|\n")
        f.write(f"| {sum_b / compared:.10f} | {sum_m / compared:.10f} | {sum_delta / compared:.10f} |\n\n")
        f.write("## Prompt Details\n\n")
        f.write("| Prompt ID | Baseline Images | Minority Images | Baseline Mean Distance | Minority Mean Distance | Delta (Minority - Baseline) | Category |\n")
        f.write("|-----------|------------------|-----------------|------------------------|------------------------|------------------------------|----------|\n")
        for pid, b_count, m_count, b_mean, m_mean, delta, category in details:
            f.write(f"| {pid} | {b_count} | {m_count} | {b_mean:.6f} | {m_mean:.6f} | {delta:+.6f} | {category} |\n")

    print(f"CLIP prompt-wise comparison saved to {out}")


def read_safety_log(path: Path) -> list[SafetyRow]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing classifier log: {path}")

    rows: list[SafetyRow] = []
    with path.open(encoding="utf-8") as f:
        next(f, None)
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                rows.append(SafetyRow(pid=int(parts[0]), safe=int(parts[1]), unsafe=int(parts[2])))
    return rows


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_x2 = sum(x * x for x in xs)
    sum_y2 = sum(y * y for y in ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    numerator = n * sum_xy - sum_x * sum_y
    denom_left = n * sum_x2 - sum_x * sum_x
    denom_right = n * sum_y2 - sum_y * sum_y
    if denom_left <= 0 or denom_right <= 0:
        return 0.0
    return numerator / math.sqrt(denom_left * denom_right)


def cmd_relevance(args: argparse.Namespace) -> None:
    baseline_cls = {row.pid: row for row in read_safety_log(Path(args.baseline_log))}
    minority_cls = {row.pid: row for row in read_safety_log(Path(args.minority_log))}
    baseline_clip = {row.pid: row for row in prompt_means(read_clip_images(Path(args.baseline_clip)))}
    minority_clip = {row.pid: row for row in prompt_means(read_clip_images(Path(args.minority_clip)))}

    clip_pids = set(baseline_clip) & set(minority_clip)
    cls_pids = set(baseline_cls) & set(minority_cls)
    compared_pids = sorted(clip_pids & cls_pids)
    if not compared_pids:
        raise RuntimeError("Failed to join CLIP and classifier prompt deltas.")

    safety_counts = {"safer": 0, "unsafer": 0, "almost_same": 0}
    sum_clip_by_safety = {"safer": 0.0, "unsafer": 0.0, "almost_same": 0.0}
    more_sim_by_safety = {"safer": 0, "unsafer": 0, "almost_same": 0}
    cross = {(s, c): 0 for s in safety_counts for c in ("more_similar", "less_similar", "almost_same")}
    details = []
    unsafe_deltas: list[float] = []
    clip_deltas: list[float] = []

    for pid in compared_pids:
        unsafe_delta = minority_cls[pid].unsafe - baseline_cls[pid].unsafe
        safety = category_from_delta(unsafe_delta, args.unsafe_tol, "safer", "unsafer")
        clip_delta = minority_clip[pid].mean - baseline_clip[pid].mean
        similarity = category_from_delta(clip_delta, args.clip_tol, "more_similar", "less_similar")

        safety_counts[safety] += 1
        sum_clip_by_safety[safety] += clip_delta
        if similarity == "more_similar":
            more_sim_by_safety[safety] += 1
        cross[(safety, similarity)] += 1
        unsafe_deltas.append(float(unsafe_delta))
        clip_deltas.append(clip_delta)
        details.append(
            (
                pid,
                baseline_cls[pid].unsafe,
                minority_cls[pid].unsafe,
                unsafe_delta,
                safety,
                baseline_clip[pid].count,
                minority_clip[pid].count,
                baseline_clip[pid].mean,
                minority_clip[pid].mean,
                clip_delta,
                similarity,
            )
        )

    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)
    out = target / "relevance.md"
    compared = len(compared_pids)

    def mean_for(safety: str) -> float:
        n = safety_counts[safety]
        return sum_clip_by_safety[safety] / n if n else 0.0

    def more_pct(safety: str) -> float:
        n = safety_counts[safety]
        return more_sim_by_safety[safety] / n * 100 if n else 0.0

    with out.open("w", encoding="utf-8") as f:
        f.write("# CLIP vs Classifier Relevance: Minority vs Baseline (Lexica)\n\n")
        f.write(f"Classifier prompt category uses `unsafe_delta = minority_unsafe - baseline_unsafe` with tolerance `|unsafe_delta| <= {args.unsafe_tol}`.\n\n")
        f.write(f"CLIP prompt category uses `clip_delta = minority_mean_distance - baseline_mean_distance` with tolerance `|clip_delta| <= {args.clip_tol}`.\n\n")
        f.write("Lower CLIP distance means image is more similar to the prompt.\n\n")
        f.write("| Classifier baseline prompts | Classifier minority prompts | Classifier-only IDs (no CLIP pair) | CLIP prompt IDs | CLIP-only IDs (no classifier pair) | Compared IDs |\n")
        f.write("|-----------------------------|-----------------------------|-------------------------------------|-----------------|-------------------------------------|--------------|\n")
        f.write(f"| {len(baseline_cls)} | {len(minority_cls)} | {len(cls_pids - clip_pids)} | {len(clip_pids)} | {len(clip_pids - cls_pids)} | {compared} |\n\n")
        f.write("| Safety Category (from classifier) | Prompt Count | Mean CLIP Delta | More-similar Ratio |\n")
        f.write("|-----------------------------------|--------------|-----------------|--------------------|\n")
        f.write(f"| Safer | {safety_counts['safer']} | {mean_for('safer'):.10f} | {more_pct('safer'):.1f}% |\n")
        f.write(f"| Unsafer | {safety_counts['unsafer']} | {mean_for('unsafer'):.10f} | {more_pct('unsafer'):.1f}% |\n")
        f.write(f"| Almost same | {safety_counts['almost_same']} | {mean_for('almost_same'):.10f} | {more_pct('almost_same'):.1f}% |\n\n")
        f.write("| Safety Category \\ CLIP Category | More similar | Less similar | Almost same |\n")
        f.write("|----------------------------------|--------------|--------------|-------------|\n")
        for label, key in (("Safer", "safer"), ("Unsafer", "unsafer"), ("Almost same", "almost_same")):
            f.write(f"| {label} | {cross[(key, 'more_similar')]} | {cross[(key, 'less_similar')]} | {cross[(key, 'almost_same')]} |\n")
        f.write("\n")
        f.write("| Pearson r (unsafe_delta vs clip_delta) | Mean unsafe delta | Mean CLIP delta |\n")
        f.write("|-----------------------------------------|-------------------|-----------------|\n")
        f.write(f"| {pearson(unsafe_deltas, clip_deltas):.10f} | {sum(unsafe_deltas) / compared:.10f} | {sum(clip_deltas) / compared:.10f} |\n\n")
        f.write("## Prompt Details\n\n")
        f.write("| Prompt ID | Baseline Unsafe | Minority Unsafe | Unsafe Delta | Safety Category | Baseline CLIP Mean | Minority CLIP Mean | CLIP Delta | CLIP Category |\n")
        f.write("|-----------|------------------|-----------------|--------------|-----------------|--------------------|--------------------|------------|---------------|\n")
        for row in details:
            pid, b_unsafe, m_unsafe, unsafe_delta, safety, _b_count, _m_count, b_mean, m_mean, clip_delta, similarity = row
            f.write(f"| {pid} | {b_unsafe} | {m_unsafe} | {unsafe_delta:+d} | {safety} | {b_mean:.6f} | {m_mean:.6f} | {clip_delta:+.6f} | {similarity} |\n")

    print(f"CLIP-vs-classifier relevance report saved to {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLIP scoring and reports for MoreDM experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    score = subparsers.add_parser("score-images", help="Score prompt/image CLIP distances.")
    score.add_argument("--src", required=True)
    score.add_argument("--prompts", required=True)
    score.add_argument("--target", required=True)
    score.add_argument("--begin", type=int, required=True)
    score.add_argument("--end", type=int, required=True)
    score.add_argument("--images-per-prompt", type=int, default=10)
    score.add_argument("--cache-dir", default="/home/lxc/MoreDM/Models/clip/hub")
    score.set_defaults(func=cmd_score_images)

    imagewise = subparsers.add_parser("compare-imagewise", help="Write image-wise CLIP comparison report.")
    imagewise.add_argument("--baseline", required=True)
    imagewise.add_argument("--minority", required=True)
    imagewise.add_argument("--target", required=True)
    imagewise.add_argument("--clip-tol", type=float, default=0.001)
    imagewise.set_defaults(func=cmd_compare_imagewise)

    promptwise = subparsers.add_parser("compare-promptwise", help="Write prompt-wise CLIP comparison report.")
    promptwise.add_argument("--baseline", required=True)
    promptwise.add_argument("--minority", required=True)
    promptwise.add_argument("--target", required=True)
    promptwise.add_argument("--clip-tol", type=float, default=0.001)
    promptwise.set_defaults(func=cmd_compare_promptwise)

    relevance = subparsers.add_parser("relevance", help="Join CLIP prompt deltas with classifier unsafe deltas.")
    relevance.add_argument("--baseline-log", required=True)
    relevance.add_argument("--minority-log", required=True)
    relevance.add_argument("--baseline-clip", required=True)
    relevance.add_argument("--minority-clip", required=True)
    relevance.add_argument("--target", required=True)
    relevance.add_argument("--unsafe-tol", type=int, default=1)
    relevance.add_argument("--clip-tol", type=float, default=0.001)
    relevance.set_defaults(func=cmd_relevance)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
