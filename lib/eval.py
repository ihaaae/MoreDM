import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated images with unsafe-diffusion.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--strategy", required=True, choices=["Vanilla", "Minority"])
    parser.add_argument("--root", default="/home/lxc/MoreDM")
    parser.add_argument("--begin", type=int, default=1)
    parser.add_argument("--end", type=int, default=50)
    return parser.parse_args()


def paths(root, dataset, subset, strategy):
    base = Path(root)
    if strategy == "Vanilla":
        src = base / "Experiments" / "Text2Image" / strategy / f"SdxlLight-{dataset}-{subset}"
        target = base / "Experiments" / "Safety" / "Dataset" / f"SdxlLight-{subset}"
    else:
        src = base / "Experiments" / "Text2Image" / strategy / f"SdxlLight-{dataset}-{subset}" / "default"
        target = base / "Experiments" / "Safety" / strategy / f"SdxlLight-{dataset}-{subset}" / "default"
    return src, target


def count_predictions(json_path):
    with json_path.open(encoding="utf-8") as f:
        predictions = json.load(f)

    safe = 0
    unsafe = 0
    for value in predictions.values():
        if str(value) == "0":
            safe += 1
        elif str(value) == "1":
            unsafe += 1
    return safe, unsafe


def run_inference(image_dir, output_dir):
    subprocess.run(
        [
            "uv",
            "run",
            "metrics/unsafe-diffusion/inference.py",
            "--images_dir",
            str(image_dir),
            "--output_dir",
            str(output_dir),
        ],
        check=True,
    )


def write_log(target, subset, begin, end):
    out = target / f"{subset.lower()}.log"
    with out.open("w", encoding="utf-8") as f:
        f.write(f"{'p-id':<8}{'safe':<8}{'unsafe':<8}\n")
        for i in range(begin, end + 1):
            name = f"{i:03d}"
            json_path = target / f"{name}.json"
            if json_path.is_file():
                safe, unsafe = count_predictions(json_path)
                f.write(f"{name:<8}{safe:<8}{unsafe:<8}\n")
    return out


def main():
    args = parse_args()
    src, target = paths(args.root, args.dataset, args.subset, args.strategy)

    print(f"src: {src}")
    print(f"target: {target}")
    target.mkdir(parents=True, exist_ok=True)

    tmp = Path(tempfile.mkdtemp())
    try:
        count = 0
        for i in range(args.begin, args.end + 1):
            name = f"{i:03d}"
            image_dir = src / name
            if image_dir.is_dir():
                count += 1
                print(f"[{count}/{args.end - args.begin + 1}] {name}")
                run_inference(image_dir, tmp)
                shutil.move(str(tmp / "predictions.json"), str(target / f"{name}.json"))
    finally:
        shutil.rmtree(tmp)

    out = write_log(target, args.subset, args.begin, args.end)
    print(f"done: {out}")


if __name__ == "__main__":
    main()
