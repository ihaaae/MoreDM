import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UD_MODULE = ROOT / "modules" / "unsafe-diffusion"
MH_CHECKPOINTS = UD_MODULE / "checkpoints" / "multi-headed"
CLIP_CACHE_DIR = "/home/lxc/MoreDM/Models/clip/hub"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated images with unsafe-diffusion.")
    parser.add_argument("--dataset")
    parser.add_argument("--subset")
    parser.add_argument("--strategy", choices=["Vanilla", "Minority"])
    parser.add_argument("--root", default="/home/lxc/MoreDM")
    parser.add_argument("--begin", type=int, default=1)
    parser.add_argument("--end", type=int, default=50)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-log", action="store_true", help="Do not rebuild the safety log after evaluation.")
    parser.add_argument(
        "--rebuild-log",
        action="store_true",
        help="Only rebuild the safety log from existing JSON predictions.",
    )
    parser.add_argument("--images_dir", help="Evaluate one image directory directly.")
    parser.add_argument("--output_dir", help="Directory for direct predictions.json output.")
    parser.add_argument("--checkpoints", default=str(MH_CHECKPOINTS))
    args = parser.parse_args()

    direct = args.images_dir or args.output_dir
    if direct:
        if not args.images_dir or not args.output_dir:
            parser.error("--images_dir and --output_dir must be used together")
    elif not args.dataset or not args.subset or not args.strategy:
        parser.error("--dataset, --subset, and --strategy are required unless using --images_dir")

    return args


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


def run_inference(image_dir, output_dir, checkpoints=MH_CHECKPOINTS):
    sys.path.insert(0, str(UD_MODULE))

    import inference
    import numpy as np
    import torch

    inference.CLIP_CACHE_DIR = CLIP_CACHE_DIR

    class RecursiveImageDataset(torch.utils.data.Dataset):
        def __init__(self, images_dir):
            root = Path(images_dir)
            self.paths = sorted(
                str(path)
                for path in root.rglob("*.png")
                if "record" not in path.relative_to(root).parts
            )

        def __getitem__(self, idx):
            return self.paths[idx]

        def __len__(self):
            return len(self.paths)

    dataset = RecursiveImageDataset(images_dir=image_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=50, drop_last=False, shuffle=False)
    result = inference.multiheaded_check(loader=loader, checkpoints=str(checkpoints))

    head_predictions = np.array([result[head] for head in inference.unsafe_contents])
    preds = np.int16(np.sum(head_predictions, axis=0) > 0)
    final_result = {item: str(preds[i]) for i, item in enumerate(dataset)}

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    with (output_path / "predictions.json").open("w", encoding="utf-8") as f:
        json.dump(final_result, f)


def write_log(target, subset, begin, end):
    out = target / f"{subset.lower()}.log"
    with out.open("w", encoding="utf-8") as f:
        f.write(f"{'p-id':<8}{'safe':<8}{'unsafe':<8}\n")
        for i in range(begin, end + 1):
            name, json_path = prediction_path(target, i)
            if json_path.is_file():
                safe, unsafe = count_predictions(json_path)
                f.write(f"{name:<8}{safe:<8}{unsafe:<8}\n")
    return out


def prediction_path(target, prompt_id):
    name = f"{prompt_id:03d}"
    json_path = target / f"{name}.json"
    if json_path.is_file():
        return name, json_path

    legacy_name = f"{prompt_id:02d}"
    legacy_json_path = target / f"{legacy_name}.json"
    if legacy_json_path.is_file():
        return legacy_name, legacy_json_path

    return name, json_path


def main():
    args = parse_args()

    if args.images_dir:
        run_inference(args.images_dir, args.output_dir, args.checkpoints)
        return

    src, target = paths(args.root, args.dataset, args.subset, args.strategy)

    print(f"src: {src}")
    print(f"target: {target}")
    target.mkdir(parents=True, exist_ok=True)

    if args.rebuild_log:
        out = write_log(target, args.subset, args.begin, args.end)
        print(f"done: {out}")
        return

    tmp = Path(tempfile.mkdtemp())
    try:
        count = 0
        for i in range(args.begin, args.end + 1):
            name = f"{i:03d}"
            image_dir = src / name
            _, existing_json_path = prediction_path(target, i)
            if args.skip_existing and existing_json_path.is_file():
                continue
            if image_dir.is_dir():
                count += 1
                print(f"[{count}/{args.end - args.begin + 1}] {name}")
                run_inference(image_dir, tmp)
                shutil.move(str(tmp / "predictions.json"), str(target / f"{name}.json"))
    finally:
        shutil.rmtree(tmp)

    if not args.no_log:
        out = write_log(target, args.subset, args.begin, args.end)
        print(f"done: {out}")


if __name__ == "__main__":
    main()
