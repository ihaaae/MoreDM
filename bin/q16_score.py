"""Score a single image with the Q16 classifier.

Outputs one line to stdout:
    <image_path>\t<inappropriate_prob>

Uses ViT-L/14 backbone with prompt-tuned embeddings.
"""

import argparse
import pickle
import sys
from pathlib import Path

import clip
import torch
from PIL import Image

Q16_ROOT = Path(__file__).resolve().parent.parent / "metrics" / "Q16"
PROMPT_PATH = Q16_ROOT / "data" / "ViT-L-14" / "prompts.p"

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-L/14", device=device, jit=False)
model.eval()

prompts = torch.HalfTensor(pickle.load(open(PROMPT_PATH, "rb"))).to(device)


def score_image(image_path: str) -> float:
    """Return P(inappropriate) for a single image."""
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features = prompts.float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T)
        probs = similarity.softmax(dim=-1)
    return probs[0, 1].item()  # index 1 = inappropriate


def main() -> None:
    parser = argparse.ArgumentParser(description="Q16 single-image unsafe score.")
    parser.add_argument("--image", type=str, required=True, help="Image file path.")
    args = parser.parse_args()

    path = Path(args.image)
    if not path.is_file():
        print(f"Not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    prob = score_image(args.image)
    print(f"{path}\t{prob:.6f}")


if __name__ == "__main__":
    main()
