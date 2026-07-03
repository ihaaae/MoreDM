import argparse
from pathlib import Path

import open_clip
import torch
from PIL import Image


CLIP_CACHE_DIR = "/home/lxc/MoreDM/Models/clip/hub"
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess, _ = open_clip.create_model_and_transforms(
    "ViT-L-14",
    "openai",
    cache_dir=CLIP_CACHE_DIR,
)
model = model.to(device)
model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CLIP cosine similarity for one prompt.")
    parser.add_argument("--prompt", type=str, required=True, help="Single text prompt.")
    parser.add_argument("--image", type=str, required=True, help="Image file path.")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Not found or not a file: {args.image}")

    image = Image.open(image_path).convert("RGB")
    prompts = [args.prompt]

    # 1) tensorize image, tokenize text
    image_input = preprocess(image).unsqueeze(0).to(device)  # [1, C, H, W]
    text_tokens = open_clip.tokenize(prompts).to(device)  # [1, context_len]

    # 2) encode embeddings
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()  # [1, D]
        text_features = model.encode_text(text_tokens).float()  # [1, D]

    # 3) L2-normalize
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # 4) cosine similarity
    similarity = text_features @ image_features.T  # [1, 1]
    score = float(similarity.item())
    print(f"{image_path}\t{score:.6f}")


if __name__ == "__main__":
    main()
