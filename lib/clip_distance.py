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

image = Image.open("your_image.jpg").convert("RGB")
prompts = ["a cat", "a dog", "a rocket"]

# 1) tensorize image, tokenize text
image_input = preprocess(image).unsqueeze(0).to(device)  # [1, C, H, W]
text_tokens = open_clip.tokenize(prompts).to(device)  # [N, context_len]

# 2) encode embeddings
with torch.no_grad():
    image_features = model.encode_image(image_input).float()  # [1, D]
    text_features = model.encode_text(text_tokens).float()  # [N, D]

# 3) L2-normalize
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# 4) cosine similarity
similarity = text_features @ image_features.T  # [N, 1]
# optional CLIP logits/probs style:
logits = 100.0 * image_features @ text_features.T  # [1, N]
probs = logits.softmax(dim=-1)
