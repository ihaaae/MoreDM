import argparse
import hashlib
import json
from pathlib import Path

import open_clip
from PIL import Image
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--arms', nargs='+', required=True)
args = parser.parse_args()
out = args.root / 'quality' / f'{args.model}-{"-".join(args.arms)}.json'
assert not out.exists()
prompts = Path('Datasets/unsafe-diffusion/Lexica.txt').read_text().splitlines()
model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', 'openai', force_quick_gelu=True)
model = model.to('cuda').eval()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
rows = {}
with torch.inference_mode():
    for arm in args.arms:
        paths = sorted((args.root / 'images' / args.model / arm).glob('*/*.png'))
        for start in range(0, len(paths), 32):
            batch = paths[start:start + 32]
            tensors = []
            for p in batch:
                with Image.open(p) as im:
                    tensors.append(preprocess(im.convert('RGB')))
            f = model.encode_image(torch.stack(tensors).to('cuda')).float()
            t = model.encode_text(open_clip.tokenize([prompts[int(p.parent.name)-1] for p in batch]).to('cuda')).float()
            values = ((f / f.norm(dim=-1, keepdim=True)) * (t / t.norm(dim=-1, keepdim=True))).sum(-1).cpu().tolist()
            for p, v in zip(batch, values):
                rows[f'{arm}/{p.parent.name}/{p.name}'] = {'clip_cosine': v, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
out.parent.mkdir(parents=True, exist_ok=True)
with out.open('x') as stream:
    json.dump({'metadata': {'clip': 'ViT-L-14/openai/QuickGELU/eval', 'truncated_text': True,
                            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, 'images': rows}, stream, indent=2)
print(args.model, args.arms, len(rows))
