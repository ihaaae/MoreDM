"""SD3-family initialization and solver comparisons with historical A/D reuse."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import time


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)


def select(args):
    raise RuntimeError("Legacy SD3 initializer selection is archived; see ARCHIVE.md.")


def generate(args):
    plan = read(args.plan)
    if (str(plan.get('solver_revision', '')).startswith('0a48703') or
            digest(args.solver) == 'b0933b5fbf8d38531ed2dbb8868a6fc47bc67ce89c7e06ccdca92383fb29c9b3'):
        raise RuntimeError("Legacy SD3 velocity solver is archived; see ARCHIVE.md.")

    import torch
    from munch import munchify
    from torchvision.utils import save_image
    from lib.gen import get_sd3_popt_kwargs, MODEL_CACHE_DIR, sd3_model_key

    assert digest(args.solver) == plan['solver_sha256']
    assert digest(args.prompts) == plan['prompts_sha256']
    spec = importlib.util.spec_from_file_location('audit_sd3', args.solver)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prompts = args.prompts.read_text().splitlines()
    ids = [pid for pid in plan['models'][args.model]
           if not plan.get('reuse_handsome', True) or plan['words'][pid] != 'handsome']
    ids = ids[args.shard::args.shards]
    if args.smoke:
        ids = ids[:1]
    if not ids:
        raise ValueError('Empty shard')
    steps, cfg = (28, 7.0) if args.model == 'sd3' else (40, 4.5)
    popt = {**get_sd3_popt_kwargs(), 'placeholder_string': '<mp>_0'}
    manifest = {'status': 'started', 'model': args.model, 'ids': ids,
                'plan_sha256': digest(args.plan), 'solver_sha256': digest(args.solver),
                'source_sha256': digest(__file__), 'gen_sha256': digest(Path(__file__).parent / 'gen.py'),
                'popt': popt, 'steps': steps, 'cfg': cfg,
                'host': socket.gethostname(), 'gpu': torch.cuda.get_device_name(),
                'visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'smoke': args.smoke}
    manifest_path = args.out / 'manifests' / f'{args.model}-{args.shard}.json'
    write(manifest_path, manifest)
    pipe = module.get_solver('flowmatch', solver_config=munchify({'num_sampling': steps}),
                             model_key=sd3_model_key(args.model), device='cuda',
                             cache_dir=str(MODEL_CACHE_DIR), reuse_sd3_text=args.model == 'sd35')
    # Stabilize dtype and vocabulary before encoding, as in the legacy ablation.
    pipe.text_encoder.to(torch.float32)
    pairs = pipe.clip_pairs if plan.get('encoder_layout') == 'dual' else [(pipe.tokenizer, pipe.text_encoder)]
    token_ids = [pipe.initialize_embedding(tokenizer, encoder, popt) for tokenizer, encoder in pairs]
    for pid in ids:
        word = plan['words'][pid]
        assert all(len(tokenizer.encode(word, add_special_tokens=False)) == 1 for tokenizer, _ in pairs)
        sample_config = {**popt, 'init_word': word}
        for iid in range(1, 2 if args.smoke else 11):
            path = args.out / args.model / pid / f'{iid:02}.png'
            if path.exists():
                raise FileExistsError(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            seed = plan['seed_base'] + int(pid) * 1000 + iid
            torch.manual_seed(seed)
            for tokenizer, encoder in pairs:
                pipe.initialize_embedding(tokenizer, encoder, sample_config)
            initial = [encoder.get_input_embeddings().weight[indices].detach().clone()
                       for (_, encoder), indices in zip(pairs, token_ids)]
            start = time.monotonic()
            with torch.no_grad():
                decoded = pipe.sample(prompt=['', prompts[int(pid) - 1]], cfg_guidance=cfg,
                                      popt_kwargs=sample_config,
                                      generator=torch.Generator('cuda').manual_seed(seed))
            assert torch.isfinite(decoded).all()
            drifts = [float((encoder.get_input_embeddings().weight[indices].detach() - original).norm())
                      for (_, encoder), indices, original in zip(pairs, token_ids, initial)]
            assert all(drift > 0 for drift in drifts)
            save_image(decoded, path, normalize=True)
            write(path.with_suffix('.json'), {
                'prompt_id': pid, 'image_id': iid, 'init_word': word, 'seed': seed,
                'placeholder_drift': drifts[0], 'placeholder_drifts': drifts,
                'seconds': time.monotonic() - start,
                'sha256': digest(path), 'decoded_min': float(decoded.min()),
                'decoded_max': float(decoded.max()),
            })
            print(args.model, pid, iid, word, f'{time.monotonic()-start:.1f}s', flush=True)
            del decoded
    manifest['status'] = 'complete'
    manifest_path.write_text(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['select', 'generate'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--solver', type=Path, required=True)
    parser.add_argument('--prompts', type=Path, default=Path('Datasets/unsafe-diffusion/Lexica.txt'))
    parser.add_argument('--base', type=Path)
    parser.add_argument('--tokenizer', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--model', choices=['sd3', 'sd35'])
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=2)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        parser.error('Invalid shard')
    if args.stage == 'select':
        select(args)
    else:
        generate(args)


if __name__ == '__main__':
    main()
