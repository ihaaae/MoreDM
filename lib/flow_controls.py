"""Matched-sampler random-v and ordinary candidate pools for SD3-family audits.

Candidate pools cover a measured TailProbe generation-time budget. They are NOT
yet scored best-of-N results: selection must charge classifier time and retain
only the affordable prefix. Random-v is a norm-matched Gaussian direction in
each CLIP embedding space, held fixed throughout sampling (zero optimizer steps).
"""

import argparse
import importlib.util
import os
from pathlib import Path
import socket
import time

from lib.flow_initializer_audit import digest, read, write


def generate(args):
    import torch
    from munch import munchify
    from torchvision.utils import save_image
    from lib.gen import get_sd3_popt_kwargs, MODEL_CACHE_DIR, sd3_model_key

    plan = read(args.plan)
    assert digest(args.solver) == plan['solver_sha256']
    assert digest(args.prompts) == plan['prompts_sha256']
    ids = plan['models'][args.model][args.shard::args.shards]
    if args.smoke:
        ids = ids[:1]
    if not ids:
        raise ValueError('Empty shard')
    spec = importlib.util.spec_from_file_location('control_solver', args.solver)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    steps, cfg = (28, 7.0) if args.model == 'sd3' else (40, 4.5)
    popt = {**get_sd3_popt_kwargs(), 'placeholder_string': '<mp>_0'}
    manifest = {'status': 'started', 'model': args.model, 'ids': ids, 'smoke': args.smoke,
                'plan_sha256': digest(args.plan), 'solver_sha256': digest(args.solver),
                'source_sha256': digest(__file__), 'popt': popt, 'steps': steps, 'cfg': cfg,
                'host': socket.gethostname(), 'gpu': torch.cuda.get_device_name(),
                'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    manifest_path = args.out / 'manifests' / f'{args.model}-{args.shard}.json'
    write(manifest_path, manifest)
    pipe = module.get_solver('flowmatch', solver_config=munchify({'num_sampling': steps}),
                            model_key=sd3_model_key(args.model), device='cuda',
                            cache_dir=str(MODEL_CACHE_DIR), reuse_sd3_text=args.model == 'sd35')
    pairs = pipe.clip_pairs
    indices = [pipe.initialize_embedding(t, e, popt) for t, e in pairs]
    norms = [e.get_input_embeddings().weight[i].detach().norm(dim=-1, keepdim=True)
             for (_, e), i in zip(pairs, indices)]
    original_initializer = pipe.initialize_embedding
    random_vectors = {}

    def initialize(tokenizer, encoder, config):
        token_ids = original_initializer(tokenizer, encoder, config)
        if random_vectors:
            with torch.no_grad():
                encoder.get_input_embeddings().weight[token_ids] = random_vectors[id(encoder)]
        return token_ids

    pipe.initialize_embedding = initialize
    prompts = args.prompts.read_text().splitlines()

    def sample(pid, iid, arm, seed, destination):
        assert not destination.exists()
        config = {**popt, 'prompt_opt': arm != 'ordinary'}
        random_vectors.clear()
        if arm == 'random-v':
            config['p_opt_iter'] = 0
            for j, ((_, encoder), token_ids, norm) in enumerate(zip(pairs, indices, norms)):
                shape = encoder.get_input_embeddings().weight[token_ids].shape
                vector = torch.randn(shape, generator=torch.Generator('cpu').manual_seed(seed + 7_000_000 + j))
                vector = vector.to(device=norm.device, dtype=norm.dtype)
                vector = vector / vector.norm(dim=-1, keepdim=True) * norm
                random_vectors[id(encoder)] = vector
        calls = {'forward': 0, 'backward': 0, 'forward_batch_elements': 0}

        def backward(_grad):
            calls['backward'] += 1

        def observe(_module, _inputs, kwargs, output):
            calls['forward'] += 1
            calls['forward_batch_elements'] += kwargs['hidden_states'].shape[0]
            tensor = output[0]
            if tensor.requires_grad:
                tensor.register_hook(backward)

        hook = pipe.transformer.register_forward_hook(observe, with_kwargs=True)
        torch.manual_seed(seed)
        torch.cuda.synchronize()
        start = time.monotonic()
        try:
            with torch.no_grad():
                decoded = pipe.sample(prompt=['', prompts[int(pid)-1]], cfg_guidance=cfg,
                                      popt_kwargs=config, generator=torch.Generator('cuda').manual_seed(seed))
            torch.cuda.synchronize()
            seconds = time.monotonic() - start
        finally:
            hook.remove()
        assert torch.isfinite(decoded).all()
        if arm == 'random-v':
            assert calls['backward'] == 0
            for (_, encoder), token_ids in zip(pairs, indices):
                assert torch.equal(encoder.get_input_embeddings().weight[token_ids].detach(), random_vectors[id(encoder)])
        if arm == 'ordinary':
            assert calls['backward'] == 0 and calls['forward'] == steps
        if arm == 'tailprobe':
            assert calls['backward'] > 0
        destination.parent.mkdir(parents=True, exist_ok=True)
        save_image(decoded, destination, normalize=True)
        record = {'model': args.model, 'prompt_id': pid, 'image_id': iid, 'arm': arm,
                  'seed': seed, 'seconds': seconds, 'calls': calls, 'sha256': digest(destination)}
        if arm == 'random-v':
            record['embedding_norms'] = [float(v.norm()) for v in random_vectors.values()]
            record['initializer'] = 'independent Gaussian directions, handsome norm, one token per CLIP encoder'
        write(destination.with_suffix('.json'), record)
        del decoded
        return record

    # Separate warmup: do not charge checkpoint loading/first-kernel overhead to
    # an image, and do not add warmup images to any evaluated candidate pool.
    sample(ids[0], 0, 'ordinary', 991337, args.out / 'warmup' / f'{args.model}-{args.shard}.png')
    for pid in ids:
        for iid in range(1, 2 if args.smoke else 11):
            seed = plan['seed_base'] + int(pid) * 1000 + iid
            old_root = Path(plan['existing_plain'][args.model])
            old = old_root / pid / f'{iid:02}.png'
            if old.is_file() and not args.smoke:
                baseline = read(old.with_suffix('.json'))
                assert digest(old) == plan['existing_image_sha256'][f'{args.model}/N/{pid}/{iid:02}.png']
                assert baseline['seed'] == seed and baseline['init_word'] == 'handsome'
                target_seconds = baseline['seconds']
                reference = {'source': str(old), 'sha256': digest(old), 'reused': True}
            else:
                target = args.out / args.model / 'tailprobe' / pid / f'{iid:02}.png'
                baseline = sample(pid, iid, 'tailprobe', seed, target)
                target_seconds = baseline['seconds']
                reference = {'source': str(target), 'sha256': baseline['sha256'], 'reused': False}
            sample(pid, iid, 'random-v', seed, args.out / args.model / 'random-v' / pid / f'{iid:02}.png')
            pool = args.out / args.model / 'ordinary' / pid / f'{iid:02}'
            elapsed, candidates = 0.0, []
            # At least one extra boundary candidate makes later detector-cost
            # accounting possible. No detector outcome is used to stop generation.
            for candidate in range(1, 65):
                candidate_seed = seed if candidate == 1 else 10_000_000_000 + int(pid) * 100_000 + iid * 100 + candidate
                row = sample(pid, iid, 'ordinary', candidate_seed, pool / f'{candidate:02}.png')
                elapsed += row['seconds']
                candidates.append(row)
                if elapsed >= target_seconds:
                    break
            else:
                raise RuntimeError('Budget not covered by 64 candidates; do not truncate silently')
            write(pool / 'budget.json', {'tailprobe': reference, 'generation_budget_seconds': target_seconds,
                                        'candidate_generation_seconds': elapsed, 'candidates': candidates,
                                        'selection_pending': True,
                                        'note': 'Not a best-of-N result until classifier costs are charged and prefix selection is evaluated.'})
            print(args.model, pid, iid, 'candidates', len(candidates), 'budget', round(target_seconds, 2), flush=True)
    manifest['status'] = 'complete'
    manifest_path.write_text(__import__('json').dumps(manifest, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--solver', type=Path, required=True)
    parser.add_argument('--model', choices=['sd3', 'sd35'], required=True)
    parser.add_argument('--prompts', type=Path, default=Path('Datasets/unsafe-diffusion/Lexica.txt'))
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=2)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        parser.error('Invalid shard')
    generate(args)
