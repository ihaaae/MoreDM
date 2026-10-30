"""JEV attribute extension of the frozen, plain Eq.6 comparison."""

import argparse
from pathlib import Path

from lib.flow_initializer_audit import digest, read, write
from lib.sd3_ablation import noise_proxy

BASE = Path('/mnt/raid10/luke/UnsafeDistribution/TailProbe-Audit')


def prepare(base):
    plain_path = base / 'flow-eq6-plan-001/plan.json'
    selection_path = base / 'flow-init-selection-001/plan.json'
    plain, selection = read(plain_path), read(selection_path)
    for key in ('models', 'prompts_sha256', 'cohorts_sha256'):
        assert plain[key] == selection[key]
    plan = {**plain, 'words': selection['words'], 'reuse_handsome': True,
            'scope': 'Current Eq.6, JEV attribute initialization; compare with current handsome',
            'plain_plan_sha256': digest(plain_path), 'selection_plan_sha256': digest(selection_path)}
    write(base / 'flow-eq6-jev-plan-001/plan.json', plan)
    inventory = []
    out = base / 'flow-eq6-jev-comparison-001'
    for model, ids in plan['models'].items():
        host = 'a800' if model == 'sd3' else 'h100'
        for pid in ids:
            reused = plan['words'][pid] == 'handsome'
            family = 'flow-eq6' if reused else 'flow-eq6-jev'
            for iid in range(1, 11):
                key = f'{pid}/{iid:02}.png'
                source = base / f'{family}-{host}-001' / model / key
                path = out / 'images' / model / 'J' / key
                path.parent.mkdir(parents=True, exist_ok=True)
                path.symlink_to(source)
                inventory.append({'model': model, 'key': key, 'source': str(source), 'reused': reused})
    write(out / 'inventory.json', inventory)


def check_smoke(base):
    from PIL import Image, ImageStat

    plan = read(base / 'flow-eq6-jev-plan-001/plan.json')
    for model, host in (('sd3', 'a800'), ('sd35', 'h100')):
        root = base / f'flow-eq6-jev-smoke-{host}-001'
        for shard in (0, 1):
            manifest = read(root / f'manifests/{model}-{shard}.json')
            assert manifest['status'] == 'complete' and manifest['smoke']
            assert manifest['solver_sha256'] == plan['solver_sha256']
            pid, = manifest['ids']
            path = root / model / pid / '01.png'
            record = read(path.with_suffix('.json'))
            assert record['sha256'] == digest(path) and record['init_word'] == plan['words'][pid]
            assert len(record['placeholder_drifts']) == 2 and all(v > 0 for v in record['placeholder_drifts'])
            with Image.open(path) as image:
                assert image.size == (1024, 1024) and max(ImageStat.Stat(image).stddev) > 0


def summarize(base):
    out = base / 'flow-eq6-jev-comparison-001'
    plain_root = base / 'flow-eq6-comparison-001'
    plan_path = base / 'flow-eq6-jev-plan-001/plan.json'
    plan, plain = read(plan_path), read(plain_root / 'results.json')
    assert plan['plain_plan_sha256'] == plain['plan_sha256']
    inventory = read(out / 'inventory.json')
    results, hashes, quality_values, score_hashes = {}, {}, {}, {}
    for model, ids in plan['models'].items():
        changed = [pid for pid in ids if plan['words'][pid] != 'handsome']
        host = 'a800' if model == 'sd3' else 'h100'
        for shard in (0, 1):
            m = read(base / f'flow-eq6-jev-{host}-001/manifests/{model}-{shard}.json')
            n = read(base / f'flow-eq6-{host}-001/manifests/{model}-{shard}.json')
            assert m['status'] == 'complete' and not m['smoke'] and m['ids'] == changed[shard::2]
            assert m['plan_sha256'] == digest(plan_path)
            for field in ('solver_sha256', 'source_sha256', 'gen_sha256', 'popt', 'steps', 'cfg'):
                assert m[field] == n[field], field
        directory = out / 'scores' / model
        assert read(directory / 'detector.json') == read(plain_root / f'scores/{model}/detector.json')
        labels = {f'{Path(k).parent.name}/{Path(k).name}': int(v) for k, v in read(directory / 'predictions.json').items()}
        heads = {f'{Path(k).parent.name}/{Path(k).name}': v for k, v in read(directory / 'head-scores.json').items()}
        original_labels = {f'{Path(k).parent.name}/{Path(k).name}': int(v)
                           for k, v in read(plain_root / f'scores/{model}/predictions.json').items()}
        quality = read(out / f'quality/{model}-J.json')
        assert quality['metadata'] == read(plain_root / f'quality/{model}-N.json')['metadata']
        expected = {f'{pid}/{i:02}.png' for pid in ids for i in range(1, 11)}
        assert labels.keys() == heads.keys() == original_labels.keys() == expected
        assert quality['images'].keys() == {f'J/{k}' for k in expected}
        rows = [r for r in inventory if r['model'] == model]
        assert len(rows) == len(expected) and {r['key'] for r in rows} == expected
        for row in rows:
            key, path = row['key'], Path(row['source'])
            record = read(path.with_suffix('.json'))
            checksum = digest(path)
            old_path = base / f'flow-eq6-{host}-001' / model / key
            old = read(old_path.with_suffix('.json'))
            assert digest(old_path) == plain['image_sha256'][f'{model}/N/{key}']
            assert checksum == record['sha256'] == quality['images'][f'J/{key}']['sha256']
            assert record['seed'] == old['seed'] and record['init_word'] == plan['words'][key.split('/')[0]]
            if row['reused']:
                assert path == old_path and labels[key] == original_labels[key]
            hashes[f'{model}/J/{key}'] = checksum
            quality_values[f'{model}/J/{key}'] = noise_proxy(path)
        stats = {
            'n': len(expected), 'unsafe': sum(labels.values()),
            'heads': {h: sum(v[h] > .5 for v in heads.values()) for h in next(iter(heads.values()))},
            'per_prompt': {pid: sum(labels[f'{pid}/{i:02}.png'] for i in range(1, 11)) for pid in ids},
            'clip_cosine_mean': sum(v['clip_cosine'] for v in quality['images'].values()) / len(expected),
            'noise_proxy_gt3': sum(quality_values[f'{model}/J/{k}'] > 3 for k in expected),
            'changed_subset_n': len(changed) * 10,
            'changed_subset_plain_unsafe': sum(v for k, v in original_labels.items() if k.split('/')[0] in changed),
            'changed_subset_jev_unsafe': sum(v for k, v in labels.items() if k.split('/')[0] in changed),
        }
        results[model] = {**plain['results'][model], 'J': stats}
        for path in (directory / 'predictions.json', directory / 'head-scores.json', out / f'quality/{model}-J.json'):
            score_hashes[str(path)] = digest(path)
    write(out / 'results.json', {
        'results': results, 'image_sha256': hashes, 'noise_proxy': quality_values,
        'score_sha256': score_hashes, 'plain_results_sha256': digest(plain_root / 'results.json'),
        'plan_sha256': digest(plan_path),
        'limitations': ['Same solver, parameters and sample seeds for N/J; byte-deterministic replay is not established.',
                        'JEV selects mood/style attributes; not a test of every initialization strategy.',
                        'A and historical D are reused exploratory references, not seed-paired controls.',
                        'Noise proxy >3 is an unvalidated heuristic, not a safety gate.'],
    })
    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'check-smoke', 'summarize'])
    parser.add_argument('--base', type=Path, default=BASE)
    args = parser.parse_args()
    {'prepare': prepare, 'check-smoke': check_smoke, 'summarize': summarize}[args.stage](args.base)
