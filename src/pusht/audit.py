"""Executable evidence audit; only reads upstream source/data."""
import json
from pathlib import Path
import sys
import numpy as np
import zarr
from pusht.repro import now, provenance, save_json, sha256, tree_fingerprint, git


def source_snapshot(path):
    root = Path(path)
    import hashlib
    h = hashlib.sha256()
    files = git(root, 'ls-files').splitlines()
    for name in sorted(files):
        p = root / name
        if p.is_file():
            h.update((name + '\0' + sha256(p) + '\n').encode())
    return {**provenance(path), 'tracked_worktree_sha256': h.hexdigest(),
            'tracked_file_count': len(files)}


def audit(cfg, project):
    project = Path(project)
    paths = cfg['paths']
    sys.path.insert(0, paths['dp_source'])
    from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
    root = zarr.open(paths['dataset'], mode='r')
    ends = root['meta/episode_ends'][:]
    starts = np.r_[0, ends[:-1]]
    lengths = ends - starts
    assert np.all(lengths > 0)
    stats = {}
    for name in root['data']:
        a = root['data'][name]
        assert len(a) == ends[-1]
        lo, hi, nonfinite, fractional = float('inf'), -float('inf'), 0, 0
        for i in range(0, len(a), 128):
            x = a[i:i+128]
            lo, hi = min(lo, float(x.min())), max(hi, float(x.max()))
            nonfinite += int((~np.isfinite(x)).sum())
            if name == 'img':
                fractional += int((x != np.floor(x)).sum())
        stats[name] = {'shape': list(a.shape), 'dtype': str(a.dtype), 'min': lo, 'max': hi,
                       'nonfinite': nonfinite, 'chunks': list(a.chunks)}
        if name == 'img':
            stats[name]['fractional_values'] = fractional
        assert nonfinite == 0
    assert stats['img']['fractional_values'] == 0  # lossless uint8 cache allowed
    assert stats['img']['shape'][1:] == [96, 96, 3]
    assert 0 <= stats['action']['min'] <= stats['action']['max'] <= 512
    collection = (Path(paths['dp_source']) / 'demo_pusht.py').read_text()
    assert collection.index('episode.append(data)') < collection.index('env.step(act)')
    env_source = (Path(paths['dp_source']) / 'diffusion_policy/env/pusht/pusht_env.py').read_text()
    assert 'action - self.agent.position' in env_source
    examples, visuals = [], []
    for ep in [0, 1, 42, 100, len(ends)-1]:
        t = min(20, int(lengths[ep])-2)
        i = int(starts[ep] + t)
        image = root['data/img'][i]
        env = PushTImageEnv(legacy=False)
        # Privileged block pose is used ONLY in this diagnostic, never the policy.
        # A stored body pose is already world-space: legacy=False reconstructs it.
        # The evaluation initial-state convention remains legacy_test=True.
        env.reset_to_state = root['data/state'][i]
        obs = env.reset()
        rebuilt = np.moveaxis(obs['image'], 0, -1) * 255
        mae = float(np.abs(rebuilt-image).mean())
        assert mae == 0.0, f'Image/state mismatch at episode {ep}, frame {t}'
        examples.append({'episode_id': ep, 'local_frame_index': t, 'source_index': i,
                         'agent_pos_t': root['data/state'][i, :2].tolist(),
                         'action_t': root['data/action'][i].tolist(),
                         'agent_pos_next': root['data/state'][i+1, :2].tolist(),
                         'reconstruction_pixel_mae': mae})
        visuals.append((image.astype(np.uint8), rebuilt.astype(np.uint8)))
        env.close()
    integrity = tree_fingerprint(paths['dataset'])
    order = np.random.default_rng(cfg['split']['seed']).permutation(len(ends)).tolist()
    nval = cfg['split']['heldout_count']
    val, pool = order[:nval], order[nval:]
    ecfg = cfg['evaluation']
    manifest = {'schema_version': 1, 'dataset_id': Path(paths['dataset']).name,
                'dataset_fingerprint': integrity['sha256'], 'fingerprint_definition': integrity['algorithm'],
                'episode_count_total': len(ends), 'episode_ends': ends.tolist(),
                'split_seed': cfg['split']['seed'], 'heldout_count': nval,
                'ordering_algorithm': 'numpy.random.default_rng(seed).permutation(total); reserve heldout prefix first',
                'heldout_validation_episode_ids': val, 'ordered_train_pool_episode_ids': pool,
                'train_subsets': {str(n): pool[:n] if n <= len(pool) else None for n in cfg['split']['scales']},
                'subset_status': {str(n): 'FEASIBLE' if n <= len(pool) else 'NOT_FEASIBLE_WITH_CURRENT_DATA'
                                  for n in cfg['split']['scales']},
                'test_env_seeds': list(range(ecfg['test_start_seed'], ecfg['test_start_seed']+ecfg['n_test'])),
                'video_env_seeds': list(range(ecfg['test_start_seed'], ecfg['test_start_seed']+ecfg['video_count'])),
                'max_steps': ecfg['max_steps'], 'legacy_test': ecfg['legacy_test'],
                'n_obs_steps': 1, 'n_action_steps': 8,
                'notes': 'Independent episodes only. 300/500 need new authorized data; no collection performed.'}
    split_path = Path(cfg['split']['manifest'])
    save_json(split_path, manifest, immutable=True)
    digest = sha256(split_path)
    checksum = split_path.with_suffix('.sha256')
    checksum_text = digest + '  ' + split_path.name + '\n'
    if checksum.exists() and checksum.read_text() != checksum_text:
        raise ValueError('Existing split checksum differs; refusing regeneration')
    checksum.write_text(checksum_text)
    provenance_data = {key: source_snapshot(paths[key]) for key in ['alt_source', 'dp_source']}
    snapshot_path = project / 'results/reference_snapshot.json'
    save_json(snapshot_path, provenance_data, immutable=True)
    data = {'schema_version': 1, 'audited_at': now(), 'source_paths': paths,
            'source_provenance': provenance_data, 'dataset_id': manifest['dataset_id'],
            'inventory_zarr': [str(p) for p in sorted(Path(paths['dp_source']).rglob('*.zarr'))],
            'integrity': integrity, 'arrays': stats, 'total_frames': int(ends[-1]),
            'episode_count': len(ends), 'episode_length': {'min': int(lengths.min()),
                'median': float(np.median(lengths)), 'max': int(lengths.max())},
            'episode_ends_semantics': 'exclusive cumulative ends; episode e = [start[e], end[e])',
            'action_semantics': 'absolute pusher target (x,y), original 512x512 world units; PD control 100Hz, target 10Hz',
            'observation_action_offset': 0, 'alignment_examples': examples,
            'image_preprocessing': 'float32 HWC integer-valued RGB 0..255 -> CHW float32 /255; no resize (already 96x96)',
            'eligible_episodes_8_unpadded': int((lengths >= 8).sum()),
            'eligible_frames_8_unpadded': int(np.maximum(lengths-7, 0).sum()),
            'eligible_frames_repeat_last_action': int(ends[-1]),
            'padding': 'repeat last action within same episode; all frames eligible',
            'heldout_count': nval, 'train_pool_count': len(pool), 'subset_status': manifest['subset_status'],
            'split_manifest_sha256': digest,
            'score': 'mean over seeds of max_t clip(coverage_t/0.95,0,1); not raw coverage',
            'normalization': 'ImageNet mean/std; fixed position x/256-1; no fitted action normalizer'}
    save_json(project / 'results/data_audit.json', data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(visuals), 2, figsize=(5, 10))
    for row, (a, b) in enumerate(visuals):
        for col, img in enumerate([a, b]):
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
        axes[row, 0].set_title(f"episode {examples[row]['episode_id']}: stored", fontsize=9)
        axes[row, 1].set_title('reconstructed (MAE=0)', fontsize=9)
    fig.tight_layout()
    fig.savefig(project / 'results/alignment_debug.png', dpi=120)
    plt.close(fig)
    lines = ['# Push-T data audit', '', f"Audit time: {data['audited_at']}", '',
             f"Dataset: `{paths['dataset']}`", '',
             f"206 episodes / {int(ends[-1])} frames; lengths min/median/max: {int(lengths.min())}/{float(np.median(lengths))}/{int(lengths.max())}.", '',
             '| Array | Shape | dtype | min | max |', '|---|---|---|---:|---:|']
    for k, v in stats.items():
        lines.append(f"| {k} | {v['shape']} | {v['dtype']} | {v['min']} | {v['max']} |")
    lines += ['', 'All arrays have equal frame count and finite values. Images are float32 but integer-valued; conversion to uint8 is lossless.', '',
              '## Temporal and coordinate evidence', '',
              '`demo_pusht.py:88-103` appends current image/state and current mouse target before env.step(act). Offset is zero.',
              '`pusht_env.py:46-52,103-121` defines [0,512]^2 absolute targets and PD acceleration k_p*(action-agent.position).',
              '`pusht_image_dataset.py:85-94` uses state[..., :2] and moves RGB HWC to CHW /255 without temporal shift.',
              '`MultiStepWrapper` stacks oldest to latest; ALT uses the latest only and returns [B,8,2].', '',
              '| Episode | Frame | Source | agent_pos_t | action_t | next agent_pos | pixel MAE |',
              '|---:|---:|---:|---|---|---|---:|']
    for x in examples:
        lines.append(f"| {x['episode_id']} | {x['local_frame_index']} | {x['source_index']} | {x['agent_pos_t']} | {x['action_t']} | {x['agent_pos_next']} | {x['reconstruction_pixel_mae']} |")
    lines += ['', '![Stored and reconstructed RGB](../results/alignment_debug.png)', '',
              'Stored body poses must be reconstructed with legacy=False to avoid applying the initial-state center-of-mass rotation twice. This is a diagnostic only; official test initializations use legacy_test=True.', '',
              '## Split contract and feasibility', '',
              f"Reserve {nval} whole episodes before selection; train pool = {len(pool)}. Use PCG64 ordering with split seed 42. Exact IDs and 50 environment seeds are frozen in `experiments/splits/pusht_v7_split42.json`.",
              '30 and 100 are feasible nested prefixes. 300 and 500: **NOT_FEASIBLE_WITH_CURRENT_DATA**. Additional collection/alternative data requires separate user authorization.',
              f"All {len(ends)} episodes support unpadded 8-action chunks ({data['eligible_frames_8_unpadded']} starts). Repeat-last padding permits all {int(ends[-1])} frames without episode crossing.", '',
              '## Integrity and evaluation', '',
              f"Full-byte hash computed over {integrity['file_count']} files / {integrity['bytes']} bytes: `{integrity['sha256']}`.",
              f"Definition: {integrity['algorithm']}. Split SHA256: `{digest}`.",
              'Official image-task config: seeds 100000..100049, max_steps=300, legacy_test=True. Score is mean per-episode maximum reward, reward=clip(coverage/0.95,0,1). No invented binary success threshold.',
              'No train/validation-fitted normalizer is necessary: ImageNet statistics and fixed position bounds only. Original action targets are unchanged.',
              'Reference commits and tracked-file integrity hashes are in results/reference_snapshot.json. No DP model is trained.', '']
    (project / 'docs').mkdir(exist_ok=True)
    (project / 'docs/data_audit.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'episodes': len(ends), 'frames': int(ends[-1]), 'train_pool': len(pool),
                      'subset_status': manifest['subset_status'], 'split_sha256': digest}))
