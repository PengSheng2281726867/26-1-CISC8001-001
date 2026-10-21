"""Forward-only retrieval diagnostics on frozen heldout frames and saved traces.

No training, simulator rollouts, source edits or modification of prior run records.
The privileged geometric reference is an analysis comparator, never a policy.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.path.insert(0, str(PROJECT/'src'))
from pusht.repro import configure_runtime, now, provenance, save_json, sha256
configure_runtime(PROJECT)


def stats(values):
    import numpy as np
    a = np.asarray(values, dtype=np.float64)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError('Empty or nonfinite diagnostic values')
    return {'mean': float(a.mean()), 'median': float(np.median(a)),
            'p05': float(np.quantile(a, .05)), 'p95': float(np.quantile(a, .95))}


def action_chunks(data, sources):
    import numpy as np
    episodes = np.searchsorted(data.ends, sources, side='right')
    indices = np.minimum(sources[:, None]+np.arange(8), data.ends[episodes, None]-1)
    assert np.all(indices[:, 0] == sources) and np.all(indices < data.ends[episodes, None])
    return data.actions[indices]


def analyze_run(project, run_id, settings, heavy, plot_rows):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from alt.memory import MemoryBank
    from pusht.data import ObservationFrames, PushTData
    from pusht.experiment import existing_run, load_model

    cfg, split, run_dir, artifacts, metadata = existing_run(project, run_id)
    if metadata['status'] != 'completed' or sha256(artifacts/'memory_bank.pt') != metadata['bank_sha256']:
        raise ValueError('Completed, verified source run required')
    model = load_model(cfg, artifacts, metadata)
    device = next(model.parameters()).device
    data = PushTData(cfg['paths']['dataset'])
    frames = ObservationFrames(data, split['heldout_validation_episode_ids'])
    bank = MemoryBank.load(artifacts/'memory_bank.pt')
    bank.embeddings = bank.embeddings.to(device)
    assert not set(bank.episode_ids.tolist()) & set(split['heldout_validation_episode_ids'])
    all_state = data.root['data/state'][:]
    bank_state = torch.as_tensor(all_state[bank.source_indices], device=device)
    bank_chunks = action_chunks(data, bank.source_indices)
    query_chunks = action_chunks(data, frames.indices)
    results = {name: [] for name in ['fusion', 'position_nearest', 'geometry_reference']}
    similarities, all_matches = [], []
    csv_path = heavy/f'{run_id}_heldout_matches.csv'
    with csv_path.open('x', newline='', encoding='utf-8') as file, torch.inference_mode():
        writer = csv.DictWriter(file, fieldnames=['source_index', 'episode_id', 'local_frame_index',
            'method', 'matched_source_index', 'cosine_for_fusion', 'pusher_error', 'block_error',
            'angle_error_degrees', 'action_chunk_rmse', 'geometry_cost'])
        writer.writeheader()
        for start in range(0, len(frames), settings['batch_size']):
            samples = [frames[i] for i in range(start, min(start+settings['batch_size'], len(frames)))]
            image = torch.stack([s[0] for s in samples]).to(device)
            pos = torch.stack([s[1] for s in samples]).to(device)
            query_source = frames.indices[start:start+len(samples)]
            state = torch.as_tensor(all_state[query_source], device=device)
            embedding = model(image, pos)
            matched, similarity = bank.query(embedding)
            pusher_dist = torch.cdist(state[:, :2], bank_state[:, :2])
            block_dist = torch.cdist(state[:, 2:4], bank_state[:, 2:4])
            angular = torch.abs(torch.remainder(state[:, 4, None]-bank_state[None, :, 4]+np.pi, 2*np.pi)-np.pi)
            cost = ((pusher_dist/settings['position_scale_world_units'])**2
                    +(block_dist/settings['position_scale_world_units'])**2
                    +(angular/settings['angle_scale_radians'])**2)
            reference = cost.argmin(dim=1)
            methods = {'fusion': matched, 'position_nearest': pusher_dist.argmin(dim=1),
                       'geometry_reference': reference}
            similarities.extend(similarity.cpu().tolist())
            all_matches.extend(matched.cpu().tolist())
            arange = torch.arange(len(samples), device=device)
            for method, indices in methods.items():
                idx = indices.cpu().numpy()
                errors = np.sqrt(np.mean((bank_chunks[idx]-query_chunks[start:start+len(samples)])**2, axis=(1, 2)))
                for j, source in enumerate(query_source):
                    ep = int(np.searchsorted(data.ends, source, side='right'))
                    row = {'source_index': int(source), 'episode_id': ep,
                           'local_frame_index': int(source-data.starts[ep]), 'method': method,
                           'matched_source_index': int(bank.source_indices[idx[j]]),
                           'cosine_for_fusion': float(similarity[j]) if method == 'fusion' else '',
                           'pusher_error': float(pusher_dist[j, indices[j]]),
                           'block_error': float(block_dist[j, indices[j]]),
                           'angle_error_degrees': float(angular[j, indices[j]]*180/np.pi),
                           'action_chunk_rmse': float(errors[j]), 'geometry_cost': float(cost[j, indices[j]])}
                    writer.writerow(row)
                    results[method].append(row)
            assert torch.all(cost[arange, reference] <= cost[arange, matched]+1e-6)

        # Artificial interventions diagnose functional input dependence, not physical validity.
        selected = np.linspace(0, len(frames)-1, min(settings['branch_probe_frames'], len(frames))).astype(int)
        samples = [frames[i] for i in selected]
        images = torch.stack([s[0] for s in samples]).to(device)
        positions = torch.stack([s[1] for s in samples]).to(device)
        perm = torch.roll(torch.arange(len(selected), device=device), len(selected)//2)
        original = model(images, positions)
        original_idx, _ = bank.query(original)
        probes = {}
        for name, alt_image, alt_pos in [('swap_position_only', images, positions[perm]),
                                        ('swap_image_only', images[perm], positions)]:
            altered = model(alt_image, alt_pos)
            altered_idx, _ = bank.query(altered)
            probes[name] = {'embedding_one_minus_cosine': stats((1-(original*altered).sum(dim=1)).cpu()),
                            'top1_changed_fraction': float((original_idx != altered_idx).float().mean())}
        probes['frames'] = len(selected)
        probes['swapped_position_distance_world_units'] = stats((positions-positions[perm]).norm(dim=1).cpu())
        img_features = model.image_encoder((images-model.image_mean)/model.image_std)
        pos_features = model.position_encoder(positions/256-1)
        image_part = F.linear(img_features, model.fusion[0].weight[:, :img_features.shape[1]])
        position_part = F.linear(pos_features, model.fusion[0].weight[:, img_features.shape[1]:])
        probes['fusion_first_layer_centered_contribution_rms'] = {
            'image': float((image_part-image_part.mean(0)).square().sum(1).mean().sqrt()),
            'position': float((position_part-position_part.mean(0)).square().sum(1).mean().sqrt())}

        rng = np.random.default_rng(settings['random_seed'])
        first = rng.integers(len(bank.embeddings), size=settings['random_bank_pairs'])
        second = (first+rng.integers(1, len(bank.embeddings), size=len(first))) % len(bank.embeddings)
        pair_cosine = (bank.embeddings[first]*bank.embeddings[second]).sum(dim=1)
        centered = bank.embeddings-bank.embeddings.mean(dim=0)
        eigenvalues = torch.linalg.eigvalsh(centered.T@centered/len(centered)).clamp_min(0)
        p = eigenvalues/eigenvalues.sum()
        p = p[p > 0]
        spread = {'random_distinct_bank_pair_cosine': stats(pair_cosine.cpu()),
                  'bank_mean_embedding_norm': float(bank.embeddings.mean(0).norm()),
                  'centered_covariance_effective_rank': float(torch.exp(-(p*p.log()).sum())),
                  'heldout_top1_cosine': stats(similarities)}

    methods_summary = {name: {field: stats([r[field] for r in rows]) for field in
        ['pusher_error', 'block_error', 'angle_error_degrees', 'action_chunk_rmse', 'geometry_cost']}
        for name, rows in results.items()}
    for ep in split['heldout_validation_episode_ids'][:settings['display_validation_episodes']]:
        candidates = np.flatnonzero((frames.indices >= data.starts[ep]) & (frames.indices < data.ends[ep]))
        i = int(candidates[len(candidates)//2])
        source = int(frames.indices[i])
        plot_rows.append({'scale': metadata['scale'], 'episode_id': ep, 'local_frame_index': int(source-data.starts[ep]),
                          'query_image': data.root['data/img'][source].astype(np.uint8),
                          'fusion_image': data.root['data/img'][results['fusion'][i]['matched_source_index']].astype(np.uint8),
                          'reference_image': data.root['data/img'][results['geometry_reference'][i]['matched_source_index']].astype(np.uint8),
                          'fusion_errors': results['fusion'][i]})
    metrics = json.loads((run_dir/'eval_metrics.json').read_text())
    trace_path = Path(metrics['retrieval_full_path'])
    trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
    same, stationary, two_cycles, transitions = 0, 0, 0, 0
    for seed in metadata['test_env_seeds']:
        q = [row for row in trace if row['evaluation_seed'] == seed]
        for i in range(1, len(q)):
            transitions += 1
            identical = q[i]['source_index'] == q[i-1]['source_index']
            delta = np.linalg.norm(np.array(q[i]['agent_pos'])-q[i-1]['agent_pos'])
            same += int(identical)
            stationary += int(identical and delta <= settings['trace_same_position_tolerance_world_units'])
            if i >= 2 and not identical:
                two_cycles += int(q[i]['source_index'] == q[i-2]['source_index'] and
                    np.linalg.norm(np.array(q[i]['agent_pos'])-q[i-2]['agent_pos']) <= settings['trace_same_position_tolerance_world_units'])
    print(json.dumps({'run_id': run_id, 'heldout_frames': len(frames), 'branch_probes': probes,
                      'methods': methods_summary, 'embedding_spread': spread}), flush=True)
    return {'run_id': run_id, 'scale': metadata['scale'], 'heldout_frames': len(frames),
            'heldout_episode_ids': split['heldout_validation_episode_ids'],
            'checkpoint_sha256': metadata['checkpoint_sha256'], 'bank_sha256': metadata['bank_sha256'],
            'split_manifest_sha256': metadata['split_manifest_sha256'],
            'methods': methods_summary, 'branch_probes': probes, 'embedding_spread': spread,
            'saved_trace': {'sha256': sha256(trace_path), 'queries':len(trace), 'transitions':transitions,
                'consecutive_same_match_transitions':same, 'same_match_and_same_position_transitions':stationary,
                'two_cycle_return_transitions_excluding_one_cycles':two_cycles},
            'full_details': {'path':csv_path.relative_to(project).as_posix(), 'bytes':csv_path.stat().st_size,
                             'sha256':sha256(csv_path)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/diagnostics/retrieval_failure_v1.json')
    args = parser.parse_args()
    config_path = Path(args.config)
    settings = json.loads(config_path.read_text())
    analysis_id = settings['analysis_id']
    if not analysis_id or any(x in analysis_id for x in ['/', '\\', '..']):
        raise ValueError('Invalid analysis ID')
    output = PROJECT/'results/diagnostics'/f'{analysis_id}.json'
    heavy = PROJECT/'artifacts/diagnostics'/analysis_id
    if output.exists() or heavy.exists():
        raise ValueError('Analysis already exists; use a new ID/config rather than overwrite')
    output.parent.mkdir(parents=True, exist_ok=True)
    heavy.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot_rows = []
    result = {'status':'running', 'started_at':now(), 'resolved_analysis_config':settings,
              'analysis_config_sha256':sha256(config_path), 'analysis_script_sha256':sha256(__file__),
              'code_at_start':provenance(PROJECT)}
    try:
        result['runs'] = [analyze_run(PROJECT, run, settings, heavy, plot_rows) for run in settings['run_ids']]
        fig, axes = plt.subplots(len(plot_rows), 3, figsize=(8, 2.4*len(plot_rows)), squeeze=False)
        for i, row in enumerate(plot_rows):
            e = row['fusion_errors']
            for col, key in enumerate(['query_image', 'fusion_image', 'reference_image']):
                axes[i, col].imshow(row[key])
                axes[i, col].axis('off')
            axes[i, 0].set_title(f"ALT-{row['scale']}: heldout {row['episode_id']}:{row['local_frame_index']}", fontsize=8)
            axes[i, 1].set_title(f"Fusion: pusher err {e['pusher_error']:.1f}, T err {e['block_error']:.1f}\nangle err {e['angle_error_degrees']:.1f} deg", fontsize=8)
            axes[i, 2].set_title('Privileged geometric reference (diagnostic only)', fontsize=8)
        fig.tight_layout()
        fig.savefig(output.with_suffix('.png'), dpi=120)
        plt.close(fig)
        result.update(status='completed', completed_at=now())
    except Exception:
        result.update(status='failed', error=traceback.format_exc(), completed_at=now())
        with (PROJECT/'docs/failure_log.md').open('a', encoding='utf-8') as f:
            f.write('\n## '+now()+' representation diagnostic failure\n\nCommand: `'+
                        ' '.join(sys.argv)+'`\n\n```text\n'+traceback.format_exc()+'```\n')
        raise
    finally:
        save_json(output, result, immutable=True)


if __name__ == '__main__':
    main()
