"""Same heldout queries/metric definitions as the baseline, with B1 pose errors."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from alt.encoder import make_augmentation
from pusht.data import PushTData, ObservationFrames
from pusht.experiment import event
from pusht.repro import configure_runtime, now, save_json, sha256
from b1alt.common import EXPERIMENT, PROJECT
from b1alt.data_adapter import PoseFrames
from b1alt.experiment import load_model, load_bank
from b1alt.loss import losses

# Reuse exact baseline diagnostic definitions without invoking its main/run writer.
sys.path.insert(0, str(PROJECT/'scripts'))
from diagnose_representation import stats, action_chunks
configure_runtime(EXPERIMENT)


def bank_validation(model, bank, data, selected):
    device = next(model.parameters()).device
    bank.embeddings = bank.embeddings.to(device)
    frames = ObservationFrames(data, selected)
    np.testing.assert_array_equal(frames.indices, bank.source_indices)
    assert set(bank.episode_ids.tolist()) == set(selected)
    exact = equivalent = 0
    minimum = 1.
    samples = []
    with torch.inference_mode():
        for start in range(0, len(bank.embeddings), 128):
            expected = torch.arange(start, min(start+128, len(bank.embeddings)), device=device)
            match, sim = bank.query(bank.embeddings[expected])
            exact += int((match == expected).sum())
            equivalent += int(((bank.embeddings[match]-bank.embeddings[expected]).norm(dim=-1) < 1e-3).sum())
            minimum = min(minimum, float(sim.min()))
        offsets = []
        for ep in selected:
            local = np.flatnonzero(bank.episode_ids == ep)
            offsets.extend([int(local[0]), int(local[len(local)//2]), int(local[-1])])
        for start in range(0, len(offsets), 64):
            chosen = offsets[start:start+64]
            observations = [frames[i] for i in chosen]
            z = model(torch.stack([s[0] for s in observations]).to(device),
                      torch.stack([s[1] for s in observations]).to(device))
            match, sim = bank.query(z)
            for j, i in enumerate(chosen):
                distance = float((z[j]-bank.embeddings[i]).norm())
                if distance > 1e-4 or float(sim[j]) < .99999:
                    raise ValueError('Deterministic raw-frame bank reencoding mismatch')
                samples.append({'bank_index': i, 'retrieved_index': int(match[j]), 'l2_error': distance})
    for ep, local, source in zip(bank.episode_ids, bank.local_frames, bank.source_indices):
        chunk, idx = data.action_chunk(ep, local)
        np.testing.assert_array_equal(chunk, data.actions[idx])
        assert idx[0] == source and idx[-1] < data.ends[ep]
        assert chunk.shape == (8, 2) and np.isfinite(chunk).all()
    if equivalent != len(frames) or minimum < .99999:
        raise ValueError('Bank self-retrieval failed')
    return {'status': 'passed', 'bank_entries': len(frames), 'self_queries': len(frames),
            'exact_self_index_count': exact, 'equivalent_embedding_count': equivalent,
            'min_self_cosine': minimum, 'raw_frame_reencoding_samples': samples,
            'action_chunks_checked': len(frames),
            'ties': 'Duplicate/equivalent embeddings or float ties may return the first equal entry; not unseen-state accuracy.'}


@torch.inference_mode()
def analyze_model(model, bank, data, episodes, heavy, label, settings, pose=False):
    device = next(model.parameters()).device
    bank.embeddings = bank.embeddings.to(device)
    frames = ObservationFrames(data, episodes)
    assert not set(bank.episode_ids.tolist()) & set(episodes)
    all_state = data.root['data/state'][:]
    bank_state = torch.as_tensor(all_state[bank.source_indices], device=device)
    bank_chunks = action_chunks(data, bank.source_indices)
    query_chunks = action_chunks(data, frames.indices)
    rows, pose_rows = [], []
    for start in range(0, len(frames), settings['batch_size']):
        batch = [frames[i] for i in range(start, min(start+settings['batch_size'], len(frames)))]
        images = torch.stack([s[0] for s in batch]).to(device)
        pos = torch.stack([s[1] for s in batch]).to(device)
        sources = frames.indices[start:start+len(batch)]
        state = torch.as_tensor(all_state[sources], device=device)
        if pose:
            z, pred = model.forward_with_pose(images, pos)
            predicted_xy = (pred[:, :2]+1)*256
            predicted_theta = torch.atan2(pred[:, 2], pred[:, 3])
            xy_error = (predicted_xy-state[:, 2:4]).norm(dim=-1)
            angle_error = torch.abs(torch.remainder(predicted_theta-state[:, 4]+np.pi, 2*np.pi)-np.pi)*180/np.pi
            for j, source in enumerate(sources):
                pose_rows.append({'source_index': int(source), 'xy_error': float(xy_error[j]),
                    'angle_error_degrees': float(angle_error[j]),
                    'pred_x': float(predicted_xy[j, 0]), 'pred_y': float(predicted_xy[j, 1]),
                    'pred_theta_radians': float(predicted_theta[j])})
        else:
            z = model(images, pos)
        match, sim = bank.query(z)
        # Match the original torch.cdist/periodic-angle definitions, including precision.
        pusher_dist = torch.cdist(state[:, :2], bank_state[:, :2])
        block_dist = torch.cdist(state[:, 2:4], bank_state[:, 2:4])
        angular = torch.abs(torch.remainder(state[:, 4, None]-bank_state[None, :, 4]+np.pi, 2*np.pi)-np.pi)
        matched = match.cpu().numpy()
        errors = np.sqrt(np.mean((bank_chunks[matched]-query_chunks[start:start+len(batch)])**2, axis=(1, 2)))
        for j, source in enumerate(sources):
            ep = int(np.searchsorted(data.ends, source, side='right'))
            rows.append({'source_index': int(source), 'episode_id': ep, 'local_frame_index': int(source-data.starts[ep]),
                'matched_source_index': int(bank.source_indices[matched[j]]),
                'cosine_for_fusion': float(sim[j]), 'pusher_error': float(pusher_dist[j, match[j]]),
                'block_error': float(block_dist[j, match[j]]), 'angle_error_degrees': float(angular[j, match[j]]*180/np.pi),
                'action_chunk_rmse': float(errors[j])})
    csv_path = heavy/f'{label}_heldout_matches.csv'
    write_csv(csv_path, rows)
    if pose:
        write_csv(heavy/f'{label}_heldout_pose.csv', pose_rows)
    selected = np.linspace(0, len(frames)-1, min(settings['branch_probe_frames'], len(frames))).astype(int)
    batch = [frames[i] for i in selected]
    images = torch.stack([s[0] for s in batch]).to(device)
    pos = torch.stack([s[1] for s in batch]).to(device)
    perm = torch.roll(torch.arange(len(selected), device=device), len(selected)//2)
    original = model(images, pos)
    idx, _ = bank.query(original)
    altered = model(images, pos[perm])
    changed, _ = bank.query(altered)
    result = {'heldout_frames': len(frames), 'heldout_episode_ids': episodes,
        'retrieval': {key: stats([r[key] for r in rows]) for key in
                      ['pusher_error', 'block_error', 'angle_error_degrees', 'action_chunk_rmse']},
        'position_swap_probe': {'frames': len(selected), 'top1_changed_fraction': float((idx != changed).float().mean()),
            'embedding_one_minus_cosine': stats((1-(original*altered).sum(dim=1)).cpu()),
            'interpretation': 'Artificial inconsistent input probe, not a physically realizable intervention'},
        'full_matches': {'path': str(csv_path), 'sha256': sha256(csv_path), 'bytes': csv_path.stat().st_size}}
    if pose:
        result['pose_prediction'] = {key: stats([r[key] for r in pose_rows]) for key in ['xy_error', 'angle_error_degrees']}
    return result, rows


def write_csv(path, rows):
    with Path(path).open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_matches(data, manifest, base_rows, b1_rows, rd):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected = []
    for ep in manifest['heldout_validation_episode_ids'][:4]:
        candidates = [i for i, row in enumerate(base_rows) if row['episode_id'] == ep]
        selected.append(candidates[len(candidates)//2])
    fig, axes = plt.subplots(4, 3, figsize=(13, 12))
    for r, i in enumerate(selected):
        source = base_rows[i]['source_index']
        assert source == b1_rows[i]['source_index']
        state = data.root['data/state'][source]
        axes[r, 0].imshow(data.root['data/img'][source].astype(np.uint8))
        axes[r, 0].set_title(f"Heldout {base_rows[i]['episode_id']}:{base_rows[i]['local_frame_index']}\n"
            f"pusher ({state[0]:.1f},{state[1]:.1f}); T ({state[2]:.1f},{state[3]:.1f},{state[4]*180/np.pi:.1f}deg)", fontsize=8)
        for col, name, rows in [(1, 'Original ALT-30', base_rows), (2, 'B1', b1_rows)]:
            row = rows[i]
            matched = data.root['data/state'][row['matched_source_index']]
            axes[r, col].imshow(data.root['data/img'][row['matched_source_index']].astype(np.uint8))
            axes[r, col].set_title(f"{name} cosine {row['cosine_for_fusion']:.4f}\n"
                f"pusher ({matched[0]:.1f},{matched[1]:.1f}); T ({matched[2]:.1f},{matched[3]:.1f},{matched[4]*180/np.pi:.1f}deg)\n"
                f"errors: pusher {row['pusher_error']:.1f}, Txy {row['block_error']:.1f}, angle {row['angle_error_degrees']:.1f}deg", fontsize=8)
        for ax in axes[r]:
            ax.axis('off')
    fig.tight_layout()
    fig.savefig(rd/'heldout_match_comparison.png', dpi=120)
    plt.close(fig)
    save_json(rd/'display_selection.json', {'rule': 'middle frame of first four frozen validation episodes',
              'source_indices': [base_rows[i]['source_index'] for i in selected]}, immutable=True)


def diagnose(cfg, manifest, rd, heavy, meta):
    if (rd/'offline_diagnostics.json').exists():
        raise ValueError('Diagnostics already exist')
    if meta['training_status'] != 'completed':
        raise ValueError('Completed formal training required')
    data = PushTData(cfg['paths']['dataset'])
    model, bank = load_model(cfg, heavy, meta), load_bank(heavy, meta)
    validation = bank_validation(model, bank, data, manifest['train_subsets']['30'])
    validation.update(checkpoint_sha256=meta['checkpoint_sha256'], bank_sha256=meta['bank_sha256'])
    save_json(rd/'bank_validation.json', validation, immutable=True)
    b1, b1_rows = analyze_model(model, bank, data, manifest['heldout_validation_episode_ids'],
                               heavy, 'b1', cfg['diagnostics'], pose=True)
    # Final-epoch two-view diagnostic; never used for selection or parameter updates.
    val_frames = PoseFrames(data, manifest['heldout_validation_episode_ids'], make_augmentation(cfg['augmentation']))
    sums = dict.fromkeys(['contrast', 'xy', 'angle', 'pose', 'total'], 0.)
    batches = 0
    model.eval()
    with torch.random.fork_rng(), torch.inference_mode():
        torch.manual_seed(45)
        for batch in DataLoader(val_frames, batch_size=32, shuffle=False, drop_last=True):
            a, b, pos, target = [x.to(next(model.parameters()).device) for x in batch]
            z1, p1 = model.forward_with_pose(a, pos)
            z2, p2 = model.forward_with_pose(b, pos)
            ls = losses(z1, z2, p1, p2, target)
            for k in sums:
                sums[k] += ls[k].item()
            batches += 1
    save_json(rd/'validation_metrics.json', {'final_epoch': 40, 'batches': batches,
        'augmented_two_view_losses': {k: v/batches for k, v in sums.items()},
        'deterministic_pose_prediction': b1['pose_prediction'], 'selection_uses_validation': False}, immutable=True)
    from pusht.experiment import existing_run, load_model as baseline_model
    from alt.memory import MemoryBank
    base_cfg, base_manifest, base_dir, base_heavy, base_meta = existing_run(PROJECT, cfg['baseline_run'])
    assert base_manifest == manifest
    if sha256(base_heavy/'memory_bank.pt') != base_meta['bank_sha256']:
        raise ValueError('Original baseline bank changed')
    base, base_rows = analyze_model(baseline_model(base_cfg, base_heavy, base_meta),
        MemoryBank.load(base_heavy/'memory_bank.pt'), data, manifest['heldout_validation_episode_ids'],
        heavy, 'baseline', cfg['diagnostics'])
    prior = json.loads((PROJECT/'results/diagnostics/retrieval_failure_v1.json').read_text())
    original = next(r for r in prior['runs'] if r['run_id'] == cfg['baseline_run'])
    for k, values in base['retrieval'].items():
        for stat in ['mean', 'median', 'p95']:
            if not np.isclose(values[stat], original['methods']['fusion'][k][stat], atol=1e-5, rtol=1e-6):
                raise ValueError(f'Original offline metric reproduction differs: {k}/{stat}')
    plot_matches(data, manifest, base_rows, b1_rows, rd)
    output = {'status': 'completed', 'completed_at': now(), 'split_manifest_sha256': meta['split_manifest_sha256'],
        'checkpoint_sha256': meta['checkpoint_sha256'], 'bank_sha256': meta['bank_sha256'],
        'diagnostic_helper_sha256': sha256(PROJECT/'scripts/diagnose_representation.py'),
        'query_order_sha256': __import__('hashlib').sha256(np.asarray([r['source_index'] for r in b1_rows], dtype='<i8').tobytes()).hexdigest(),
        'settings': cfg['diagnostics'], 'baseline_reproduces_saved_diagnostics': True, 'baseline': base, 'b1': b1,
        'limitations': 'Pose errors are predictive diagnostics. Baseline has no trained pose head; no baseline pose-accuracy comparison. Retrieval action RMSE is descriptive, not accuracy. One model seed.'}
    save_json(rd/'offline_diagnostics.json', output, immutable=True)
    meta.update(stage='offline_diagnosed', status='running', offline_diagnostics_sha256=sha256(rd/'offline_diagnostics.json'))
    event(rd, 'offline_diagnostics', 'completed', frames=b1['heldout_frames'])
    print(json.dumps({'pose': b1['pose_prediction'], 'baseline_retrieval': base['retrieval'], 'b1_retrieval': b1['retrieval'],
                      'position_top1_changed_fraction': b1['position_swap_probe']['top1_changed_fraction']}), flush=True)
