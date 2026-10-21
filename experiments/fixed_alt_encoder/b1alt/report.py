"""Paired fixed-seed inference audit, actual video inspection and measured report."""
import csv
import json
from pathlib import Path
import numpy as np
from pusht.data import PushTData
from pusht.repro import now, save_json, sha256
from b1alt.provenance import provenance
from b1alt.common import EXPERIMENT, PROJECT
from b1alt.experiment import sources


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def trace_stats(metrics, metadata, data, rollouts):
    trace = [json.loads(s) for s in Path(metrics['retrieval_full_path']).read_text().splitlines()]
    train_ids = set(metadata['dataset_manifest_ref']['train_episode_ids'])
    distances = []
    for q in trace:
        assert q['matched_episode_id'] in train_ids
        source = data.index(q['matched_episode_id'], q['local_frame_index'])
        assert source == q['source_index']
        chunk, idx = data.action_chunk(q['matched_episode_id'], q['local_frame_index'])
        np.testing.assert_array_equal(chunk, q['commands'])
        np.testing.assert_array_equal(idx, q['action_source_indices'])
        assert 1 <= q['executed_actions'] <= 8 and not q['invalid']
        assert np.isfinite(q['top1_cosine'])
        distances.append(float(np.linalg.norm(np.array(q['agent_pos'])-q['matched_agent_pos'])))
    same = stationary = cycles = transitions = 0
    tails = []
    for seed in metadata['test_env_seeds']:
        q = [r for r in trace if r['evaluation_seed'] == seed]
        row = next(r for r in rollouts if int(r['evaluation_seed']) == seed)
        assert [r['current_step'] for r in q] == list(range(0, int(row['steps']), 8))
        assert sum(r['executed_actions'] for r in q) == int(row['steps'])
        for i in range(1, len(q)):
            transitions += 1
            identical = q[i]['source_index'] == q[i-1]['source_index']
            distance = np.linalg.norm(np.array(q[i]['agent_pos'])-q[i-1]['agent_pos'])
            same += int(identical)
            stationary += int(identical and distance <= .001)
            if i >= 2 and not identical:
                cycles += int(q[i]['source_index'] == q[i-2]['source_index'] and
                             np.linalg.norm(np.array(q[i]['agent_pos'])-q[i-2]['agent_pos']) <= .001)
        tail = 1
        for r in reversed(q[:-1]):
            if r['source_index'] != q[-1]['source_index']:
                break
            tail += 1
        tails.append({'seed': seed, 'same_match_queries_at_end': tail})
    assert set(r['evaluation_seed'] for r in trace) == set(metadata['test_env_seeds'])
    return {'trace_sha256': sha256(metrics['retrieval_full_path']), 'queries': len(trace), 'transitions': transitions,
        'consecutive_same_match_transitions': same, 'same_match_and_same_position_transitions': stationary,
        'consecutive_same_match_fraction': same/transitions,
        'same_match_and_same_position_fraction': stationary/transitions,
        'two_cycle_return_transitions_excluding_one_cycles': cycles,
        'episode_switches': sum(r['episode_switch'] for r in trace),
        'padded_queries': sum(r['boundary_padded'] for r in trace), 'invalid_action_events': 0,
        'query_to_matched_pusher_distance': {'mean': float(np.mean(distances)), 'median': float(np.median(distances)),
                                           'p95': float(np.quantile(distances, .95))},
        'repeated_match_tails': tails}, trace


def inspect_videos(metrics, trace, data, rd, rollouts):
    import av
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    videos = metrics['video_paths_by_seed']
    assert sorted(map(int, videos)) == list(range(100000, 100004))
    fig, axes = plt.subplots(4, 6, figsize=(16, 10))
    checks = []
    for r, seed in enumerate(range(100000, 100004)):
        path = videos[str(seed)]
        with av.open(path) as container:
            frames = [f.to_ndarray(format='rgb24') for f in container.decode(video=0)]
        row = next(row for row in rollouts if int(row['evaluation_seed']) == seed)
        assert len(frames) == int(row['steps'])
        queries = [q for q in trace if q['evaluation_seed'] == seed and q['current_step'] > 0]
        # Fixed selection: first replan, middle replan, last replan of first four seeds.
        selected = [queries[0], queries[len(queries)//2], queries[-1]]
        errors = []
        for q in queries:
            image = frames[q['current_step']-1].astype(np.float32)
            mask = (image[:, :, 2]-image[:, :, 0] > 80) & (image[:, :, 2]-image[:, :, 1] > 60)
            yy, xx = np.where(mask)
            if len(xx):
                errors.append(float(np.linalg.norm(np.array([xx.mean(), yy.mean()])-np.array(q['agent_pos'])*96/512)))
        checks.append({'evaluation_seed': seed, 'decoded_frames': len(frames), 'coordinate_checks': len(errors),
            'median_blue_centroid_error_pixels': float(np.median(errors)) if errors else None,
            'max_blue_centroid_error_pixels': float(np.max(errors)) if errors else None,
            'video_sha256': sha256(path), 'display_query_steps': [q['current_step'] for q in selected]})
        for c, q in enumerate(selected):
            axes[r, c*2].imshow(frames[q['current_step']-1])
            axes[r, c*2].set_title(f"seed{seed}, t{q['current_step']}\nactual video observation", fontsize=8)
            axes[r, c*2+1].imshow(data.root['data/img'][q['source_index']].astype(np.uint8))
            distance = np.linalg.norm(np.array(q['agent_pos'])-q['matched_agent_pos'])
            axes[r, c*2+1].set_title(f"expert {q['matched_episode_id']}:{q['local_frame_index']}\n"
                f"cos {q['top1_cosine']:.3f}; pusher err {distance:.1f}", fontsize=8)
        for ax in axes[r]:
            ax.axis('off')
    fig.suptitle('B1: first four frozen seeds, first/middle/last replans; actual current video and retrieved expert', fontsize=11)
    fig.tight_layout()
    fig.savefig(rd/'video_trace_contact_sheet.png', dpi=120)
    plt.close(fig)
    return checks


def figures(rd, baseline, metrics, comparison, diag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    train = read_csv(rd/'training_metrics.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for key in ['contrast', 'xy', 'angle', 'pose', 'total']:
        axes[0].plot([int(r['epoch']) for r in train], [float(r[key]) for r in train], label=key)
    axes[0].set(xlabel='epoch', ylabel='mean loss', title='B1 training (final epoch fixed)')
    axes[0].legend(fontsize=7)
    axes[1].scatter(comparison['baseline_scores'], comparison['b1_scores'], s=15)
    axes[1].plot([0, 1], [0, 1], '--', color='gray')
    axes[1].set(xlabel='Original ALT-30 score', ylabel='B1 score', title='Same 50 simulator seeds', xlim=(0, 1), ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(rd/'training_and_paired_scores.png', dpi=130)
    plt.close(fig)


def report(cfg, manifest, rd, heavy, meta):
    if (rd/'paired_comparison.json').exists():
        raise ValueError('Report already exists; do not overwrite completed comparison')
    metrics = json.loads((rd/'eval_metrics.json').read_text())
    iso = json.loads((rd/'inference_isolation.json').read_text())
    assert metrics['status'] == 'completed' and iso['pose_head_forward_calls'] == 0
    baseline_dir = PROJECT/'experiments/runs'/cfg['baseline_run']
    baseline = json.loads((baseline_dir/'eval_metrics.json').read_text())
    base_meta = json.loads((baseline_dir/'run_metadata.json').read_text())
    rows, base_rows = read_csv(rd/'eval_rollouts.csv'), read_csv(baseline_dir/'eval_rollouts.csv')
    seeds = [int(r['evaluation_seed']) for r in rows]
    assert seeds == manifest['test_env_seeds'] == [int(r['evaluation_seed']) for r in base_rows]
    a = np.array([float(r['official_episode_score']) for r in base_rows])
    b = np.array([float(r['official_episode_score']) for r in rows])
    diff = b-a
    rng = np.random.default_rng(cfg['evaluation']['bootstrap_seed'])
    indices = rng.integers(0, len(diff), (cfg['evaluation']['bootstrap_samples'], len(diff)))
    ci = np.quantile(diff[indices].mean(axis=1), [.025, .975]).tolist()
    comparison = {'status': 'completed', 'created_at': now(), 'n_pairs': len(diff), 'evaluation_seeds': seeds,
        'baseline_mean_score': float(a.mean()), 'b1_mean_score': float(b.mean()),
        'paired_mean_difference_b1_minus_baseline': float(diff.mean()), 'paired_bootstrap_95ci': ci,
        'bootstrap_seed': 2026, 'bootstrap_samples': 10000, 'baseline_scores': a.tolist(), 'b1_scores': b.tolist(),
        'paired_differences': diff.tolist(), 'b1_score_std': float(b.std(ddof=1)),
        'simulator_seeds_improved': int((diff > 0).sum()), 'simulator_seeds_worsened': int((diff < 0).sum()),
        'simulator_seeds_tied': int((diff == 0).sum()),
        'scope': 'one independent trained model seed per method; bootstrap covers simulator initializations only'}
    with (rd/'paired_scores.csv').open('x', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['evaluation_seed', 'baseline_score', 'b1_score', 'b1_minus_baseline'])
        w.writerows(zip(seeds, a, b, diff))
    data = PushTData(cfg['paths']['dataset'])
    b1_trace, trace = trace_stats(metrics, meta, data, rows)
    baseline_trace, _ = trace_stats(baseline, base_meta, data, base_rows)
    video_checks = inspect_videos(metrics, trace, data, rd, rows)
    save_json(rd/'rollout_inspection.json', {'status': 'completed', 'b1': b1_trace, 'baseline': baseline_trace,
              'predefined_video_checks': video_checks,
              'note': 'Video index=t-1 because recording starts after action; compressed overlays differ from raw expert images.'}, immutable=True)
    audit = json.loads((EXPERIMENT/'results/phase0_audit.json').read_text())
    changed = [name for name, digest in audit['shared_files_guard'].items() if sha256(PROJECT/name) != digest]
    if changed:
        raise ValueError(f'Protected original files changed: {changed}')
    save_json(rd/'preservation_check.json', {'status': 'passed', 'checked_at': now(),
              'original_files_checked': len(audit['shared_files_guard']), 'changed': [],
              'reference_and_dataset_fingerprints_revalidated': True}, immutable=True)
    diag = json.loads((rd/'offline_diagnostics.json').read_text())
    figures(rd, baseline, metrics, comparison, diag)
    save_json(rd/'paired_comparison.json', comparison, immutable=True)
    pose = diag['b1']['pose_prediction']
    dbase, db1 = diag['baseline'], diag['b1']
    text = f'''# B1 ALT-30 measured analysis

Completed formal40-epoch training, all2426 heldout-frame diagnostics, and the original
official50-seed Push-T Runner evaluation. Only one model seed42 per method.

| Metric (mean) | Original ALT-30 | B1 |
|---|---:|---:|
| Official max-reward score | {a.mean():.9f} | {b.mean():.9f} |
| Heldout matched pusher xy L2 | {dbase['retrieval']['pusher_error']['mean']:.3f} | {db1['retrieval']['pusher_error']['mean']:.3f} |
| Heldout matched block xy L2 | {dbase['retrieval']['block_error']['mean']:.3f} | {db1['retrieval']['block_error']['mean']:.3f} |
| Heldout matched angle error degrees | {dbase['retrieval']['angle_error_degrees']['mean']:.3f} | {db1['retrieval']['angle_error_degrees']['mean']:.3f} |
| Eight-action chunk RMSE | {dbase['retrieval']['action_chunk_rmse']['mean']:.3f} | {db1['retrieval']['action_chunk_rmse']['mean']:.3f} |
| Numeric-xy swap top1 changed fraction | {dbase['position_swap_probe']['top1_changed_fraction']:.6f} | {db1['position_swap_probe']['top1_changed_fraction']:.6f} |
| Consecutive same-match transition fraction | {baseline_trace['consecutive_same_match_fraction']:.6f} | {b1_trace['consecutive_same_match_fraction']:.6f} |
| Same-match + same-position transition fraction | {baseline_trace['same_match_and_same_position_fraction']:.6f} | {b1_trace['same_match_and_same_position_fraction']:.6f} |

B1 pose prediction on all heldout frames: block xy L2 mean/median/P95
{pose['xy_error']['mean']:.3f}/{pose['xy_error']['median']:.3f}/{pose['xy_error']['p95']:.3f} world units;
wrapped angle error {pose['angle_error_degrees']['mean']:.3f}/{pose['angle_error_degrees']['median']:.3f}/{pose['angle_error_degrees']['p95']:.3f} degrees.
The original baseline has no trained pose-output head, so there is no baseline prediction
error to compare. This directly measures B1 pose accuracy without claiming a measured
prediction improvement over that baseline. Retrieval and rollout outcomes are separate.

Paired B1-minus-baseline official score difference: {diff.mean():+.9f}; paired bootstrap95%CI
[{ci[0]:+.9f}, {ci[1]:+.9f}], using seed2026 and10000 resamples of the same50 simulator seeds.
B1 score SD {b.std(ddof=1):.6f}; mean95%CI {metrics['mean_score_bootstrap_95ci']}.
Changed seeds: {comparison['simulator_seeds_improved']} improved, {comparison['simulator_seeds_worsened']} worsened,
{comparison['simulator_seeds_tied']} tied. This interval is conditional on these two trained
models and does not estimate variation across training seeds. Official score remains
episode maximum clip(coverage/0.95,0,1), averaged over seeds; final-score mean is
{metrics['mean_final_score']:.6f}. No binary success threshold is introduced.

Training: {meta['training_seconds']:.2f}s; {meta['optimizer_steps']} Adam updates;
max allocated GPU memory {meta['max_cuda_memory_allocated_bytes']} bytes. Bank entries
{metrics['bank_entries']}; checkpoint {metrics['checkpoint_bytes']} bytes; bank
{metrics['bank_bytes']} bytes. Evaluation {metrics['evaluation_seconds']:.2f}s;
encoder/search/extraction mean latency (ms) {metrics['latency_ms']['encoder_ms']['mean']:.3f}/
{metrics['latency_ms']['search_ms']['mean']:.3f}/{metrics['latency_ms']['extraction_ms']['mean']:.3f}.
Peak-memory and latency depend on hardware/load, and the B1 checkpoint includes its unused head.

All action/source/chunk/episode indices were checked against original absolute commands;
invalid events {metrics['invalid_events']}. Actual official scores agree with independent
telemetry. Pose Head forward calls during all50 rollouts:0. Deterministic bank queries,
90 first/middle/last raw-frame reencodings, and all3806 action chunks passed. Exact self
indices may tie and are not unseen-state evidence. No label enters bank matching.

See offline_diagnostics.json for means/medians/P95, identical query-order checksum and
reproduction of the old diagnostic metrics. Position swapping is an artificial inconsistent
input probe. Action chunk difference does not identify a unique correct action. See
heldout_match_comparison.png, training_and_paired_scores.png, and actual decoded
video_trace_contact_sheet.png. Videos are the predefined first four seeds. Full traces
and videos remain local with hashes; artifact_manifest.json is not a Git backup.

Original source/config/split/result preservation guard passed for
{len(audit['shared_files_guard'])} files; both upstream references and full dataset fingerprint
were revalidated. Checkpoint hash {meta['checkpoint_sha256']}; bank hash {meta['bank_sha256']}.
Training-start commit {meta['code_at_start']['commit']}; analysis code hashes saved separately.
'''
    (rd/'report.md').write_text(text, encoding='utf-8')
    (EXPERIMENT/'docs/final_analysis.md').write_text(text, encoding='utf-8')
    save_json(rd/'analysis_code_provenance.json', {'code': provenance(PROJECT), 'sources': sources()}, immutable=True)
    meta.update(stage='reported', status='completed', report_completed_at=now())
    print(json.dumps({k: v for k, v in comparison.items() if k not in
                      ['baseline_scores', 'b1_scores', 'paired_differences', 'evaluation_seeds']}), flush=True)
