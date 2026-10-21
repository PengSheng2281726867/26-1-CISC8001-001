"""Offline gate -> one immutable formal run -> report. Never trains a model."""
import argparse
import csv
import io
import json
from pathlib import Path
import platform
import time
import traceback
from common import HERE, PROJECT, load, now, provenance, read, save, sha, stats


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def setup():
    import torch
    cfg, base, manifest = load()
    torch.set_num_threads(base['training']['torch_threads'])
    from geometry_bank import GeometryBank
    bank = GeometryBank(base['paths']['dataset'], manifest['train_subsets']['30'],
                        cfg['distance']['weights'], cfg['distance']['position_scale'])
    if len(bank.states) != cfg['expected_bank_frames']:
        raise ValueError('Unexpected number of training frames')
    return cfg, base, manifest, bank


def offline():
    import numpy as np
    cfg, base, manifest, bank = setup()
    from geometry_bank import wrap
    directory = HERE / 'artifacts' / 'offline_v1'
    directory.mkdir(parents=True, exist_ok=False)
    prov = provenance(cfg, base, manifest)
    bank.save(directory / 'geometry_bank.npz')
    self_indices = []
    zero_count = 0
    for i, state in enumerate(bank.states):
        j, d, _ = bank.query(state)
        if d != 0:
            raise ValueError('Self-retrieval not zero')
        self_indices.append(j)
        zero_count += d == 0
    exact = all(np.array_equal(bank.states[i], bank.data.root['data/state'][int(s)])
                for i, s in enumerate(bank.source_index))
    if not exact:
        raise ValueError('Bank state differs from source')
    rows = []
    for ep in manifest['heldout_validation_episode_ids']:
        states = bank.data.root['data/state'][bank.data.starts[ep]:bank.data.ends[ep]]
        for frame, state in enumerate(states):
            i, d, terms = bank.query(state)
            target, _ = bank.data.action_chunk(ep, frame, 8)
            matched, _ = bank.action_chunk(i)
            error = state.astype(float)-bank.states[i].astype(float)
            rows.append({'query_episode': ep, 'query_frame': frame, 'bank_index': i,
                         'matched_episode': int(bank.episode_id[i]), 'matched_frame': int(bank.frame_index[i]),
                         'pusher_error': float(np.linalg.norm(error[:2])),
                         'block_error': float(np.linalg.norm(error[2:4])),
                         'angle_error_degrees': float(abs(wrap(error[4]))*180/np.pi),
                         'action_chunk_rmse': float(np.sqrt(np.mean((target.astype(float)-matched)**2))),
                         'geometry_distance': d, 'pusher_component': terms[0],
                         'block_component': terms[1], 'angle_component': terms[2]})
    assert len(rows) == cfg['expected_validation_frames']
    write_csv(directory / 'validation_matches.csv', rows)
    result = {'status': 'passed', 'completed_at': now(), 'provenance': prov,
              'bank': {'episodes': len(bank.episodes), 'frames': len(bank.states),
                       'state_dtype': str(bank.states.dtype), 'source_equal_every_element': exact,
                       'self_query_zero_distance_count': zero_count,
                       'self_index_recovered_count': int(np.sum(np.asarray(self_indices) == np.arange(len(bank.states)))),
                       'exact_duplicate_state_rows': len(bank.states)-len(np.unique(bank.states, axis=0)),
                       'tie_rule': cfg['distance']['tie_break'],
                       'raw_angle_range_radians': [float(bank.states[:, 4].min()), float(bank.states[:, 4].max())],
                       'angle_unit_evidence': 'DP PushTEnv._get_info block_pose reads pymunk block.angle (radians); collector state uses this pose'},
              'validation_frames': len(rows),
              'retrieval': {key: stats([r[key] for r in rows]) for key in
                            ['pusher_error', 'block_error', 'angle_error_degrees', 'action_chunk_rmse', 'geometry_distance']},
              'artifacts': [{'path': str(p.relative_to(PROJECT)), 'bytes': p.stat().st_size, 'sha256': sha(p)}
                            for p in sorted(directory.iterdir()) if p.is_file()]}
    save(HERE / 'results' / 'offline_diagnostics.json', result)
    print(json.dumps({'offline': result['retrieval'], 'self': result['bank']}, indent=2))


def evaluate():
    cfg, base, manifest, bank = setup()
    gate = read(HERE / 'results' / 'offline_diagnostics.json')
    tests = read(HERE / 'results' / 'test_results.json')
    if gate['status'] != 'passed' or tests['status'] != 'passed':
        raise ValueError('Offline/tests gate not passed')
    if gate['provenance']['config_sha256'] != sha(HERE / 'config.yaml'):
        raise ValueError('Config changed after offline gate')
    for name, digest in tests['code_files'].items():
        if sha(HERE / name) != digest:
            raise ValueError(f'Code changed after tests: {name}')
    if list(bank.weights) != [1., 1., 1.]:
        raise ValueError('Frozen weights changed')
    directory = HERE / 'artifacts' / cfg['experiment_id']
    directory.mkdir(parents=True, exist_ok=False)  # never silently repeat/overwrite a formal run
    prov = provenance(cfg, base, manifest)
    import yaml
    (directory / 'resolved_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    metadata = {'status': 'running', 'started_at': now(), 'provenance': prov,
                'platform': platform.platform(), 'training_seconds': 0.,
                'policy': 'privileged 5D state; deterministic top-1; absolute 8 actions; offset0; repeat-last; no OOD'}
    save(directory / 'run_metadata.json', metadata)
    from geometry_policy import GeometryPolicy
    from oracle_runner import run_official
    try:
        with (directory / 'retrieval_full.jsonl').open('w', encoding='utf-8') as sink:
            policy = GeometryPolicy(bank, cfg['diagnostics']['repeat_position_atol'], sink)
            proxy, elapsed = run_official(policy, cfg, manifest['test_env_seeds'], directory)
        write_csv(HERE / 'results' / 'eval_rollouts.csv', proxy.records)
        import numpy as np
        scores = np.array([r['official_episode_score'] for r in proxy.records])
        rng = np.random.default_rng(cfg['diagnostics']['bootstrap_seed'])
        means = scores[rng.integers(0, 50, (cfg['diagnostics']['bootstrap_samples'], 50))].mean(axis=1)
        transitions = len(policy.query_records)-50
        metrics = {'status': 'completed', 'completed_at': now(), 'n_rollouts': 50,
                   'evaluation_seeds': manifest['test_env_seeds'], 'mean_score': float(scores.mean()),
                   'score_std': float(scores.std(ddof=1)),
                   'mean_score_bootstrap_95ci': np.quantile(means, [.025, .975]).tolist(),
                   'score_definition': 'official Runner mean(max_t clip(coverage_t/0.95,0,1))',
                   'mean_final_score': float(np.mean([r['final_score'] for r in proxy.records])),
                   'total_environment_steps': sum(r['steps'] for r in proxy.records),
                   'evaluation_seconds': elapsed, 'bank_entries': len(bank.states), 'query_count': len(policy.query_records),
                   'sync_checks': proxy.sync_checks, 'max_pusher_sync_error': proxy.max_pusher_sync_error,
                   'video_paths_by_seed': {str(k): v for k, v in proxy.video_paths.items() if v},
                   'retrieval_full_path': str(directory / 'retrieval_full.jsonl'),
                   'retrieval': {k: stats([r[k] for r in policy.query_records]) for k in
                                 ['pusher_error', 'block_error', 'angle_error_degrees', 'geometry_distance']},
                   'transitions': transitions,
                   'same_match_transitions': sum(r['same_match'] for r in policy.query_records),
                   'same_match_and_same_position_transitions': sum(r['same_match_and_same_position'] for r in policy.query_records),
                   'state_stall_transitions': sum(r['state_stall'] for r in policy.query_records),
                   'episode_switches': sum(r['episode_switch'] for r in policy.query_records),
                   'padded_queries': sum(r['boundary_padded'] for r in policy.query_records),
                   'latency_ms': {k: stats([r[k] for r in policy.query_records]) for k in ['state_read_ms', 'search_ms', 'extraction_ms']}}
        metadata.update(status='completed', completed_at=now(), metrics=metrics)
        save(directory / 'run_metadata.json', metadata)
        save(HERE / 'results' / 'run_metadata.json', metadata)
        save(HERE / 'results' / 'eval_metrics.json', metrics)
        sampled = [r for r in policy.query_records if r['evaluation_seed'] < 100004 and
                   (r['current_step'] == 0 or r['current_step'] >= 288)]
        (HERE / 'results' / 'retrieval_sample.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in sampled), encoding='utf-8')
        save(HERE / 'results' / 'artifact_manifest.json',
             [{'path': str(p.relative_to(PROJECT)), 'bytes': p.stat().st_size, 'sha256': sha(p)}
              for p in sorted(directory.rglob('*')) if p.is_file()])
        print(json.dumps(metrics, indent=2))
    except BaseException:
        metadata.update(status='failed', failed_at=now(), traceback=traceback.format_exc())
        save(directory / 'run_metadata.json', metadata)
        raise


def tests():
    import unittest
    import hashlib
    load()
    suite = unittest.defaultTestLoader.discover(str(HERE / 'tests'))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    output = stream.getvalue()
    save(HERE / 'results' / 'test_results.json',
         {'status': 'passed' if result.wasSuccessful() else 'failed', 'completed_at': now(),
          'tests': result.testsRun, 'output': output,
          'code_files': {p.name: sha(p) for p in HERE.glob('*.py')},
          'test_source_sha256': sha(HERE / 'tests' / 'test_oracle.py')})
    print(output)
    if not result.wasSuccessful():
        raise RuntimeError('Tests failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['tests', 'offline', 'evaluate', 'report', 'verify'])
    args = parser.parse_args()
    try:
        if args.stage in ['report', 'verify']:
            from reporting import report, verify
            {'report': report, 'verify': verify}[args.stage]()
        else:
            {'tests': tests, 'offline': offline, 'evaluate': evaluate}[args.stage]()
    except BaseException:
        failure = {'time': now(), 'stage': args.stage, 'traceback': traceback.format_exc()}
        log = HERE / 'artifacts' / 'failure_events.jsonl'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a', encoding='utf-8') as f:
            f.write(json.dumps(failure)+'\n')
        raise
