"""Run a fixed-seed action-replanning-horizon ablation for the geometry oracle.

The existing formal run stays immutable. This script compares 1/3/5/7 actions
per retrieval query and reruns 8 actions as a control using identical seeds,
bank, environment and official Runner scoring.
"""
import argparse
import csv
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from common import HERE, load, now, provenance, read, save, sha, stats
from run import setup


HORIZONS = (1, 3, 5, 7, 8)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_one(horizon, cfg, manifest, bank, artifact_dir, indices, run_provenance):
    from geometry_policy import GeometryPolicy
    from oracle_runner import run_official

    artifact_dir.mkdir(parents=True, exist_ok=False)
    resolved = {
        'experiment_id': 'geometry_oracle_alt030_action_horizon_ablation_v1',
        'action_horizon': horizon,
        'interpretation': 'number of absolute target commands executed before the next memory-bank query',
        'policy_input': 'privileged current [pusher_x, pusher_y, block_x, block_y, block_theta]',
        'retrieval': 'deterministic top-1 geometry distance; frozen 1:1:1 weights; no OOD fallback',
        'action': {'alignment_offset': 0, 'units': 'original absolute pusher target position',
                   'chunk_length': horizon, 'episode_boundary': 'repeat last action within same episode'},
        'environment': {'n_obs_steps': 1, 'n_action_steps': horizon,
                        'max_steps': cfg['evaluation']['max_steps'], 'legacy_test': cfg['evaluation']['legacy_test'],
                        'n_envs': 1, 'fps': cfg['evaluation']['fps']},
        'evaluation_seeds': manifest['test_env_seeds'],
        'source_formal_config_sha256': sha(HERE / 'config.yaml'),
        'source_formal_metrics_sha256': sha(HERE / 'results' / 'eval_metrics.json'),
        'split_manifest_sha256': cfg['split_sha256'],
    }
    import yaml
    (artifact_dir / 'resolved_config.yaml').write_text(yaml.safe_dump(resolved, sort_keys=False), encoding='utf-8')
    metadata = {'status': 'running', 'started_at': now(), 'action_horizon': horizon,
                'provenance': run_provenance, 'resolved_config': resolved,
                'training_seconds': 0.0}
    save(artifact_dir / 'run_metadata.json', metadata)

    try:
        with (artifact_dir / 'retrieval_full.jsonl').open('w', encoding='utf-8') as sink:
            policy = GeometryPolicy(bank, cfg['diagnostics']['repeat_position_atol'], sink,
                                    action_steps=horizon)
            proxy, elapsed = run_official(policy, cfg, manifest['test_env_seeds'], artifact_dir,
                                          action_steps=horizon)

        query_rows = policy.query_records
        if len(proxy.records) != len(manifest['test_env_seeds']):
            raise ValueError('Runner did not complete every frozen evaluation seed')
        if any(row.get('executed_actions', 0) > horizon for row in query_rows):
            raise ValueError('A retrieval query executed more than the requested horizon')
        scores = np.asarray([row['official_episode_score'] for row in proxy.records], dtype=float)
        bootstrap_indices = indices
        means = scores[bootstrap_indices].mean(axis=1)
        metrics = {
            'status': 'completed', 'completed_at': now(), 'action_horizon': horizon,
            'n_rollouts': len(proxy.records), 'evaluation_seeds': manifest['test_env_seeds'],
            'mean_score': float(scores.mean()), 'score_std': float(scores.std(ddof=1)),
            'mean_score_bootstrap_95ci': np.quantile(means, [.025, .975]).tolist(),
            'score_definition': 'official Runner mean(max_t clip(coverage_t/0.95,0,1))',
            'mean_final_score': float(np.mean([row['final_score'] for row in proxy.records])),
            'total_environment_steps': int(sum(row['steps'] for row in proxy.records)),
            'evaluation_seconds': elapsed, 'bank_entries': len(bank.states),
            'query_count': len(query_rows), 'mean_queries_per_rollout': len(query_rows) / len(proxy.records),
            'executed_actions_per_query': stats([row['executed_actions'] for row in query_rows]),
            'sync_checks': proxy.sync_checks, 'max_pusher_sync_error': proxy.max_pusher_sync_error,
            'episode_switches': int(sum(row['episode_switch'] for row in query_rows)),
            'padded_queries': int(sum(row['boundary_padded'] for row in query_rows)),
            'video_paths_by_seed': {str(k): v for k, v in proxy.video_paths.items() if v},
            'retrieval_full_path': str(artifact_dir / 'retrieval_full.jsonl'),
        }
        write_csv(artifact_dir / 'eval_rollouts.csv', proxy.records)
        metadata.update(status='completed', completed_at=now(), metrics=metrics)
        save(artifact_dir / 'run_metadata.json', metadata)
        save(artifact_dir / 'eval_metrics.json', metrics)
        save(artifact_dir / 'artifact_manifest.json', [
            {'path': str(path.relative_to(HERE)), 'bytes': path.stat().st_size, 'sha256': sha(path)}
            for path in sorted(artifact_dir.rglob('*')) if path.is_file()
        ])
        return metrics, proxy.records
    except BaseException:
        metadata.update(status='failed', failed_at=now(), traceback=traceback.format_exc())
        save(artifact_dir / 'run_metadata.json', metadata)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default=None,
                        help='unique result ID; defaults to a UTC timestamp (existing IDs are never overwritten)')
    args = parser.parse_args()

    cfg, base, manifest, bank = setup()
    if read(HERE / 'results' / 'offline_diagnostics.json')['status'] != 'passed':
        raise ValueError('Existing offline audit gate has not passed')
    if read(HERE / 'results' / 'test_results.json')['status'] != 'passed':
        raise ValueError('Current code test gate has not passed')
    if list(bank.weights) != [1.0, 1.0, 1.0]:
        raise ValueError('Frozen geometry weights changed')
    if cfg['evaluation']['n_action_steps'] != 8:
        raise ValueError('The formal baseline protocol is expected to remain at eight actions')

    run_id = args.run_id or ('action_horizon_ablation_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    result_dir = HERE / 'results' / run_id
    artifact_root = HERE / 'artifacts' / run_id
    result_dir.mkdir(parents=True, exist_ok=False)
    artifact_root.mkdir(parents=True, exist_ok=False)

    run_provenance = provenance(cfg, base, manifest)
    bootstrap_rng = np.random.default_rng(cfg['diagnostics']['bootstrap_seed'])
    bootstrap_indices = bootstrap_rng.integers(0, len(manifest['test_env_seeds']),
                                               (cfg['diagnostics']['bootstrap_samples'],
                                                len(manifest['test_env_seeds'])))
    root_metadata = {'status': 'running', 'started_at': now(), 'run_id': run_id,
                     'horizons': list(HORIZONS), 'provenance': run_provenance,
                     'artifact_root': str(artifact_root)}
    save(result_dir / 'run_metadata.json', root_metadata)

    results = {}
    rollout_rows = []
    try:
        for horizon in HORIZONS:
            metrics, per_seed = run_one(horizon, cfg, manifest, bank,
                                        artifact_root / f'n_action_steps_{horizon}', bootstrap_indices,
                                        run_provenance)
            results[horizon] = metrics
            for row in per_seed:
                rollout_rows.append({'action_horizon': horizon, **row})
            print(json.dumps({'horizon': horizon, 'mean_score': metrics['mean_score'],
                              'score_std': metrics['score_std'], 'n_rollouts': metrics['n_rollouts'],
                              'evaluation_seconds': metrics['evaluation_seconds']}, ensure_ascii=False))

        baseline = read(HERE / 'results' / 'eval_metrics.json')
        with (HERE / 'results' / 'eval_rollouts.csv').open(newline='', encoding='utf-8') as stream:
            baseline_rollouts = {int(row['evaluation_seed']): float(row['official_episode_score'])
                                 for row in csv.DictReader(stream)}
        control_rows = {int(row['evaluation_seed']): float(row['official_episode_score'])
                        for row in rollout_rows if row['action_horizon'] == 8}
        seed_order = manifest['test_env_seeds']
        control_delta = np.asarray([control_rows[s] - baseline_rollouts[s] for s in seed_order])
        summary_rows = []
        for horizon in HORIZONS:
            metric = results[horizon]
            scores = np.asarray([float(row['official_episode_score']) for row in rollout_rows
                                 if row['action_horizon'] == horizon], dtype=float)
            paired_delta = scores - np.asarray([control_rows[s] for s in seed_order])
            paired_means = paired_delta[bootstrap_indices].mean(axis=1)
            summary_rows.append({
                'action_horizon': horizon, 'n_rollouts': metric['n_rollouts'],
                'mean_score': metric['mean_score'], 'score_std': metric['score_std'],
                'mean_score_ci95_low': metric['mean_score_bootstrap_95ci'][0],
                'mean_score_ci95_high': metric['mean_score_bootstrap_95ci'][1],
                'mean_final_score': metric['mean_final_score'],
                'paired_delta_vs_8_mean': float(paired_delta.mean()),
                'paired_delta_vs_8_ci95_low': float(np.quantile(paired_means, .025)),
                'paired_delta_vs_8_ci95_high': float(np.quantile(paired_means, .975)),
                'mean_queries_per_rollout': metric['mean_queries_per_rollout'],
                'evaluation_seconds': metric['evaluation_seconds'],
            })
        write_csv(result_dir / 'eval_rollouts.csv', rollout_rows)
        write_csv(result_dir / 'summary.csv', summary_rows)
        report_lines = [
            '# Action horizon ablation: geometry-oracle ALT-30', '',
            f"Run ID: `{run_id}`. All conditions use the same 50 frozen environment seeds, the same 30-episode bank, the same 1:1:1 geometry distance, and the official Runner score.", '',
            'The requested 1, 3, 5, and 7 actions per memory-bank query were evaluated. An 8-action condition was rerun as a control and compared with the previously recorded formal 8-action result.', '',
            'This directory is the privileged 5D geometry-oracle diagnostic (current pusher and block state), not the image-encoder ALT policy.', '',
            '| Actions before re-query | Mean score | SD | 95% bootstrap CI | Paired Δ vs 8 (95% CI) | Mean queries/rollout |',
            '|---:|---:|---:|---:|---:|---:|'
        ]
        for row in summary_rows:
            report_lines.append(
                f"| {row['action_horizon']} | {row['mean_score']:.6f} | {row['score_std']:.6f} | "
                f"[{row['mean_score_ci95_low']:.6f}, {row['mean_score_ci95_high']:.6f}] | "
                f"{row['paired_delta_vs_8_mean']:+.6f} [{row['paired_delta_vs_8_ci95_low']:+.6f}, "
                f"{row['paired_delta_vs_8_ci95_high']:+.6f}] | {row['mean_queries_per_rollout']:.2f} |"
            )
        report_lines.extend([
            '', f"The rerun 8-action mean was `{results[8]['mean_score']:.9f}`; the previous formal run was `{baseline['mean_score']:.9f}`. Across matched seeds, mean score delta was `{control_delta.mean():+.9f}` (maximum absolute per-seed delta `{np.abs(control_delta).max():.9g}`).", '',
            'Official score is the runner’s maximum clipped coverage per episode. Paired intervals resample the same evaluation seeds. Full per-seed scores and retrieval/action traces are saved beside this report and under the ignored artifact directory.', ''
        ])
        (result_dir / 'report.md').write_text('\n'.join(report_lines), encoding='utf-8')
        root_metadata.update(status='completed', completed_at=now(),
                             baseline_8_step_mean=float(baseline['mean_score']),
                             rerun_8_step_mean=results[8]['mean_score'],
                             rerun_8_step_paired_delta_mean=float(control_delta.mean()),
                             rerun_8_step_max_abs_seed_delta=float(np.abs(control_delta).max()),
                             summary_path=str(result_dir / 'summary.csv'),
                             report_path=str(result_dir / 'report.md'))
        save(result_dir / 'run_metadata.json', root_metadata)
        save(result_dir / 'summary.json', {'run_id': run_id, 'status': 'completed',
                                           'summary': summary_rows, 'metrics_by_horizon': results,
                                           'previous_formal_8_step_mean': float(baseline['mean_score']),
                                           'rerun_8_step_paired_delta': stats(control_delta),
                                           'bootstrap_seed': cfg['diagnostics']['bootstrap_seed'],
                                           'bootstrap_samples': cfg['diagnostics']['bootstrap_samples']})
        print(f'RESULT_DIR={result_dir}')
    except BaseException:
        root_metadata.update(status='failed', failed_at=now(), traceback=traceback.format_exc())
        save(result_dir / 'run_metadata.json', root_metadata)
        raise


if __name__ == '__main__':
    main()
