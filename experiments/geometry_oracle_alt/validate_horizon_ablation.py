"""Audit action alignment, execution intervals and score aggregation post-run."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from common import HERE, load, now, read, save, sha
from geometry_bank import GeometryBank


HORIZONS = (1, 3, 5, 7, 8)


def rows_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def validate(run_id):
    cfg, base, manifest = load()
    bank = GeometryBank(base['paths']['dataset'], manifest['train_subsets']['30'])
    result_dir = HERE / 'results' / run_id
    artifact_root = HERE / 'artifacts' / run_id
    root_meta = read(result_dir / 'run_metadata.json')
    if root_meta.get('status') != 'completed':
        raise ValueError(f"Run is not completed: {root_meta.get('status')}")

    combined = rows_csv(result_dir / 'eval_rollouts.csv')
    if len(combined) != len(HORIZONS) * len(manifest['test_env_seeds']):
        raise ValueError('Unexpected combined rollout row count')
    seeds = manifest['test_env_seeds']
    expected_seeds = set(seeds)
    checks = {}

    for horizon in HORIZONS:
        directory = artifact_root / f'n_action_steps_{horizon}'
        run_rows = rows_csv(directory / 'eval_rollouts.csv')
        run_seed_scores = {int(row['evaluation_seed']): float(row['official_episode_score']) for row in run_rows}
        if set(run_seed_scores) != expected_seeds or len(run_rows) != len(seeds):
            raise ValueError(f'Horizon {horizon}: missing or duplicated evaluation seeds')

        log_rows = [json.loads(line) for line in (directory / 'retrieval_full.jsonl').read_text(encoding='utf-8').splitlines()]
        per_seed = {seed: [] for seed in seeds}
        action_count = 0
        for row in log_rows:
            seed = int(row['evaluation_seed'])
            if seed not in per_seed:
                raise ValueError(f'Horizon {horizon}: unexpected query seed {seed}')
            episode = int(row['matched_episode'])
            frame = int(row['local_frame'])
            source_index = bank.data.index(episode, frame)
            if int(row['source_index']) != source_index:
                raise ValueError(f'Horizon {horizon}: source index does not match episode/frame')
            expected_indices = np.minimum(source_index + np.arange(horizon), bank.data.ends[episode] - 1)
            actual_indices = np.asarray(row['action_source_indices'], dtype=np.int64)
            if actual_indices.shape != (horizon,) or not np.array_equal(actual_indices, expected_indices):
                raise ValueError(f'Horizon {horizon}: action chunk crosses or misindexes an episode boundary')
            commands = np.asarray(row['commands'], dtype=np.float32)
            if commands.shape != (horizon, 2) or not np.array_equal(commands, bank.data.actions[expected_indices]):
                raise ValueError(f'Horizon {horizon}: commands differ from source absolute actions')
            executed = int(row['executed_actions'])
            if not 1 <= executed <= horizon:
                raise ValueError(f'Horizon {horizon}: invalid executed action count {executed}')
            per_seed[seed].append(row)
            action_count += executed

        for seed in seeds:
            queries = per_seed[seed]
            if not queries or int(queries[0]['current_step']) != 0:
                raise ValueError(f'Horizon {horizon}, seed {seed}: first query missing at step 0')
            for previous, current in zip(queries, queries[1:]):
                if int(current['current_step']) != int(previous['current_step']) + int(previous['executed_actions']):
                    raise ValueError(f'Horizon {horizon}, seed {seed}: re-query interval does not match executed actions')
                if int(previous['executed_actions']) != horizon:
                    raise ValueError(f'Horizon {horizon}, seed {seed}: non-final query executed a partial chunk')
            final_step = int(queries[-1]['current_step']) + int(queries[-1]['executed_actions'])
            rollout = next(row for row in run_rows if int(row['evaluation_seed']) == seed)
            if final_step != int(rollout['steps']):
                raise ValueError(f'Horizon {horizon}, seed {seed}: trace length disagrees with rollout steps')

        metric = read(directory / 'eval_metrics.json')
        if metric['query_count'] != len(log_rows) or metric['n_rollouts'] != len(seeds):
            raise ValueError(f'Horizon {horizon}: metrics disagree with detailed trace/rollout count')
        if not np.isclose(metric['mean_score'], np.mean(list(run_seed_scores.values())), atol=1e-12, rtol=0):
            raise ValueError(f'Horizon {horizon}: mean score aggregation mismatch')
        checks[str(horizon)] = {'rollouts': len(run_rows), 'queries': len(log_rows),
                                'executed_actions': action_count,
                                'all_chunks_match_source_actions': True,
                                'no_cross_episode_chunks': True,
                                'query_intervals_match_execution': True,
                                'mean_score_recomputed': metric['mean_score']}

        for entry in read(directory / 'artifact_manifest.json'):
            path = HERE / entry['path']
            if not path.is_file() or path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
                raise ValueError(f'Horizon {horizon}: artifact checksum mismatch: {entry["path"]}')

    old_rows = {int(row['evaluation_seed']): float(row['official_episode_score'])
                for row in rows_csv(HERE / 'results' / 'eval_rollouts.csv')}
    new_eight = {int(row['evaluation_seed']): float(row['official_episode_score'])
                 for row in rows_csv(artifact_root / 'n_action_steps_8' / 'eval_rollouts.csv')}
    if old_rows != new_eight:
        raise ValueError('Eight-action rerun does not exactly reproduce the formal baseline scores')

    result = {'status': 'passed', 'validated_at': now(), 'run_id': run_id,
              'evaluation_seed_count': len(seeds), 'horizons': list(HORIZONS),
              'checks_by_horizon': checks,
              'eight_action_control_exactly_reproduces_formal_baseline': True,
              'artifact_checksums_verified': True}
    save(result_dir / 'trace_validation.json', result)
    root_meta.update(validation_status='passed', validation_path=str(result_dir / 'trace_validation.json'))
    save(result_dir / 'run_metadata.json', root_meta)
    report_path = result_dir / 'report.md'
    with report_path.open('a', encoding='utf-8') as stream:
        stream.write('\nTrace validation passed: all five conditions have 50 unique frozen evaluation seeds; action commands and source indices match the selected demonstration frame; chunks stay within their source episode; executed counts match every re-query interval; and all artifact checksums verify. The 8-action control reproduces the prior per-seed scores exactly.\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_id')
    args = parser.parse_args()
    print(json.dumps(validate(args.run_id), indent=2, ensure_ascii=False))
