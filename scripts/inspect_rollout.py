"""Audit completed traces and predefined videos; no policy or simulator changes."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import traceback
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.path.insert(0, str(PROJECT/'src'))
from pusht.repro import configure_runtime, now, save_json, sha256
configure_runtime(PROJECT)


def main():
    import av
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if not args.run_id or any(x in args.run_id for x in ['/', '\\', '..']):
        raise ValueError('Invalid run ID')
    run_dir = PROJECT/'experiments/runs'/args.run_id
    metadata = json.loads((run_dir/'run_metadata.json').read_text())
    metrics = json.loads((run_dir/'eval_metrics.json').read_text())
    rows = list(csv.DictReader((run_dir/'eval_rollouts.csv').open()))
    queries = [json.loads(s) for s in Path(metrics['retrieval_full_path']).read_text().splitlines()]
    train_ids = set(metadata['dataset_manifest_ref']['train_episode_ids'])
    distances = []
    for q in queries:
        assert q['matched_episode_id'] in train_ids
        assert len(q['commands']) == 8 and np.asarray(q['commands']).shape == (8, 2)
        assert np.isfinite(q['commands']).all()
        assert np.min(q['commands']) >= 0 and np.max(q['commands']) <= 512
        assert 1 <= q['executed_actions'] <= 8
        distances.append(float(np.linalg.norm(np.array(q['agent_pos'])-q['matched_agent_pos'])))
    assert set(q['evaluation_seed'] for q in queries) == set(metadata['test_env_seeds'])
    video_checks = []
    for seed, path in metrics['video_paths_by_seed'].items():
        with av.open(path) as container:
            frames = [f.to_ndarray(format='rgb24') for f in container.decode(video=0)]
        trace = [q for q in queries if q['evaluation_seed'] == int(seed)]
        errors = []
        for q in trace:
            if q['current_step'] == 0:
                continue  # video recording starts after the first action, not at reset
            image = frames[q['current_step']-1].astype(np.float32)
            mask = (image[:, :, 2]-image[:, :, 0] > 80) & (image[:, :, 2]-image[:, :, 1] > 60)
            yy, xx = np.where(mask)
            if len(xx):
                centroid = np.array([xx.mean(), yy.mean()])
                errors.append(float(np.linalg.norm(centroid-np.array(q['agent_pos'])*96/512)))
        video_checks.append({'evaluation_seed': int(seed), 'decoded_frames': len(frames),
                             'query_coordinate_checks': len(errors),
                             'median_blue_centroid_error_pixels': float(np.median(errors)),
                             'max_blue_centroid_error_pixels': float(np.max(errors)),
                             'note': 'H264/96px raster centroid is approximate; not a policy input'})
        row = next(r for r in rows if int(r['evaluation_seed']) == int(seed))
        assert len(frames) == int(row['steps'])
    repeated = []
    for seed in metadata['test_env_seeds']:
        trace = [q for q in queries if q['evaluation_seed'] == seed]
        tail = 1
        for i in range(len(trace)-2, -1, -1):
            a, b = trace[i], trace[-1]
            if (a['matched_episode_id'], a['local_frame_index']) != (b['matched_episode_id'], b['local_frame_index']):
                break
            tail += 1
        repeated.append({'evaluation_seed': seed, 'consecutive_same_match_queries_at_end': tail,
                         'last_matched_episode': trace[-1]['matched_episode_id'],
                         'last_matched_frame': trace[-1]['local_frame_index'],
                         'last_cosine': trace[-1]['top1_cosine']})
    result = {'status': 'completed', 'inspected_at': now(), 'run_id': args.run_id,
              'trace_queries_checked': len(queries), 'invalid_action_events': 0,
              'episode_switches': sum(q['episode_switch'] for q in queries),
              'padded_queries': sum(q['boundary_padded'] for q in queries),
              'query_to_matched_position_distance_world_units': {
                  'mean': float(np.mean(distances)), 'median': float(np.median(distances)),
                  'p95': float(np.quantile(distances, .95)), 'max': float(np.max(distances))},
              'predefined_video_coordinate_checks': video_checks, 'repeated_match_tails': repeated,
              'diagnostic_script_sha256': sha256(__file__),
              'interpretation': 'Similarity alone is not task success. Absolute actions can be poorly suited to spatially mismatched retrieved states. Repeated-match tails can indicate stagnation; no compensation, threshold, blending or seed changes were introduced.'}
    save_json(run_dir/'rollout_inspection.json', result, immutable=True)
    print({k: v for k, v in result.items() if k not in ['repeated_match_tails', 'interpretation']})


if __name__ == '__main__':
    try:
        main()
    except Exception:
        with (PROJECT/'docs/failure_log.md').open('a', encoding='utf-8') as f:
            f.write('\n## '+now()+' rollout inspection failure\n\nCommand: `'+
                    ' '.join(sys.argv)+'`\n\n```text\n'+traceback.format_exc()+'```\n')
        raise
