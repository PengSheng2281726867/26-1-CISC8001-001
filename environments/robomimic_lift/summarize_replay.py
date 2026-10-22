"""Read saved replay traces; no simulation or training is performed."""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def main():
    result = json.loads((HERE / 'results/validation.json').read_text())
    directory = PROJECT / result['artifact_directory']
    summary = {'source_validation': result['artifact_directory']+'/validation.json',
               'source_validation_sha256': hashlib.sha256((HERE / 'results/validation.json').read_bytes()).hexdigest(),
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'strict_full_state_gate_changed': False, 'episodes': []}
    for episode in result['episodes']:
        path = directory / (episode['episode']+'_action_replay.npz')
        with np.load(path) as trace:
            actual, expected = trace['actual_states'], trace['expected_states']
        assert actual.shape == expected.shape and np.isfinite(actual).all()
        error = np.abs(actual-expected)
        layout = episode['model_state_layout']
        start = 1+layout['cube_qpos_start']
        cube_pos_error = np.linalg.norm(actual[:, start:start+3]-expected[:, start:start+3], axis=1)
        qa, qb = actual[:, start+3:start+7], expected[:, start+3:start+7]
        qa = qa / np.linalg.norm(qa, axis=1, keepdims=True)
        qb = qb / np.linalg.norm(qb, axis=1, keepdims=True)
        angle = 2*np.arccos(np.clip(np.abs(np.sum(qa*qb, axis=1)), 0, 1))
        peak = tuple(int(x) for x in np.unravel_index(np.argmax(error), error.shape))
        dim = peak[1]
        nq = layout['nq']
        cube_vel_start = layout['nv']-6
        label = ('time' if dim == 0 else 'qpos['+str(dim-1)+']' if dim <= nq
                 else 'qvel['+str(dim-1-nq)+']')
        if dim > nq and dim-1-nq >= cube_vel_start+3:
            label += ' (cube angular velocity)'
        first_over = {}
        for threshold in [1e-10, 1e-8, 1e-6, 1e-5]:
            indices = np.flatnonzero(error.max(axis=1) > threshold)
            first_over[str(threshold)] = int(indices[0]) if len(indices) else None
        row = {'episode': episode['episode'], 'transition_count': len(error),
               'trace_artifact': {'path': str(path.relative_to(PROJECT)).replace('\\', '/'),
                                  'bytes': path.stat().st_size,
                                  'sha256': hashlib.sha256(path.read_bytes()).hexdigest()},
               'first_action_index_above_max_state_error': first_over,
               'peak_error_action_index': peak[0], 'peak_error_dimension': label,
               'max_cube_position_error_mm': float(cube_pos_error.max()*1000),
               'max_cube_orientation_error_degrees': float(np.rad2deg(angle).max()),
               'max_abs_state_error': float(error.max()),
               'task_success': episode['continuous_action_replay_final_success']}
        summary['episodes'].append(row)
    summary['all_tasks_successful'] = all(r['task_success'] for r in summary['episodes'])
    summary['strict_full_state_gate_passed'] = result['continuous_action_replay_passed']
    summary['state_restoration_passed'] = result['state_restoration_passed']
    summary['note'] = ('Task success, state restoration, and strict open-loop state fidelity are separate. '
                       'No tolerance, simulator parameter, action, or dataset was modified.')
    (HERE / 'results/replay_diagnosis.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
