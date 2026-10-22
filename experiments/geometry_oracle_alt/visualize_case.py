"""Visualize one saved test query; reads existing video/trace and never steps an env."""
import argparse
import json
import numpy as np
from common import HERE, load, read, save, sha


def main(seed, step):
    cfg, base, manifest = load()
    if seed not in manifest['test_env_seeds'] or step < 1:
        raise ValueError('Need a saved test seed and a positive query step')
    metrics = read(HERE / 'results' / 'eval_metrics.json')
    trace_path = metrics['retrieval_full_path']
    with open(trace_path, encoding='utf-8') as f:
        selected = [r for r in map(json.loads, f)
                    if r['evaluation_seed'] == seed and r['current_step'] == step]
    if len(selected) != 1:
        raise ValueError('Requested query is absent or ambiguous')
    row = selected[0]
    import cv2
    import zarr
    root = zarr.open(base['paths']['dataset'], mode='r')
    source = row['source_index']
    np.testing.assert_array_equal(root['data/state'][source], row['matched_state'])
    assert row['matched_episode'] in manifest['train_subsets']['30']
    delta = np.asarray(row['current_state'])-np.asarray(row['matched_state'])
    angle = abs((delta[4]+np.pi) % (2*np.pi)-np.pi)
    pusher, block = np.linalg.norm(delta[:2]), np.linalg.norm(delta[2:4])
    distance = (pusher/512)**2+(block/512)**2+(angle/np.pi)**2
    np.testing.assert_allclose([pusher, block, angle*180/np.pi, distance],
                               [row['pusher_error'], row['block_error'],
                                row['angle_error_degrees'], row['geometry_distance']], atol=1e-12, rtol=0)
    video_path = metrics['video_paths_by_seed'][str(seed)]
    cap = cv2.VideoCapture(video_path)
    # Video frame 0 is post-action step 1; frame step-1 is the query-time observation.
    cap.set(cv2.CAP_PROP_POS_FRAMES, step-1)
    ok, current = cap.read()
    cap.release()
    if not ok:
        raise ValueError('Requested video frame is absent')
    matched = root['data/img'][source].astype(np.uint8)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 6.6))
    axes[0].imshow(cv2.cvtColor(current, cv2.COLOR_BGR2RGB), interpolation='nearest')
    axes[1].imshow(matched, interpolation='nearest')
    axes[0].set_title(f'Current test observation\nSeed {seed}, step {step}', fontsize=13)
    axes[1].set_title(f"Matched training observation\nEpisode {row['matched_episode']}, frame {row['local_frame']}", fontsize=13)
    for ax, state in zip(axes, [row['current_state'], row['matched_state']]):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f'Pusher: ({state[0]:.2f}, {state[1]:.2f})\n'
                      f'T: ({state[2]:.2f}, {state[3]:.2f}), angle {state[4]*180/np.pi:.2f} deg', fontsize=10)
    fig.suptitle('Geometry-Oracle: top-1 over the frozen 3,806-frame training bank', fontsize=13)
    fig.text(.5, .13, f'Pusher error = {pusher:.4f}    T position error = {block:.2f}    '
             f'T angle error = {angle*180/np.pi:.2f} deg\n'
             f'Joint distance D = {distance:.6f}   (fixed weights 1 : 1 : 1)', ha='center', fontsize=11)
    fig.text(.5, .035, 'Current image: saved post-step video (red action marker). Match: original Zarr image.\n'
             'Positions use 512-world coordinates. This is one diagnostic example, not a representative-sample claim.',
             ha='center', fontsize=8, color='#475569')
    fig.subplots_adjust(left=.04, right=.98, top=.85, bottom=.32, wspace=.12)
    output = HERE / 'results' / 'figures' / f'query_seed{seed}_step{step}.png'
    fig.savefig(output, dpi=160)
    plt.close(fig)
    selection = ('first frozen test seed, final query' if (seed, step) == (100000, 296)
                 else f'explicitly requested seed {seed}, query step {step}')
    save(output.with_suffix('.json'), {'selection': selection+'; illustrative, not representative',
         'query': row, 'video_frame_index': step-1, 'trace_sha256': sha(trace_path),
         'video_sha256': sha(video_path), 'script_sha256': sha(__file__), 'figure_sha256': sha(output)})
    print(json.dumps({'figure': str(output), 'pusher_error': pusher, 'block_error': block,
                      'angle_error_degrees': angle*180/np.pi, 'D': distance}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=100000)
    parser.add_argument('--step', type=int, default=296)
    args = parser.parse_args()
    main(args.seed, args.step)
