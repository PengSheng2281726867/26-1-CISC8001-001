"""Display saved test-query video frames beside their actual retrieved expert frames.

Video frame t-1 is the observation at query step t; reset images are not recorded.
Compressed rendered video includes the environment's action marker. It is not a
claim that lossless raw encoder inputs were saved. No rollout or model is run.
"""
import argparse
import json
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.path.insert(0, str(PROJECT/'src'))
from pusht.repro import configure_runtime, load_split, now, save_json, sha256
configure_runtime(PROJECT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=100000)
    parser.add_argument('--steps', type=int, nargs='+', default=[8, 160, 296])
    parser.add_argument('--output-name', default='test_frame_matches_v1')
    args = parser.parse_args()
    if not args.output_name or any(s in args.output_name for s in ['/', '\\', '..']):
        raise ValueError('Invalid output name')
    if any(step <= 0 for step in args.steps):
        raise ValueError('Reset observation is not present in saved videos; select positive query steps')
    output = PROJECT/'results/visualizations'
    manifest_path = output/f'{args.output_name}.json'
    names = ['alt_030_split42_model42_pilot_v1', 'alt_100_split42_model42_pilot_v1']
    pngs = [output/f'{args.output_name}_alt{scale}.png' for scale in [30, 100]]
    if manifest_path.exists() or any(p.exists() for p in pngs):
        raise ValueError('Visualization exists; choose another --output-name')
    import av
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np
    import yaml
    import zarr
    font_path = Path(os.environ.get('WINDIR', 'C:/Windows'))/'Fonts/msyh.ttc'
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for name, destination in zip(names, pngs):
        run = PROJECT/'experiments/runs'/name
        meta = json.loads((run/'run_metadata.json').read_text())
        metrics = json.loads((run/'eval_metrics.json').read_text())
        cfg = yaml.safe_load((run/'resolved_config.yaml').read_text())
        split = load_split(cfg['split']['manifest'], cfg['paths']['dataset'])
        assert meta['status'] == 'completed' and args.seed in split['test_env_seeds']
        video = Path(metrics['video_paths_by_seed'][str(args.seed)])
        trace = Path(metrics['retrieval_full_path'])
        inventory = json.loads((run/'artifact_manifest.json').read_text())['files']
        for path in [video, trace]:
            entry = next(e for e in inventory if e['path'] == path.relative_to(PROJECT).as_posix())
            assert sha256(path) == entry['sha256']
        queries = [json.loads(line) for line in trace.read_text().splitlines()]
        queries = [q for q in queries if q['evaluation_seed'] == args.seed]
        with av.open(str(video)) as container:
            frames = [frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]
        data = zarr.open(cfg['paths']['dataset'], mode='r')
        ends = data['meta/episode_ends'][:]
        starts = np.r_[0, ends[:-1]]
        fig, axes = plt.subplots(len(args.steps), 2, figsize=(8, 3.35*len(args.steps)), squeeze=False)
        selected = []
        for row, step in enumerate(args.steps):
            q = next(q for q in queries if q['current_step'] == step)
            assert step <= len(frames)
            assert starts[q['matched_episode_id']]+q['local_frame_index'] == q['source_index']
            np.testing.assert_allclose(data['data/state'][q['source_index'], :2], q['matched_agent_pos'], atol=1e-5)
            axes[row, 0].imshow(frames[step-1], interpolation='nearest')
            axes[row, 1].imshow(data['data/img'][q['source_index']].astype(np.uint8), interpolation='nearest')
            current, matched = q['agent_pos'], q['matched_agent_pos']
            axes[row, 0].set_title(f'当前测试画面 · 第 {step} 步\n推杆 ({current[0]:.1f}, {current[1]:.1f})', fontsize=12, pad=8)
            axes[row, 1].set_title(f"匹配示范 {q['matched_episode_id']} · 第 {q['local_frame_index']} 帧\n相似度 {q['top1_cosine']:.4f} · 推杆 ({matched[0]:.1f}, {matched[1]:.1f})", fontsize=12, pad=8)
            for ax in axes[row]:
                ax.axis('off')
            selected.append({**{k:q[k] for k in ['current_step', 'matched_episode_id', 'local_frame_index',
                'source_index', 'top1_cosine', 'agent_pos', 'matched_agent_pos']}, 'video_frame_index':step-1})
        fig.suptitle(f"ALT-{meta['scale']}：当前帧与实际检索匹配帧\n测试种子 {args.seed}", fontsize=16, y=.995)
        fig.text(.5, .008, '蓝点：推杆　绿色 T：目标　灰色 T：被推动物体\n左图来自已保存测试视频，包含渲染标记；右图为原始示范图像。', ha='center', fontsize=10, color='#555555')
        fig.tight_layout(rect=(0, .045, 1, .94), h_pad=1.3)
        fig.savefig(destination, dpi=150, facecolor='white')
        plt.close(fig)
        records.append({'run_id':name, 'evaluation_seed':args.seed, 'selected_queries':selected,
            'video_sha256':sha256(video), 'trace_sha256':sha256(trace),
            'dataset_fingerprint':split['dataset_fingerprint'],
            'figure':destination.relative_to(PROJECT).as_posix(), 'figure_sha256':sha256(destination)})
    save_json(manifest_path, {'created_at':now(), 'script_sha256':sha256(__file__), 'status':'completed',
        'source':'existing test rollouts only; no new experiment', 'video_index_convention':'query t uses saved video frame t-1',
        'raw_input_limitation':'Current images are compressed rendered video frames, including action markers; not lossless saved policy tensors.',
        'records':records}, immutable=True)
    print('\n'.join(str(path) for path in pngs))


if __name__ == '__main__':
    main()
