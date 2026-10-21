"""Visualize saved B1 test observations, actual retrieved frames and chunk repeats.

Read-only analysis of completed rollouts. No model, simulator, retraining, or new
split. Adapted from the project's attributed scripts/visualize_test_matches.py;
all new outputs stay inside this isolated experiment.
"""
import argparse
import json
import os
from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = EXPERIMENT.parents[1]
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.path.insert(0, str(PROJECT/'src'))
sys.path.insert(0, str(EXPERIMENT))
from pusht.repro import configure_runtime, now, save_json, sha256
configure_runtime(EXPERIMENT)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id', default='b1_alt030_split42_model42_pose1_pilot_v1')
    p.add_argument('--seed', type=int, default=100000)
    p.add_argument('--steps', type=int, nargs='+', default=[8, 160, 296])
    p.add_argument('--repeat-start', type=int, default=152)
    p.add_argument('--output-name', default='b1_test_matches_v1')
    args = p.parse_args()
    import re
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.output_name):
        raise ValueError('Use a safe, unused output name')
    from b1alt.common import config, contract
    from b1alt.experiment import existing
    cfg = config()
    split, rd, heavy, meta = existing(cfg, args.run_id)
    if meta['status'] != 'completed' or args.seed not in split['test_env_seeds']:
        raise ValueError('Completed frozen-seed source run required')
    output = EXPERIMENT/'results/visualizations'
    output.mkdir(parents=True, exist_ok=True)
    png = output/f'{args.output_name}.png'
    repeat_png = output/f'{args.output_name}_repetition.png'
    manifest = output/f'{args.output_name}.json'
    if any(path.exists() for path in [png, repeat_png, manifest]):
        raise ValueError('Outputs exist; choose a new output name, never overwrite')
    import av
    import numpy as np
    import zarr
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = Path(os.environ.get('WINDIR', 'C:/Windows'))/'Fonts/msyh.ttc'
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    metrics = json.loads((rd/'eval_metrics.json').read_text())
    inspection = json.loads((rd/'rollout_inspection.json').read_text())
    video = Path(metrics['video_paths_by_seed'][str(args.seed)])
    trace = Path(metrics['retrieval_full_path'])
    inventory = json.loads((rd/'artifact_manifest.json').read_text())['files']
    for path in [video, trace]:
        entry = next(v for v in inventory if v['path'] == path.relative_to(PROJECT).as_posix())
        assert sha256(path) == entry['sha256'] and path.stat().st_size == entry['bytes']
    queries = [json.loads(s) for s in trace.read_text().splitlines()]
    queries = [q for q in queries if q['evaluation_seed'] == args.seed]
    q_by_step = {q['current_step']: q for q in queries}
    with av.open(str(video)) as container:
        frames = [f.to_ndarray(format='rgb24') for f in container.decode(video=0)]
    data = zarr.open(cfg['paths']['dataset'], mode='r')
    actions = data['data/action'][:]
    ends = data['meta/episode_ends'][:]
    starts = np.r_[0, ends[:-1]]
    selected_steps = args.steps + [args.repeat_start+8*k for k in range(3)]
    for step in selected_steps:
        if step <= 0 or step not in q_by_step or step > len(frames):
            raise ValueError('Choose recorded positive replan steps present in the video')
        q = q_by_step[step]
        assert starts[q['matched_episode_id']]+q['local_frame_index'] == q['source_index']
        np.testing.assert_allclose(data['data/state'][q['source_index'], :2], q['matched_agent_pos'], atol=1e-5)
        ids = np.minimum(q['source_index']+np.arange(8), ends[q['matched_episode_id']]-1)
        np.testing.assert_array_equal(ids, q['action_source_indices'])
        np.testing.assert_array_equal(actions[ids], q['commands'])
    fig, axes = plt.subplots(len(args.steps), 2, figsize=(9, 3.7*len(args.steps)), squeeze=False)
    for row, step in enumerate(args.steps):
        q = q_by_step[step]
        current, matched = q['agent_pos'], q['matched_agent_pos']
        distance = np.linalg.norm(np.array(current)-matched)
        axes[row, 0].imshow(frames[step-1], interpolation='nearest')
        axes[row, 1].imshow(data['data/img'][q['source_index']].astype(np.uint8), interpolation='nearest')
        axes[row, 0].set_title(f'测试中的当前画面 · 第 {step} 步\n推杆 ({current[0]:.1f}, {current[1]:.1f})', fontsize=12, pad=10)
        axes[row, 1].set_title(f"实际匹配：示范 {q['matched_episode_id']} / 帧 {q['local_frame_index']}\n"
                              f"推杆 ({matched[0]:.1f}, {matched[1]:.1f}) · 距离误差 {distance:.1f}\n"
                              f"cosine = {q['top1_cosine']:.4f}", fontsize=11, pad=10)
        for ax in axes[row]:
            ax.axis('off')
    fig.suptitle(f'B1：测试画面与检索到的专家示范\n同一测试 seed {args.seed}', fontsize=17, y=.995)
    fig.text(.5, .008, '蓝点：推杆　灰色 T：被推动物体　绿色 T：目标\n'
             '左图取自已保存测试视频，带压缩和动作标记；右图是原始示范图像。', ha='center', fontsize=10, color='#555555')
    fig.tight_layout(rect=(0, .045, 1, .94), h_pad=1.3)
    fig.savefig(png, dpi=145, facecolor='white')
    plt.close(fig)

    repeated = [q_by_step[args.repeat_start+8*k] for k in range(3)]
    # The selected example must actually repeat the same entry and all 8 commands.
    assert len({q['source_index'] for q in repeated}) == 1
    assert all(q['commands'] == repeated[0]['commands'] for q in repeated)
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1], hspace=.38, wspace=.32)
    for col, q in enumerate(repeated):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(frames[q['current_step']-1], interpolation='nearest')
        ax.axis('off')
        ax.set_title(f"第 {q['current_step']} 步：再次查询\n匹配 {q['matched_episode_id']}:{q['local_frame_index']}\n"
                     f"当前推杆 ({q['agent_pos'][0]:.2f}, {q['agent_pos'][1]:.2f})", fontsize=12, pad=10)
    ax = fig.add_subplot(gs[1, 0])
    q = repeated[0]
    commands = np.array(q['commands'])
    ax.plot(commands[:, 0], commands[:, 1], '-o', color='#7a3ca3', markersize=4)
    for i, xy in enumerate(commands):
        ax.annotate(str(i+1), xy, xytext=(3, 6), textcoords='offset points', fontsize=9)
    ax.scatter(*q['agent_pos'], s=70, color='#2563eb', label='当前推杆', zorder=3)
    ax.scatter(*q['matched_agent_pos'], s=65, color='#dc6b17', label='示范推杆', zorder=3)
    ax.set(xlabel='世界坐标 x', ylabel='世界坐标 y', title='每次都从目标 1 重新执行到目标 8')
    ax.invert_yaxis()
    ax.grid(alpha=.2)
    ax.legend(fontsize=9, loc='best')
    ax.text(.5, -.26, '紫线是目标命令顺序，不是实测运动轨迹', transform=ax.transAxes, ha='center', fontsize=9, color='#555555')
    ax = fig.add_subplot(gs[1, 1:])
    ax.axis('off')
    rows = []
    for item in repeated:
        source_frames = [int(x-starts[item['matched_episode_id']]) for x in item['action_source_indices']]
        rows.append([f"t={item['current_step']}", f"{item['matched_episode_id']}:{item['local_frame_index']}",
                     f"{source_frames[0]}–{source_frames[-1]}"])
    table = ax.table(cellText=rows, colLabels=['重规划时刻', '检索的示范:帧', '再次执行的示范动作帧'],
                     cellLoc='center', colWidths=[.22, .32, .46], bbox=[0, .52, 1, .43])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    a, b = inspection['baseline'], inspection['b1']
    ax.text(0, .39, '全部 50 seeds 的相邻查询统计：', fontsize=12, weight='bold')
    ax.text(0, .28, f"同一条目被再次检索：{a['consecutive_same_match_fraction']:.2%} → {b['consecutive_same_match_fraction']:.2%}", fontsize=12)
    ax.text(0, .17, f"同一条目且重规划位置相同：{a['same_match_and_same_position_fraction']:.2%} → {b['same_match_and_same_position_fraction']:.2%}", fontsize=12)
    ax.text(0, .04, '含义：8 步后再次查询，又重播同一段动作。\n'
            '统计不表示 8 步内一直静止，也不是失败回合占比。', fontsize=11, color='#444444')
    fig.suptitle('“重复动作” = 反复从同一示范帧重播八步，而非继续往后执行', fontsize=17, y=.98)
    fig.subplots_adjust(top=.88, bottom=.12)
    fig.savefig(repeat_png, dpi=145, facecolor='white')
    plt.close(fig)
    fields = ['current_step', 'matched_episode_id', 'local_frame_index', 'source_index',
              'top1_cosine', 'agent_pos', 'matched_agent_pos', 'commands', 'action_source_indices', 'executed_actions']
    save_json(manifest, {'status': 'completed', 'created_at': now(), 'run_id': args.run_id,
        'evaluation_seed': args.seed, 'query_steps': args.steps, 'repeat_start': args.repeat_start,
        'script_sha256': sha256(__file__), 'video_path': str(video), 'video_sha256': sha256(video),
        'trace_path': str(trace), 'trace_sha256': sha256(trace),
        'split_sha256': meta['split_manifest_sha256'], 'dataset_fingerprint': split['dataset_fingerprint'],
        'video_index': 'query at current_step t -> video index t-1; no reset frame saved',
        'source_note': 'Existing saved rollout only. Video has lossy compression and action overlays; not archived lossless policy input tensors.',
        'repetition_definition': 'same bank source index on consecutive queries; same-position additionally uses L2 <=0.001 world units',
        'selected_queries': [{k: q_by_step[step][k] for k in fields} for step in args.steps],
        'repeated_queries': [{k: q[k] for k in fields} for q in repeated],
        'all50_trace_statistics': {key: {k: inspection[key][k] for k in
            ['queries', 'transitions', 'consecutive_same_match_transitions', 'same_match_and_same_position_transitions',
             'consecutive_same_match_fraction', 'same_match_and_same_position_fraction']} for key in ['baseline', 'b1']},
        'figures': [{'path': path.relative_to(EXPERIMENT).as_posix(), 'sha256': sha256(path)} for path in [png, repeat_png]],
        'no_new_model_or_rollout': True}, immutable=True)
    print(json.dumps({'status': 'completed', 'figures': [str(png), str(repeat_png)], 'all_action_indices_checked': True}))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        with (EXPERIMENT/'docs/failure_log.md').open('a', encoding='utf-8') as f:
            f.write('\n## '+now()+' saved-test visualization\n\nCommand: `'+
                    ' '.join(sys.argv)+'`\n\n```text\n'+traceback.format_exc()+'```\n')
        raise
