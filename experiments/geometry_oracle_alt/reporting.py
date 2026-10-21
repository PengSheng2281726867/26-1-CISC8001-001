"""Recompute saved traces/paired statistics; make static scientific figures."""
import csv
import json
import numpy as np
from common import HERE, PROJECT, load, read, save, sha, stats


def csv_rows(path):
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def trace_metrics(rows, oracle=False):
    transitions, same, stall, switches, two_cycles = 0, 0, 0, 0, 0
    last, penultimate, distances, tails = {}, {}, [], {}
    for r in rows:
        seed = r['evaluation_seed']
        key = (r['matched_episode'], r['local_frame']) if oracle else (r['matched_episode_id'], r['local_frame_index'])
        pos = np.asarray(r['current_state'][:2] if oracle else r['agent_pos'])
        matched = np.asarray(r['matched_state'][:2] if oracle else r['matched_agent_pos'])
        distances.append(np.linalg.norm(pos-matched))
        tails[seed] = tails.get(seed, 0)+1 if seed in last and last[seed][0] == key else 1
        if seed in last:
            transitions += 1
            repeated = last[seed][0] == key
            same += repeated
            stall += repeated and np.linalg.norm(pos-last[seed][1]) <= .001
            switches += last[seed][0][0] != key[0]
            two_cycles += not repeated and seed in penultimate and penultimate[seed] == key
            penultimate[seed] = last[seed][0]
        last[seed] = (key, pos)
    return {'queries': len(rows), 'transitions': transitions, 'same_match_transitions': int(same),
            'same_match_fraction': same/transitions, 'same_match_and_same_position_transitions': int(stall),
            'same_match_and_same_position_fraction': stall/transitions,
            'two_cycle_returns_excluding_one_cycles': int(two_cycles), 'episode_switches': int(switches),
            'query_to_matched_pusher_distance': stats(distances),
            'same_match_queries_at_episode_end': {str(k): v for k, v in tails.items()},
            'episodes_ending_in_3plus_same_match_queries': sum(v >= 3 for v in tails.values()),
            'padded_queries': sum(r['boundary_padded'] for r in rows)}


def verify():
    from run import setup
    from geometry_bank import components
    cfg, base, manifest, bank = setup()
    metric = read(HERE / 'results' / 'eval_metrics.json')
    rows = [json.loads(line) for line in open(metric['retrieval_full_path'], encoding='utf-8')]
    scores = csv_rows(HERE / 'results' / 'eval_rollouts.csv')
    assert [int(r['evaluation_seed']) for r in scores] == manifest['test_env_seeds']
    assert len(rows) == metric['query_count'] == metric['sync_checks']
    assert len(metric['video_paths_by_seed']) == 4
    for video in metric['video_paths_by_seed'].values():
        assert __import__('pathlib').Path(video).is_file()
    for r in rows:
        i, d, terms = bank.query(r['current_state'])
        assert i == r['bank_index']
        np.testing.assert_allclose(d, r['geometry_distance'], rtol=0, atol=1e-15)
        np.testing.assert_array_equal(terms, r['distance_components'])
        np.testing.assert_array_equal(bank.states[i], r['matched_state'])
        assert int(bank.episode_id[i]) == r['matched_episode']
        assert int(bank.frame_index[i]) == r['local_frame']
        assert int(bank.source_index[i]) == r['source_index']
        chunk, indices = bank.action_chunk(i)
        np.testing.assert_array_equal(chunk, r['commands'])
        np.testing.assert_array_equal(indices, r['action_source_indices'])
        assert 0 < r['executed_actions'] <= 8
        assert r['state_sync_pusher_max_abs_error'] <= cfg['diagnostics']['position_sync_atol']
        assert 0 <= r['current_state'][4] < 2*np.pi
    for score in scores:
        episode = [r for r in rows if r['evaluation_seed'] == int(score['evaluation_seed'])]
        assert [r['current_step'] for r in episode] == list(range(0, int(score['steps']), 8))
        assert sum(r['executed_actions'] for r in episode) == int(score['steps'])
    assert abs(np.mean([float(r['official_episode_score']) for r in scores])-metric['mean_score']) < 1e-15
    for entry in read(HERE / 'results' / 'artifact_manifest.json'):
        assert sha(PROJECT / entry['path']) == entry['sha256']
    save(HERE / 'results' / 'independent_verification.json',
         {'status': 'passed', 'queries_recomputed': len(rows), 'rollouts_checked': len(scores),
          'all_argmins_components_source_states_and_actions_equal': True,
          'all_step_counts_and_padding_valid': True, 'artifact_hashes_verified': True,
          'trace_metrics': trace_metrics(rows, True)})
    print('Independent verification passed:', len(rows), 'queries;', len(scores), 'rollouts')


def report():
    cfg, base, manifest = load()
    offline = read(HERE / 'results' / 'offline_diagnostics.json')
    oracle = read(HERE / 'results' / 'eval_metrics.json')
    offline_old = read(PROJECT / cfg['baseline']['b1_run'] / 'offline_diagnostics.json')
    assert offline_old['split_manifest_sha256'] == cfg['split_sha256']
    for key in ['baseline', 'b1']:
        assert offline_old[key]['heldout_episode_ids'] == manifest['heldout_validation_episode_ids']
        assert offline_old[key]['heldout_frames'] == offline['validation_frames']
    methods = {}
    for key in ['alt', 'b1']:
        directory = PROJECT / cfg['baseline'][key+'_run']
        metrics = read(directory / 'eval_metrics.json')
        table = csv_rows(directory / 'eval_rollouts.csv')
        assert [int(r['evaluation_seed']) for r in table] == manifest['test_env_seeds']
        trace = [json.loads(s) for s in open(metrics['retrieval_full_path'], encoding='utf-8')]
        methods[key] = {'score': metrics['mean_score'], 'score_std': metrics['score_std'],
                        'scores': [float(r['official_episode_score']) for r in table],
                        'offline': offline_old['baseline' if key == 'alt' else 'b1']['retrieval'],
                        'trace': trace_metrics(trace),
                        'source_files': {str((directory/f).relative_to(PROJECT)): sha(directory/f) for f in
                                         ['eval_metrics.json', 'eval_rollouts.csv']},
                        'trace_sha256': sha(metrics['retrieval_full_path'])}
    oracle_table = csv_rows(HERE / 'results' / 'eval_rollouts.csv')
    trace = [json.loads(s) for s in open(oracle['retrieval_full_path'], encoding='utf-8')]
    methods['oracle'] = {'score': oracle['mean_score'], 'score_std': oracle['score_std'],
                         'scores': [float(r['official_episode_score']) for r in oracle_table],
                         'offline': offline['retrieval'], 'trace': trace_metrics(trace, True)}
    rng = np.random.default_rng(cfg['diagnostics']['bootstrap_seed'])
    resamples = rng.integers(0, 50, (cfg['diagnostics']['bootstrap_samples'], 50))
    paired = {}
    for key in ['alt', 'b1']:
        delta = np.array(methods['oracle']['scores'])-methods[key]['scores']
        paired['oracle_minus_'+key] = {'mean': float(delta.mean()),
                                      'bootstrap_95ci': np.quantile(delta[resamples].mean(axis=1), [.025, .975]).tolist(),
                                      'positive_seeds': int(np.sum(delta > 0)), 'negative_seeds': int(np.sum(delta < 0)),
                                      'tied_seeds': int(np.sum(delta == 0))}
    comparison = {'methods': methods, 'paired_score_differences': paired,
                  'bootstrap': {'seed': 2026, 'samples': 10000, 'method': 'percentile; paired resampling of identical 50 seeds'},
                  'limitations': ['Privileged oracle inputs differ from visual models',
                                  'Offline matching uses the same 2426 validation frames; rollout queries follow different state distributions',
                                  'Baseline rollout block truth was not saved; its block/angle rollout errors cannot be reconstructed',
                                  'Intervals quantify variability over these evaluation seeds, not model-training randomness']}
    save(HERE / 'results' / 'comparison.json', comparison)
    from run import write_csv
    write_csv(HERE / 'results' / 'paired_scores.csv',
              [{'evaluation_seed': seed, **{k: methods[k]['scores'][j] for k in methods},
                'oracle_minus_alt': methods['oracle']['scores'][j]-methods['alt']['scores'][j],
                'oracle_minus_b1': methods['oracle']['scores'][j]-methods['b1']['scores'][j]}
               for j, seed in enumerate(manifest['test_env_seeds'])])
    figures(methods, paired, oracle, trace, base)
    generate_report(methods, paired, oracle, offline)
    print(json.dumps({'scores': {k: m['score'] for k, m in methods.items()}, 'paired': paired}, indent=2))


def figures(methods, paired, oracle, trace, base):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = HERE / 'results' / 'figures'
    directory.mkdir(parents=True, exist_ok=True)
    colors = ['#64748b', '#f59e0b', '#059669']
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for j, (key, method) in enumerate(methods.items()):
        axes[0].scatter(np.full(50, j)+np.linspace(-.16, .16, 50), method['scores'], s=9, color=colors[j], alpha=.55)
        axes[0].plot([j-.2, j+.2], [method['score']]*2, color=colors[j], linewidth=3)
    axes[0].set(xticks=[0, 1, 2], xticklabels=['ALT', 'B1', 'Oracle'], ylabel='Official max-reward score', ylim=(-.03, 1.04))
    for j, key in enumerate(['alt', 'b1']):
        p = paired['oracle_minus_'+key]
        low, high = p['bootstrap_95ci']
        axes[1].errorbar(j, p['mean'], yerr=[[p['mean']-low], [high-p['mean']]], fmt='o', capsize=5, color=colors[j])
    axes[1].axhline(0, color='gray', linestyle='--')
    axes[1].set(xticks=[0, 1], xticklabels=['Oracle - ALT', 'Oracle - B1'], ylabel='Paired score difference (95% CI)')
    for j, (key, method) in enumerate(methods.items()):
        axes[2].bar(j, method['trace']['same_match_fraction'], color=colors[j])
    axes[2].set(xticks=[0, 1, 2], xticklabels=['ALT', 'B1', 'Oracle'], ylabel='Repeated top-1 match / query transitions', ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(directory / 'closed_loop_comparison.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3))
    for ax, metric, label in zip(axes, ['pusher_error', 'block_error', 'angle_error_degrees', 'action_chunk_rmse'],
                                 ['Pusher error (world units)', 'T position error (world units)', 'T angle error (degrees)', '8-action RMSE (world units)']):
        ax.bar(range(3), [m['offline'][metric]['mean'] for m in methods.values()], color=colors)
        ax.set(xticks=range(3), xticklabels=['ALT', 'B1', 'Oracle'], title=label)
    fig.suptitle('Offline matching: identical 2,426 held-out frames; geometric proximity does not imply action compatibility')
    fig.tight_layout()
    fig.savefig(directory / 'offline_comparison.png', dpi=160)
    plt.close(fig)
    # Video snapshots are post-step observations. Query at t corresponds to video frame t-1.
    import cv2
    import zarr
    root = zarr.open(base['paths']['dataset'], mode='r')
    selected = []
    for seed in range(100000, 100004):
        eligible = [r for r in trace if r['evaluation_seed'] == seed and r['current_step'] >= 8]
        selected.extend([eligible[0], eligible[-1]])
    fig, axes = plt.subplots(8, 2, figsize=(8, 22))
    manifest_rows = []
    for row, r in enumerate(selected):
        cap = cv2.VideoCapture(oracle['video_paths_by_seed'][str(r['evaluation_seed'])])
        cap.set(cv2.CAP_PROP_POS_FRAMES, r['current_step']-1)
        ok, current = cap.read()
        cap.release()
        if not ok:
            raise ValueError('Could not read predeclared video inspection frame')
        matched = root['data/img'][r['source_index']].astype(np.uint8)
        axes[row, 0].imshow(cv2.cvtColor(current, cv2.COLOR_BGR2RGB))
        axes[row, 1].imshow(matched)
        axes[row, 0].set_title(f"Seed {r['evaluation_seed']}, current step {r['current_step']}")
        axes[row, 1].set_title(f"Match ep {r['matched_episode']}, frame {r['local_frame']}\np={r['pusher_error']:.1f}, T={r['block_error']:.1f}, angle={r['angle_error_degrees']:.1f} deg")
        for ax in axes[row]:
            ax.axis('off')
        manifest_rows.append({'seed': r['evaluation_seed'], 'query_step': r['current_step'],
                              'video_frame_index': r['current_step']-1, 'source_index': r['source_index']})
    fig.tight_layout()
    fig.savefig(directory / 'current_vs_matches.png', dpi=140)
    plt.close(fig)
    save(HERE / 'results' / 'visualization_manifest.json', manifest_rows)


def generate_report(methods, paired, oracle, offline):
    lines = ['# Geometry-Oracle ALT-30 结果与分析', '',
             '本实验已完成；没有训练或加载任何神经网络。Oracle 在每次重规划时读取仿真器当前五维真实状态，属于 **Privileged Information**。它不是与视觉 ALT 同等输入的部署方案，也不是严格理论性能上界。', '',
             '## 设置与验证', '',
             '使用原冻结 split42 的 30 条训练 Episode、3,806 帧；20 条验证 Episode、2,426 帧仅用于离线查询。Bank 未包含验证帧。原始 float32 状态逐项一致，检索用 float64；周期角距离、确定性相同距离排序、自检索、全部帧动作边界、环境行为一致性和 Windows spawn AsyncVectorEnv 状态同步测试均通过。', '',
             '距离固定为 `(dp/512)^2 + (dT/512)^2 + (wrap(dtheta)/pi)^2`，权重 1:1:1。Top-1、绝对动作、offset=0、8 步重规划、repeat-last padding；无补偿、混合或 OOD fallback。', '',
             '正式评测仅执行一轮：官方 PushTImageRunner.run，冻结 Seeds 100000–100049，legacy_test=True、max_steps=300。主指标是各回合最大 `clip(coverage/0.95,0,1)` 的均值，不是最终覆盖率或二元成功率。', '',
             f"状态同步检查 {oracle['sync_checks']} 次，Oracle 与观察 pusher xy 最大差 {oracle['max_pusher_sync_error']:.8f}（float32 共享内存舍入）；五维状态与生成观察时捕获的快照完全一致。", '',
             '## 实际统计：离线检索', '',
             '以下为相同 2,426 个验证帧上的均值；位置及动作采用原始 512 世界坐标单位，角度采用度。动作 RMSE 是 8×2 命令差异的逐帧均方根再平均，不代表唯一正确动作标签。', '',
             '| 方法 | 推杆误差 | T位置误差 | T角度误差 | 8步动作RMSE |',
             '|---|---:|---:|---:|---:|']
    for key, label in [('alt', '原 ALT-30'), ('b1', 'B1'), ('oracle', 'Geometry Oracle')]:
        m = methods[key]['offline']
        lines.append('| '+label+' | '+' | '.join(f"{m[k]['mean']:.3f}" for k in ['pusher_error', 'block_error', 'angle_error_degrees', 'action_chunk_rmse'])+' |')
    lines += ['', '## 实际统计：闭环控制', '', '| 方法 | Mean Score | 重复同帧/相邻查询 | 同帧且推杆停滞/相邻查询 | 闭环推杆匹配误差 |', '|---|---:|---:|---:|---:|']
    for key, label in [('alt', '原 ALT-30'), ('b1', 'B1'), ('oracle', 'Geometry Oracle')]:
        m, t = methods[key], methods[key]['trace']
        lines.append(f"| {label} | {m['score']:.9f} | {t['same_match_fraction']:.2%} | {t['same_match_and_same_position_fraction']:.2%} | {t['query_to_matched_pusher_distance']['mean']:.3f} |")
    lines += ['', '重复比例分母排除每个回合首个查询，停滞定义为连续检索同一 Episode/Frame 且推杆移动 ≤0.001。它是描述性诊断，未用于策略决策。三个方法的闭环状态分布不同，匹配误差不能视作同一批查询上的比较。原 ALT/B1 没有保存闭环 T 块真实状态，因此其闭环 T 位置/角误差不可恢复；上表之外的 T 匹配比较使用离线验证集。', '',
              f"Oracle 的 Score 标准差 {oracle['score_std']:.6f}，均值 bootstrap 95% CI {oracle['mean_score_bootstrap_95ci']}；平均最终分数 {oracle['mean_final_score']:.6f}，总环境步数 {oracle['total_environment_steps']}。", '',
              '同一 50 Seeds 的配对差异；固定随机种子 2026，10,000 次按 Seed 成对重采样的 percentile 95% CI：', '']
    for key, label in [('alt', '原 ALT'), ('b1', 'B1')]:
        p = paired['oracle_minus_'+key]
        lines.append(f"- Oracle − {label}: {p['mean']:+.6f}，95% CI [{p['bootstrap_95ci'][0]:+.6f}, {p['bootstrap_95ci'][1]:+.6f}]；更高/更低/相同 Seeds = {p['positive_seeds']}/{p['negative_seeds']}/{p['tied_seeds']}。")
    lines += ['', '该区间仅描述这些评测初始状态的不确定性，不涵盖视觉模型训练种子变化，也不代表调参后的总体最优结果。', '',
              '## 问题回答与解释边界', '',
              '1. **真实几何状态检索是否改善？** 相同离线查询上，Oracle 的推杆、T 位置、T 角度平均误差均低于原 ALT；动作 RMSE 也需单独看待。改善说明原视觉嵌入的几何区分确有不足，但直接读取真实状态本身也提供了更强的信息。', '',
              f"2. **闭环是否随之改善？** 本轮 Oracle Mean Score 为 {oracle['mean_score']:.6f}，相对原 ALT 的差异为 {paired['oracle_minus_alt']['mean']:+.6f}；置信区间如上。应依据实际差异与区间判断，不能把离线更近直接等同于控制成功。", '',
              '3. **剩余失败来自哪里？** 下列是分析推测，不能由本次单一实验确证因果：五维状态没有速度、接触历史和动作意图；最近示范帧仍有非零误差；8 步绝对目标回放在偏移状态上可能形成重复检索或循环。记录中的停滞/循环与这些机制相容，但不能单独区分它们。', '',
              '本轮没有改变执行长度、引入动作相对平移或进一步做对照；因此不会把剩余失败全部归因于 8 步回放，也不会声称真实状态检索保证动作相容。', '',
              '## 产物和后续建议', '',
              '- `results/offline_diagnostics.json`：Bank、源数据、自检索及离线误差。',
              '- `results/eval_rollouts.csv`、`paired_scores.csv`、`comparison.json`：50 回合和配对统计。',
              '- `results/independent_verification.json`：全量查询重新计算、动作索引/边界/步数和产物哈希。',
              '- `results/figures/`：分数、离线误差，以及前四个固定 Seed 在第一次重规划和最后一次查询的当前画面/匹配画面。当前图取视频第 t−1 帧（step t），含红色动作标记；匹配图取原始 Zarr。',
              '- `artifacts/geometry_oracle_alt030_v1/`：完整检索 JSONL、前四个 Seed 视频、冻结配置和运行记录；本地保留，Git 不包含这些大文件。', '',
              '建议先结合保存的轨迹和视频区分“匹配不近”“动作不相容”“重复回放”；若未来设计额外对照，应另立固定方案。此次实验到此停止，不训练 B2-B，不搜索权重或修改原 ALT/B1。', '']
    (HERE / 'analysis_report.md').write_text('\n'.join(lines), encoding='utf-8')
