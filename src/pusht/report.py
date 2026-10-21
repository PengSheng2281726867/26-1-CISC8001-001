"""Plots derived exclusively from completed, saved runs and recorded videos."""
import csv
import json
from pathlib import Path
import numpy as np


def video_diagnostic(project, run_dir, metadata, metrics):
    if not metrics['video_paths_by_seed']:
        return
    import av
    import matplotlib.pyplot as plt
    import zarr
    seed = min(int(k) for k in metrics['video_paths_by_seed'])
    with av.open(metrics['video_paths_by_seed'][str(seed)]) as container:
        frames = [f.to_ndarray(format='rgb24') for f in container.decode(video=0)]
    queries = [json.loads(line) for line in Path(metrics['retrieval_full_path']).read_text().splitlines()]
    queries = [q for q in queries if q['evaluation_seed'] == seed]
    cfg = __import__('yaml').safe_load((run_dir/'resolved_config.yaml').read_text())
    root = zarr.open(cfg['paths']['dataset'], mode='r')
    fig, axes = plt.subplots(2, 6, figsize=(12, 4.5))
    for col, i in enumerate(np.linspace(0, len(frames)-1, 6).astype(int)):
        step = int(i+1)  # recorder stores one frame after each actual control action
        q = max((q for q in queries if q['current_step'] < step), key=lambda q: q['current_step'])
        axes[0, col].imshow(frames[i])
        axes[0, col].set_title(f'seed {seed}, step {step}', fontsize=8)
        axes[1, col].imshow(root['data/img'][q['source_index']].astype(np.uint8))
        axes[1, col].set_title(f"expert {q['matched_episode_id']}:{q['local_frame_index']}\ncos={q['top1_cosine']:.3f}", fontsize=8)
        for row in range(2):
            axes[row, col].axis('off')
    fig.suptitle(metadata['run_id']+' | rollout (top) and retrieved expert frame (bottom)', fontsize=10)
    fig.tight_layout()
    fig.savefig(Path(project)/'results'/f"{metadata['run_id']}_diagnostic.png", dpi=120)
    plt.close(fig)


def report(project):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    project = Path(project)
    runs = []
    for run_dir in sorted((project/'experiments/runs').glob('*')):
        if not (run_dir/'eval_metrics.json').exists():
            continue
        metadata = json.loads((run_dir/'run_metadata.json').read_text())
        metrics = json.loads((run_dir/'eval_metrics.json').read_text())
        if metadata['status'] != 'completed':
            continue
        video_diagnostic(project, run_dir, metadata, metrics)
        if metadata['scope'] != 'single_model_seed_scale_pilot':
            continue
        runs.append((run_dir, metadata, metrics))
    if not runs:
        print('Video diagnostics generated; no trained pilot is completed yet.')
        return
    columns = ['run_id', 'scale', 'model_seed', 'mean_score', 'score_std', 'n_rollouts',
               'training_seconds', 'bank_entries', 'checkpoint_bytes', 'bank_bytes']
    with (project/'results/scale_summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for _, m, e in runs:
            writer.writerow({'run_id': m['run_id'], 'scale': m['scale'], 'model_seed': m['seeds']['model'],
                             'training_seconds': m['training_seconds'],
                             **{k: e[k] for k in columns if k in e}})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, (run_dir, m, e) in enumerate(runs):
        training = list(csv.DictReader((run_dir/'training_metrics.csv').open()))
        label = f"ALT-{m['scale']}, seed {m['seeds']['model']}"
        axes[0].plot([int(r['epoch']) for r in training], [float(r['mean_nt_xent']) for r in training], label=label)
        lo, hi = e['mean_score_bootstrap_95ci']
        axes[1].errorbar(i, e['mean_score'], yerr=[[e['mean_score']-lo], [hi-e['mean_score']]], fmt='o', capsize=5)
    axes[0].set(xlabel='Epoch', ylabel='Train NT-Xent (not policy score)', title='Fixed training recipe')
    axes[0].legend(fontsize=8)
    axes[1].set(xticks=range(len(runs)), xticklabels=[f"{m['scale']} / seed{m['seeds']['model']}" for _, m, _ in runs],
                ylim=(0, 1), xlabel='Independent train episodes / model seed',
                ylabel='Official closed-loop mean score', title='Pilot results; bootstrap 95% CI over fixed seeds')
    fig.tight_layout()
    fig.savefig(project/'results/pilot_comparison.png', dpi=150)
    plt.close(fig)
    paired = []
    for left_dir, left, left_metrics in runs:
        for right_dir, right, right_metrics in runs:
            if left['scale'] != 30 or right['scale'] != 100 or left['seeds']['model'] != right['seeds']['model']:
                continue
            if left['test_env_seeds'] != right['test_env_seeds'] or left['split_manifest_sha256'] != right['split_manifest_sha256']:
                raise ValueError('Paired comparison requires identical frozen split and simulator seeds')
            if left['dataset_fingerprint'] != right['dataset_fingerprint']:
                raise ValueError('Paired comparison dataset content differs')
            cfg_left = __import__('yaml').safe_load((left_dir/'resolved_config.yaml').read_text())
            cfg_right = __import__('yaml').safe_load((right_dir/'resolved_config.yaml').read_text())
            cfg_left['training'].pop('scale')
            cfg_right['training'].pop('scale')
            if cfg_left != cfg_right or left['pretrained_sha256'] != right['pretrained_sha256']:
                raise ValueError('Non-scale method/initialization configuration differs')
            a = {int(r['evaluation_seed']): float(r['official_episode_score']) for r in csv.DictReader((left_dir/'eval_rollouts.csv').open())}
            b = {int(r['evaluation_seed']): float(r['official_episode_score']) for r in csv.DictReader((right_dir/'eval_rollouts.csv').open())}
            if sorted(a) != sorted(b) or sorted(a) != sorted(left['test_env_seeds']):
                raise ValueError('Paired rollout coverage mismatch')
            difference = np.array([b[seed]-a[seed] for seed in sorted(a)])
            rng = np.random.default_rng(left_metrics['bootstrap_seed'])
            samples = difference[rng.integers(0, len(difference), (left_metrics['bootstrap_samples'], len(difference)))].mean(axis=1)
            paired.append({'model_seed': left['seeds']['model'], 'left_run_id': left['run_id'],
                           'right_run_id': right['run_id'], 'n_paired_simulator_seeds': len(difference),
                           'split_manifest_sha256': left['split_manifest_sha256'],
                           'dataset_fingerprint': left['dataset_fingerprint'],
                           'pretrained_sha256': left['pretrained_sha256'],
                           'code_start_commits': {'30': left['code_at_start']['commit'], '100': right['code_at_start']['commit']},
                           'optimizer_steps': {'30': left['optimizer_steps'], '100': right['optimizer_steps']},
                           'training_frame_counts': {'30': left['train_frame_count'], '100': right['train_frame_count']},
                           'resolved_config_differences': ['training.scale'],
                           'mean_score_100_minus_30': float(difference.mean()),
                           'paired_bootstrap_95ci': np.quantile(samples, [.025, .975]).tolist(),
                           'bootstrap_seed': left_metrics['bootstrap_seed'],
                           'bootstrap_samples': left_metrics['bootstrap_samples'],
                           'per_seed_difference': {str(seed): float(b[seed]-a[seed]) for seed in sorted(a)},
                           'interpretation_limit': 'Conditional on this model seed; simulator-seed resampling is not model-seed replication. Equal 40 epochs are not an equal optimizer-step budget.'})
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter([a[seed] for seed in sorted(a)], [b[seed] for seed in sorted(a)], alpha=.7)
            ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
            ax.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02), xlabel='ALT-30 official episode score',
                   ylabel='ALT-100 official episode score',
                   title=f"Same {len(difference)} simulator seeds; model seed {left['seeds']['model']}")
            ax.set_aspect('equal')
            fig.tight_layout()
            fig.savefig(project/'results'/f"paired_scale_scores_model{left['seeds']['model']}.png", dpi=150)
            plt.close(fig)
    if paired:
        from pusht.repro import save_json
        save_json(project/'results/paired_scale_comparison.json', {'comparisons': paired})
    print('Saved results/scale_summary.csv, pilot_comparison.png and video diagnostics.')
