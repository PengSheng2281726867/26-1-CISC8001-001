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
    print('Saved results/scale_summary.csv, pilot_comparison.png and video diagnostics.')
