"""Independent NumPy FP64 recheck of saved per-frame diagnostic quantities."""
import csv
import json
from pathlib import Path
import numpy as np
from pusht.repro import save_json, sha256


def verify_offline(rd, heavy, data):
    result = {}
    states = data.root['data/state'][:].astype(np.float64)
    diagnostic = json.loads((rd/'offline_diagnostics.json').read_text())
    for method, filename in [('baseline', 'baseline_heldout_matches.csv'), ('b1', 'b1_heldout_matches.csv')]:
        path = heavy/filename
        with path.open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        source = np.array([int(r['source_index']) for r in rows])
        matched = np.array([int(r['matched_source_index']) for r in rows])
        q, m = states[source], states[matched]
        fields = {
            'pusher_error': np.linalg.norm(q[:, :2]-m[:, :2], axis=-1),
            'block_error': np.linalg.norm(q[:, 2:4]-m[:, 2:4], axis=-1),
            'angle_error_degrees': np.abs((q[:, 4]-m[:, 4]+np.pi) % (2*np.pi)-np.pi)*180/np.pi}
        def chunks(indices):
            episodes = np.searchsorted(data.ends, indices, side='right')
            ids = np.minimum(indices[:, None]+np.arange(8), data.ends[episodes, None]-1)
            return data.actions[ids].astype(np.float64)
        fields['action_chunk_rmse'] = np.sqrt(np.mean((chunks(source)-chunks(matched))**2, axis=(1, 2)))
        max_errors = {}
        independent_means = {}
        for name, values in fields.items():
            saved = np.array([float(r[name]) for r in rows])
            max_errors[name] = float(np.max(np.abs(values-saved)))
            # torch.cdist computes a squared-distance dot-product formula in FP32;
            # cancellation near zero can exceed relative-error tolerances. Record
            # actual error; 0.5/512 is a conservative numerical bound, not a score.
            tolerance = .5 if name in ['pusher_error', 'block_error'] else 1e-3
            np.testing.assert_allclose(saved, values, rtol=0, atol=tolerance)
            independent_means[name] = float(values.mean())
            np.testing.assert_allclose(saved.mean(), diagnostic[method]['retrieval'][name]['mean'], rtol=1e-10, atol=1e-10)
        result[method] = {'frames_checked': len(rows), 'max_abs_saved_fp32_minus_numpy_fp64': max_errors,
                          'independent_fp64_means': independent_means, 'csv_sha256': sha256(path)}
    save_json(rd/'independent_offline_verification.json', {'status': 'passed', 'methods': result,
              'verification_source_sha256': sha256(__file__),
              'note': 'Coordinate errors use world units; periodic angle degrees; same-episode 8-action RMSE.'}, immutable=True)


def final_verify(cfg, manifest, rd, heavy, meta):
    """Recompute aggregate results independently; no model or simulator execution."""
    from b1alt.common import PROJECT
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics = json.loads((rd/'eval_metrics.json').read_text())
    comparison = json.loads((rd/'paired_comparison.json').read_text())
    with (rd/'paired_scores.csv').open(newline='', encoding='utf-8') as f:
        pairs = list(csv.DictReader(f))
    assert [int(r['evaluation_seed']) for r in pairs] == manifest['test_env_seeds']
    baseline = np.array([float(r['baseline_score']) for r in pairs])
    b1 = np.array([float(r['b1_score']) for r in pairs])
    difference = b1-baseline
    np.testing.assert_allclose(difference, [float(r['b1_minus_baseline']) for r in pairs], atol=0, rtol=0)
    np.testing.assert_allclose(b1.mean(), metrics['mean_score'], rtol=0, atol=1e-15)
    np.testing.assert_allclose(b1.std(ddof=1), metrics['score_std'], rtol=0, atol=1e-15)
    rng = np.random.default_rng(2026)
    # Independent loop reproduces the matrix-bootstrap's ordered RNG draws.
    boot = np.array([difference[rng.integers(0, len(difference), size=len(difference))].mean() for _ in range(10000)])
    independent_ci = np.quantile(boot, [.025, .975])
    np.testing.assert_allclose(independent_ci, comparison['paired_bootstrap_95ci'], rtol=0, atol=1e-15)
    assets = json.loads((rd/'artifact_manifest.json').read_text())['files']
    for asset in assets:
        p = PROJECT/asset['path']
        assert p.stat().st_size == asset['bytes'] and sha256(p) == asset['sha256']
    assert sha256(rd/'resolved_config.yaml') == meta['resolved_config_sha256']
    with (rd/'training_metrics.csv').open(newline='', encoding='utf-8') as f:
        train = list(csv.DictReader(f))
    assert [int(r['epoch']) for r in train] == list(range(1, 41))
    assert int(train[-1]['optimizer_steps']) == 4720 == meta['optimizer_steps']
    for row in train:
        np.testing.assert_allclose(float(row['total']), float(row['contrast'])+float(row['pose']), atol=1e-6, rtol=0)
        np.testing.assert_allclose(float(row['pose']), float(row['xy'])+float(row['angle']), atol=1e-9, rtol=0)
    # Separate scales make the small auxiliary-loss curves readable.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, keys in [(axes[0], ['contrast', 'total']), (axes[1], ['xy', 'angle', 'pose'])]:
        for key in keys:
            ax.plot([int(r['epoch']) for r in train], [float(r[key]) for r in train], label=key)
        ax.set(xlabel='epoch', ylabel='mean loss')
        ax.legend()
    axes[0].set_title('Contrastive and total loss')
    axes[1].set_title('Auxiliary losses (log scale)')
    axes[1].set_yscale('log')
    fig.tight_layout()
    target = rd/'pose_loss_detail.png'
    if target.exists():
        raise ValueError('Final verification figure exists; preserve completed verification')
    fig.savefig(target, dpi=130)
    plt.close(fig)
    output = {'status': 'passed', 'artifact_files_rehashed': len(assets), 'paired_seeds': len(pairs),
              'training_epochs': len(train), 'optimizer_steps': 4720,
              'independent_paired_ci': independent_ci.tolist(),
              'config_checksum_verified': True, 'verification_source_sha256': sha256(__file__)}
    save_json(rd/'final_verification.json', output, immutable=True)
    print(json.dumps(output), flush=True)
