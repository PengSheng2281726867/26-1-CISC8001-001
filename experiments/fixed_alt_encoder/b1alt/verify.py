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
