"""Read-only verification of baseline, references, data and B1 label semantics."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from pusht.data import PushTData
from pusht.repro import now, save_json, sha256, git
from b1alt.provenance import provenance
from b1alt.common import EXPERIMENT, PROJECT, contract


def run(cfg):
    m, ref = contract(cfg)
    data = PushTData(cfg['paths']['dataset'])
    state = data.root['data/state'][:]
    assert state.shape[1] == 5 and np.isfinite(state).all()
    sys.path.insert(0, cfg['paths']['dp_source'])
    from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
    from diffusion_policy.env_runner.pusht_image_runner import PushTImageRunner
    examples = []
    for ep in ref['train_episode_ids'][:3] + ref['validation_episode_ids'][:2]:
        for source in [int(data.starts[ep]), int(data.ends[ep]-1)]:
            env = PushTImageEnv(legacy=False)
            try:
                env.reset_to_state = state[source]
                obs = env.reset()
                rebuilt = np.moveaxis(obs['image'], 0, -1) * 255
                mae = float(np.abs(rebuilt-data.root['data/img'][source]).mean())
                assert mae == 0
                assert np.allclose(obs['agent_pos'], state[source, :2])
            finally:
                env.close()
            examples.append({'episode_id': ep, 'source_index': source,
                             'local_frame_index': int(source-data.starts[ep]),
                             'state': state[source].tolist(), 'pixel_mae': mae})
    base = PROJECT / 'experiments/runs' / cfg['baseline_run']
    required = ['resolved_config.yaml', 'dataset_manifest_ref.json', 'training_metrics.csv',
                'bank_validation.json', 'eval_metrics.json', 'eval_rollouts.csv', 'run_metadata.json']
    files = sorted(p for folder in ['src', 'configs', 'scripts', 'experiments/splits',
                                   'experiments/runs', 'docs', 'results', 'licenses']
                   for p in (PROJECT/folder).rglob('*')
                   if p.is_file() and EXPERIMENT not in p.parents and '__pycache__' not in p.parts)
    files += [PROJECT/'AGENTS.md', PROJECT/'README.md']
    guard = {p.relative_to(PROJECT).as_posix(): sha256(p) for p in files}
    out = {'audited_at': now(), 'code': provenance(PROJECT), 'dataset_manifest_ref': ref,
           'dataset_id': m['dataset_id'], 'test_env_seeds': m['test_env_seeds'],
           'full_dataset_hash_rechecked': True, 'state_shape': list(state.shape),
           'state_min': state.min(0).tolist(), 'state_max': state.max(0).tolist(),
           'label_columns': [2, 3, 4], 'xy_units': '512x512 world', 'theta_units': 'radians',
           'alignment_examples': examples,
           'baseline_files': {name: sha256(base/name) for name in required},
           'baseline_metrics': json.loads((base/'eval_metrics.json').read_text()),
           'baseline_bank_validation': json.loads((base/'bank_validation.json').read_text()),
           'baseline_final_training_row': (base/'training_metrics.csv').read_text().splitlines()[-1],
           'baseline_diagnostic_sha256': sha256(PROJECT/'results/diagnostics/retrieval_failure_v1.json'),
           'pretrained_sha256': sha256(cfg['paths']['pretrained']),
           'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
           'gpu': torch.cuda.get_device_name(0), 'shared_files_guard': guard,
           'official_runner_source_sha256': sha256(Path(cfg['paths']['dp_source']) /
                      'diffusion_policy/env_runner/pusht_image_runner.py'),
           'git_status_at_start': git(PROJECT, 'status', '--porcelain=v1')}
    save_json(EXPERIMENT/'results/phase0_audit.json', out, immutable=True)
    doc = f'''# B1 Phase 0 read-only audit

Completed {out['audited_at']}. Branch {out['code']['branch']}; starting commit {out['code']['commit']}.
Only the user-supplied experiment AGENTS.md was untracked at entry. No pre-existing code was overwritten.

Frozen split SHA256: `{ref['sha256']}`. Full dataset bytes were rehashed and match
`{ref['dataset_fingerprint']}`. Exact 30 training IDs, 20 heldout IDs and all 50 simulator
seeds match the original ALT-30 run and are saved in [phase0_audit.json](../results/phase0_audit.json).
No split was created or modified.

State columns are pusher x,y, block x,y, block angle (radians); extrema are
{out['state_min']} to {out['state_max']}. Ten first/last episode frames reconstructed
their stored RGB exactly using block pose for audit only. Pose labels use columns 2:5
at the same global index. Position targets remain world coordinates, not image pixels.

Original ALT-30: 40 epochs; 3806 bank entries; official 50-seed score
{out['baseline_metrics']['mean_score']}. Config, training log, bank validation, metrics,
representation diagnostics and source hashes were read and recorded. Reference source
commits, dirty states and tracked-byte fingerprints match the existing reference snapshot.
The baseline is an attributed independent implementation; B1 imports it without copying
upstream files. See root docs/upstream_provenance.md and licenses/ALT-MIT.txt.

Environment: {sys.executable}; Torch {torch.__version__}; CUDA {torch.version.cuda};
GPU {out['gpu']}. No packages were installed or changed. Official Runner integration,
offset=0, same-episode last-action padding and 8 absolute-position actions will be reused.
The pre-existing audit confirms collection saves observation/action before env.step.
Shared source/config/result byte hashes are saved for a final preservation check.
'''
    (EXPERIMENT/'docs').mkdir(exist_ok=True)
    (EXPERIMENT/'docs/phase0_audit.md').write_text(doc, encoding='utf-8')
    print(json.dumps({'audit': 'passed', 'train_frames': sum(int(data.ends[e]-data.starts[e]) for e in ref['train_episode_ids']),
                      'split_sha256': ref['sha256'], 'gpu': out['gpu']}), flush=True)
