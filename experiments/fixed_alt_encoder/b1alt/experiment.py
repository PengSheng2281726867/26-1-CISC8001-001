"""Isolated run lifecycle; shares read-only baseline helpers, never its registry."""
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sys
import traceback
import torch
import yaml
from alt.memory import MemoryBank, build_bank
from pusht.data import PushTData
from pusht.experiment import configure_torch, event
from pusht.repro import now, save_json, sha256
from b1alt.provenance import provenance
from b1alt.common import EXPERIMENT, PROJECT, contract
from b1alt.encoder import B1FusionEncoder


def sources():
    files = [EXPERIMENT/'run.py'] + sorted((EXPERIMENT/'b1alt').glob('*.py')) + sorted((EXPERIMENT/'configs').glob('*'))
    return {p.relative_to(EXPERIMENT).as_posix(): sha256(p) for p in files}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def locations(cfg, run_id):
    if not run_id or not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('A unique safe run ID is required')
    return EXPERIMENT/'runs'/run_id, Path(cfg['artifacts_root'])/run_id


def create(cfg, run_id):
    for name in ['phase1_tests.json', 'phase1_smoke.json']:
        if json.loads((EXPERIMENT/'results'/name).read_text())['status'] != 'passed':
            raise ValueError('Phase 1 tests and official integration smoke must pass')
    manifest, ref = contract(cfg)
    rd, heavy = locations(cfg, run_id)
    if rd.exists() or heavy.exists():
        raise ValueError('Run ID exists; choose a fresh ID, never overwrite')
    # Check artifact permission before creating a partially initialized run.
    heavy.mkdir(parents=True)
    rd.mkdir(parents=True)
    (rd/'resolved_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=True), encoding='utf-8')
    save_json(rd/'dataset_manifest_ref.json', ref, immutable=True)
    baseline_meta = json.loads((PROJECT/'experiments/runs'/cfg['baseline_run']/'run_metadata.json').read_text())
    metadata = {'run_id': run_id, 'created_at': now(), 'status': 'running', 'stage': 'setup',
        'scope': 'B1_pose_supervised_ALT30_single_model_seed_pilot', 'scale': 30,
        'dataset_id': manifest['dataset_id'], 'dataset_fingerprint': manifest['dataset_fingerprint'],
        'dataset_manifest_ref': ref, 'split_manifest_sha256': ref['sha256'],
        'test_env_seeds': manifest['test_env_seeds'],
        'seeds': {'split': 42, 'model': 42, 'shuffle': 43, 'augmentation': 44, 'validation_augmentation': 45},
        'code_at_start': provenance(PROJECT), 'experiment_source_hashes': sources(),
        'source_provenance': baseline_meta['source_provenance'],
        'baseline_run': cfg['baseline_run'], 'architecture': cfg['encoder'], 'b1': cfg['b1'],
        'inputs': 'one latest RGB frame and current pusher xy ONLY; no block pose/prediction at inference',
        'image_preprocessing': 'same ObservationFrames: HWC integer /255 -> CHW96, ImageNet mean/std',
        'position_preprocessing': 'fixed x/256-1; no fitted normalizer',
        'label_only': 'same-index state[:,2:5], xy/256-1, sin/cos radians; both augmented views',
        'action_units': 'original absolute 512x512 world coordinates; offset0; same-episode repeat-last',
        'augmentation': cfg['augmentation'], 'hyperparameters': cfg['training'],
        'retrieval': cfg['retrieval'], 'evaluation': cfg['evaluation'],
        'software': {k: importlib.metadata.version(k) for k in
                     ['torch', 'torchvision', 'numpy', 'zarr', 'gym', 'pymunk', 'pygame', 'av', 'wandb']},
        'python': sys.version, 'platform': platform.platform(), 'cuda': torch.version.cuda,
        'hardware': torch.cuda.get_device_name(0), 'command': sys.argv,
        'pretrained_sha256': sha256(cfg['paths']['pretrained']),
        'resolved_config_sha256': sha256(rd/'resolved_config.yaml'), 'training_status': 'not_started',
        'timing': 'perf_counter; baseline CUDA-synchronized inference timing; epoch includes data+optimizer'}
    save_json(rd/'run_metadata.json', metadata)
    event(rd, 'setup', 'completed')
    return manifest, rd, heavy, metadata


def existing(cfg, run_id):
    rd, heavy = locations(cfg, run_id)
    meta = json.loads((rd/'run_metadata.json').read_text())
    saved_cfg = yaml.safe_load((rd/'resolved_config.yaml').read_text())
    if saved_cfg != cfg or sha256(rd/'resolved_config.yaml') != meta['resolved_config_sha256']:
        raise ValueError('Immutable resolved configuration differs')
    manifest, ref = contract(cfg)
    if ref != json.loads((rd/'dataset_manifest_ref.json').read_text()):
        raise ValueError('Frozen data contract differs')
    return manifest, rd, heavy, meta


def save_state(rd, heavy, meta):
    files = sorted(p for p in heavy.rglob('*') if p.is_file())
    save_json(rd/'artifact_manifest.json', {'backed_up_by_git': False, 'files': [
        {'path': p.relative_to(PROJECT).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in files]})
    save_json(rd/'run_metadata.json', meta)
    path = EXPERIMENT/'registry.csv'
    fields = ['run_id', 'status', 'stage', 'reason', 'mean_score']
    rows = list(csv.DictReader(path.open(newline='', encoding='utf-8'))) if path.exists() else []
    rows = [r for r in rows if r['run_id'] != meta['run_id']]
    rows.append({'run_id': meta['run_id'], 'status': meta['status'], 'stage': meta['stage'],
                 'reason': 'Prespecified B1 lambda1 seed42 pilot', 'mean_score': meta.get('mean_score', '')})
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fresh(cfg):
    device = configure_torch(cfg)
    return B1FusionEncoder.from_pretrained(cfg['paths']['pretrained'],
        pose_hidden_dim=cfg['b1']['pose_hidden_dim']).to(device)


def load_model(cfg, heavy, meta):
    path = heavy/'encoder_final.pt'
    if sha256(path) != meta['checkpoint_sha256']:
        raise ValueError('Checkpoint checksum differs')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if (saved['split_manifest_sha256'] != meta['split_manifest_sha256'] or
            saved['train_episode_ids'] != meta['dataset_manifest_ref']['train_episode_ids']):
        raise ValueError('Checkpoint data contract differs')
    model = B1FusionEncoder(pose_hidden_dim=cfg['b1']['pose_hidden_dim'])
    model.load_state_dict(saved['state_dict'], strict=True)
    return model.to(configure_torch(cfg)).eval()


def load_bank(heavy, meta):
    path = heavy/'memory_bank.pt'
    if sha256(path) != meta['bank_sha256']:
        raise ValueError('Bank checksum differs')
    bank = MemoryBank.load(path)
    if (bank.metadata['checkpoint_sha256'] != meta['checkpoint_sha256'] or
            bank.metadata['split_manifest_sha256'] != meta['split_manifest_sha256'] or
            bank.metadata['train_episode_ids'] != meta['dataset_manifest_ref']['train_episode_ids']):
        raise ValueError('Bank/model/data contract mismatch')
    return bank


def bank(cfg, manifest, rd, heavy, meta):
    if (heavy/'memory_bank.pt').exists():
        raise ValueError('Bank exists; cannot overwrite')
    model = load_model(cfg, heavy, meta)
    data = PushTData(cfg['paths']['dataset'])
    selected = manifest['train_subsets']['30']
    result = build_bank(model, data, selected, {'checkpoint_sha256': meta['checkpoint_sha256'],
             'split_manifest_sha256': meta['split_manifest_sha256'], 'train_episode_ids': selected})
    if set(result.episode_ids) != set(selected):
        raise ValueError('Unexpected bank episodes')
    result.save(heavy/'memory_bank.pt')
    meta.update(stage='bank_ready', bank_entries=len(result.embeddings), bank_sha256=sha256(heavy/'memory_bank.pt'))
    event(rd, 'bank', 'completed', entries=meta['bank_entries'])


def failure(command, rd=None, heavy=None, meta=None):
    error = traceback.format_exc()
    with (EXPERIMENT/'docs/failure_log.md').open('a', encoding='utf-8') as f:
        f.write(f'\n## {now()} — {command}\n\nCommand: `{sys.executable} '+
                ' '.join(sys.argv)+f'`\n\nRun: {None if meta is None else meta["run_id"]}; model seed42.\n\n```text\n{error}```\n')
    if rd is not None:
        event(rd, command, 'failed', error=error)
        # Preserve completion of earlier stages while marking this stage failed.
        meta.update(status='failed', failed_stage=command, error=error)
        save_state(rd, heavy, meta)
