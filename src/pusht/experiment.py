"""Immutable experiment setup, fresh contrastive training and official evaluation."""
import copy
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import re
import sys
import time
import traceback
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from alt.encoder import FusionEncoder, make_augmentation
from alt.loss import nt_xent
from alt.memory import MemoryBank, build_bank
from pusht.audit import source_snapshot
from pusht.data import ObservationFrames, PushTData
from pusht.policy import ALTPolicy
from pusht.repro import load_split, now, provenance, save_json, sha256


def configure_torch(cfg):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    t = cfg['training']
    torch.set_num_threads(t['torch_threads'])
    torch.use_deterministic_algorithms(t['deterministic'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = t['deterministic']
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(t['model_seed'])
    np.random.seed(t['model_seed'])
    torch.manual_seed(t['model_seed'])
    if t['device'].startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Configured CUDA device is unavailable; no silent CPU fallback')
    return torch.device(t['device'])


def validate_method(cfg, manifest):
    if cfg['encoder'] != {'image_dim': 128, 'position_dim': 32, 'embedding_dim': 128,
            'image_size': 96, 'pretrained_weights': 'ResNet18_Weights.IMAGENET1K_V1',
            'fine_tune': True, 'position_transform': 'x / 256 - 1'}:
        raise ValueError('Unsupported architecture/preprocessing change')
    if cfg['retrieval'] != {'metric': 'cosine', 'top_k': 1, 'ood_enabled': False,
            'action_offset': 0, 'action_chunk_length': 8, 'boundary_policy': 'repeat_last_action'}:
        raise ValueError('Frozen retrieval method changed')
    e = cfg['evaluation']
    for k in ['max_steps', 'legacy_test', 'n_obs_steps', 'n_action_steps']:
        if e[k] != manifest[k]:
            raise ValueError(f'Frozen evaluation setting changed: {k}')
    if e['n_envs'] != 1:
        raise ValueError('This audited telemetry adapter evaluates one environment at a time')


def verify_references(cfg, project):
    baseline = json.loads((Path(project)/'results/reference_snapshot.json').read_text())
    for key in ['alt_source', 'dp_source']:
        current = source_snapshot(cfg['paths'][key])
        for field in ['commit', 'dirty_status', 'tracked_worktree_sha256']:
            if current[field] != baseline[key][field]:
                raise ValueError(f'Read-only reference changed: {key}/{field}')
    return baseline


def code_fingerprint(project):
    h = hashlib.sha256()
    files = sorted(list((Path(project)/'src').rglob('*.py')) +
                   list((Path(project)/'scripts').rglob('*.py')))
    for p in files:
        h.update((p.relative_to(project).as_posix()+'\0'+sha256(p)+'\n').encode())
    return h.hexdigest()


def registry(project, run_id, metadata, mean_score=''):
    path = Path(project)/'experiments/registry.csv'
    fields = ['run_id', 'status', 'scale', 'model_seed', 'reason', 'mean_score', 'notes']
    rows = list(csv.DictReader(path.open(newline='', encoding='utf-8')))
    rows = [r for r in rows if r['run_id'] != run_id]
    rows.append({'run_id': run_id, 'status': metadata['status'], 'scale': metadata['scale'],
                 'model_seed': metadata['seeds']['model'], 'reason': metadata['scope'],
                 'mean_score': mean_score, 'notes': metadata.get('stage', '')})
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def event(run_dir, stage, status, **extra):
    with (run_dir/'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps({'time': now(), 'stage': stage, 'status': status, **extra})+'\n')


def create_run(cfg, project, run_id, smoke=False):
    if not run_id or not re.fullmatch('[A-Za-z0-9_-]+', run_id):
        raise ValueError('Provide a new --run-id containing letters, digits, underscores or hyphens')
    project = Path(project)
    manifest = load_split(cfg['split']['manifest'], cfg['paths']['dataset'], cfg['training']['scale'])
    validate_method(cfg, manifest)
    upstream = verify_references(cfg, project)
    cfg = copy.deepcopy(cfg)
    if smoke:
        cfg['training']['epochs'] = 0
    run_dir = project/'experiments/runs'/run_id
    artifacts = project/'artifacts/runs'/run_id
    if run_dir.exists() or artifacts.exists():
        raise ValueError('Run ID already exists; never overwrite previous configs or artifacts')
    run_dir.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    text = yaml.safe_dump(cfg, sort_keys=True)
    (run_dir/'resolved_config.yaml').write_text(text, encoding='utf-8')
    selected = manifest['train_subsets'][str(cfg['training']['scale'])]
    ref = {'path': cfg['split']['manifest'], 'sha256': sha256(cfg['split']['manifest']),
           'dataset_fingerprint': manifest['dataset_fingerprint'],
           'train_episode_ids': selected, 'validation_episode_ids': manifest['heldout_validation_episode_ids']}
    save_json(run_dir/'dataset_manifest_ref.json', ref, immutable=True)
    t = cfg['training']
    metadata = {'run_id': run_id, 'created_at': now(), 'status': 'running', 'stage': 'setup',
                'scope': 'untrained_ImageNet_integration_smoke' if smoke else 'single_model_seed_scale_pilot',
                'scale': t['scale'], 'dataset_id': manifest['dataset_id'],
                'dataset_fingerprint': manifest['dataset_fingerprint'],
                'split_manifest_sha256': ref['sha256'], 'dataset_manifest_ref': ref,
                'test_env_seeds': manifest['test_env_seeds'][:1] if smoke else manifest['test_env_seeds'],
                'seeds': {'split': manifest['split_seed'], 'model': t['model_seed'],
                          'shuffle': t['shuffle_seed'], 'augmentation': t['augmentation_seed'],
                          'validation_augmentation': t['augmentation_seed']+1},
                'code_at_start': provenance(project), 'code_tree_sha256': code_fingerprint(project),
                'source_provenance': upstream, 'architecture': cfg['encoder'],
                'inputs': 'one latest RGB frame and current 2D agent_pos only',
                'image_preprocessing': 'HWC integer RGB /255 -> CHW; native96; ImageNet mean/std',
                'position_preprocessing': 'fixed x/256-1; no fitted normalizer',
                'action_units': 'original absolute 512x512 world target coordinates',
                'augmentation': cfg['augmentation'], 'hyperparameters': t,
                'retrieval': cfg['retrieval'], 'evaluation': cfg['evaluation'],
                'software': {k: importlib.metadata.version(k) for k in
                    ['torch', 'torchvision', 'numpy', 'zarr', 'gym', 'pymunk', 'pygame', 'av', 'wandb']},
                'python': sys.version, 'platform': platform.platform(),
                'hardware': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                'timing': 'perf_counter wall time; encoder/search/extraction CUDA-synchronized',
                'pretrained_sha256': sha256(cfg['paths']['pretrained']),
                'training_status': 'not_run' if smoke else 'running'}
    save_json(run_dir/'run_metadata.json', metadata)
    event(run_dir, 'setup', 'completed')
    registry(project, run_id, metadata)
    return cfg, manifest, run_dir, artifacts, metadata


def existing_run(project, run_id):
    if not run_id or not re.fullmatch('[A-Za-z0-9_-]+', run_id):
        raise ValueError('A valid existing --run-id is required')
    run_dir = Path(project)/'experiments/runs'/run_id
    metadata = json.loads((run_dir/'run_metadata.json').read_text())
    cfg = yaml.safe_load((run_dir/'resolved_config.yaml').read_text())
    ref = json.loads((run_dir/'dataset_manifest_ref.json').read_text())
    if sha256(cfg['split']['manifest']) != ref['sha256']:
        raise ValueError('Run split reference no longer matches frozen manifest')
    manifest = load_split(cfg['split']['manifest'], cfg['paths']['dataset'], metadata['scale'])
    if manifest['train_subsets'][str(metadata['scale'])] != ref['train_episode_ids']:
        raise ValueError('Run training episode IDs changed')
    validate_method(cfg, manifest)
    verify_references(cfg, project)
    return cfg, manifest, run_dir, Path(project)/'artifacts/runs'/run_id, metadata


def artifact_manifest(project, run_dir, artifacts):
    files = sorted(p for p in artifacts.rglob('*') if p.is_file() and p.suffix in ['.pt', '.mp4'])
    save_json(run_dir/'artifact_manifest.json', {'files': [
        {'path': p.relative_to(project).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha256(p)}
        for p in files], 'backed_up_by_git': False})


def fresh_model(cfg):
    device = configure_torch(cfg)
    return FusionEncoder.from_pretrained(cfg['paths']['pretrained'],
        image_dim=cfg['encoder']['image_dim'], position_dim=cfg['encoder']['position_dim'],
        embedding_dim=cfg['encoder']['embedding_dim']).to(device)


def train_model(cfg, manifest, run_dir, artifacts, metadata):
    event(run_dir, 'training', 'running')
    model = fresh_model(cfg)
    initial_conv = model.image_encoder.conv1.weight.detach().cpu().clone()
    data = PushTData(cfg['paths']['dataset'])
    episodes = manifest['train_subsets'][str(cfg['training']['scale'])]
    frames = ObservationFrames(data, episodes, make_augmentation(cfg['augmentation']))
    t = cfg['training']
    if t['optimizer'] != 'Adam' or t['num_workers'] != 0:
        raise ValueError('Audited pilot uses Adam and single-process loading')
    loader = DataLoader(frames, batch_size=t['batch_size'], shuffle=True,
                        generator=torch.Generator().manual_seed(t['shuffle_seed']), drop_last=True,
                        num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=t['learning_rate'])
    torch.manual_seed(t['augmentation_seed'])
    model.train()
    start = time.perf_counter()
    steps = 0
    with (run_dir/'training_metrics.csv').open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'mean_nt_xent', 'optimizer_steps', 'epoch_seconds'])
        writer.writeheader()
        for epoch in range(1, t['epochs']+1):
            loss_sum, batches = 0.0, 0
            epoch_start = time.perf_counter()
            for a, b, pos in loader:
                a, b, pos = a.to(next(model.parameters()).device), b.to(next(model.parameters()).device), pos.to(next(model.parameters()).device)
                optimizer.zero_grad(set_to_none=True)
                loss = nt_xent(model(a, pos), model(b, pos), t['temperature'])
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                if steps == 0:
                    grad = model.image_encoder.conv1.weight.grad
                    if not torch.isfinite(grad).all() or grad.abs().sum() == 0:
                        raise ValueError('Backbone did not receive finite nonzero contrastive gradients')
                    metadata['first_backbone_gradient_norm'] = float(grad.norm())
                optimizer.step()
                loss_sum += loss.item()
                batches += 1
                steps += 1
            row = {'epoch': epoch, 'mean_nt_xent': loss_sum/batches,
                   'optimizer_steps': steps, 'epoch_seconds': time.perf_counter()-epoch_start}
            writer.writerow(row)
            f.flush()
            print(json.dumps(row), flush=True)
    training_seconds = time.perf_counter()-start
    if torch.equal(initial_conv, model.image_encoder.conv1.weight.detach().cpu()):
        raise ValueError('Pretrained backbone did not change during fine-tuning')
    # Held-out diagnostic at final epoch only, no checkpoint selection or test feedback.
    validation = ObservationFrames(data, manifest['heldout_validation_episode_ids'],
                                   make_augmentation(cfg['augmentation']))
    val_sum = val_batches = 0
    model.eval()
    with torch.random.fork_rng(), torch.inference_mode():
        torch.manual_seed(t['augmentation_seed']+1)
        for a, b, pos in DataLoader(validation, batch_size=t['batch_size'], shuffle=False, drop_last=True):
            device = next(model.parameters()).device
            val_sum += nt_xent(model(a.to(device), pos.to(device)), model(b.to(device), pos.to(device)), t['temperature']).item()
            val_batches += 1
    save_json(run_dir/'validation_metrics.json', {'final_epoch_mean_nt_xent': val_sum/val_batches,
        'batches': val_batches, 'episode_ids': manifest['heldout_validation_episode_ids'],
        'checkpoint_selection_uses_validation': False}, immutable=True)
    path = artifacts/'encoder_final.pt'
    torch.save({'state_dict': model.state_dict(), 'encoder_config': cfg['encoder'],
                'split_manifest_sha256': metadata['split_manifest_sha256'],
                'train_episode_ids': episodes, 'pretrained_sha256': metadata['pretrained_sha256'],
                'optimizer_steps': steps, 'model_seed': t['model_seed']}, path)
    metadata.update(training_status='completed', stage='trained', optimizer_steps=steps,
                    train_frame_count=len(frames), training_seconds=training_seconds,
                    checkpoint_sha256=sha256(path), backbone_finetuning_verified=True,
                    validation_final_nt_xent=val_sum/val_batches)
    event(run_dir, 'training', 'completed', optimizer_steps=steps, seconds=training_seconds)
    return model, data


def load_model(cfg, artifacts, metadata):
    path = artifacts/'encoder_final.pt'
    if sha256(path) != metadata['checkpoint_sha256']:
        raise ValueError('Checkpoint checksum changed')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['split_manifest_sha256'] != metadata['split_manifest_sha256']:
        raise ValueError('Checkpoint split differs from run')
    model = FusionEncoder()
    model.load_state_dict(saved['state_dict'], strict=True)
    return model.to(configure_torch(cfg)).eval()


def create_bank(cfg, manifest, run_dir, artifacts, metadata, model=None, data=None):
    path = artifacts/'memory_bank.pt'
    if path.exists():
        raise ValueError('Bank already exists; do not overwrite')
    model = load_model(cfg, artifacts, metadata) if model is None else model
    data = PushTData(cfg['paths']['dataset']) if data is None else data
    episodes = manifest['train_subsets'][str(metadata['scale'])]
    start = time.perf_counter()
    bank = build_bank(model, data, episodes,
        {'split_manifest_sha256': metadata['split_manifest_sha256'],
         'checkpoint_sha256': metadata['checkpoint_sha256'], 'train_episode_ids': episodes})
    if set(bank.episode_ids.tolist()) != set(episodes):
        raise ValueError('Bank does not contain exact training subset')
    bank.save(path)
    metadata.update(stage='bank_ready', bank_entries=len(bank.embeddings), bank_sha256=sha256(path),
                    bank_creation_seconds=time.perf_counter()-start)
    event(run_dir, 'bank', 'completed', entries=len(bank.embeddings))
    return model, data, bank


def evaluate(cfg, manifest, run_dir, artifacts, metadata, model=None, data=None, bank=None):
    sys.path.insert(0, cfg['paths']['dp_source'])
    from pusht.rollout import run_official
    model = load_model(cfg, artifacts, metadata) if model is None else model
    data = PushTData(cfg['paths']['dataset']) if data is None else data
    if bank is None:
        if sha256(artifacts/'memory_bank.pt') != metadata['bank_sha256']:
            raise ValueError('Bank checksum changed')
        bank = MemoryBank.load(artifacts/'memory_bank.pt')
    if bank.metadata['checkpoint_sha256'] != metadata['checkpoint_sha256']:
        raise ValueError('Bank checkpoint differs from evaluated model')
    output = run_dir/'eval_metrics.json'
    if output.exists():
        raise ValueError('Evaluation already completed; do not overwrite')
    attempt = now().replace(':', '').replace('+', '_')
    heavy_eval = artifacts/'evaluation'/attempt
    policy = ALTPolicy(model, bank, data, metadata['test_env_seeds'])
    event(run_dir, 'evaluation', 'running', attempt=attempt)
    try:
        rows, videos, elapsed = run_official(policy, cfg, metadata['test_env_seeds'], heavy_eval)
    finally:
        heavy_eval.mkdir(parents=True, exist_ok=True)
        with (heavy_eval/'retrieval_full.jsonl').open('x', encoding='utf-8') as f:
            for row in policy.query_records:
                f.write(json.dumps(row, allow_nan=False)+'\n')
    for row in rows:
        row.update(run_id=metadata['run_id'], scale=metadata['scale'],
                   train_seed=metadata['seeds']['model'], split_manifest_sha256=metadata['split_manifest_sha256'])
        q = [r for r in policy.query_records if r['evaluation_seed'] == row['evaluation_seed']]
        row.update(query_count=len(q), episode_switches=sum(r['episode_switch'] for r in q),
                   min_cosine=min(r['top1_cosine'] for r in q),
                   mean_cosine=float(np.mean([r['top1_cosine'] for r in q])),
                   padded_queries=sum(r['boundary_padded'] for r in q))
    with (run_dir/'eval_rollouts.csv').open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Keep two complete auditable query records per episode in Git; full traces stay local.
    with (run_dir/'retrieval_sample.jsonl').open('x', encoding='utf-8') as f:
        for seed in metadata['test_env_seeds']:
            for q in [r for r in policy.query_records if r['evaluation_seed'] == seed][:2]:
                f.write(json.dumps(q, allow_nan=False)+'\n')
    scores = np.array([r['official_episode_score'] for r in rows])
    e = cfg['evaluation']
    rng = np.random.default_rng(e['bootstrap_seed'])
    bootstrap = scores[rng.integers(0, len(scores), (e['bootstrap_samples'], len(scores)))].mean(axis=1)
    metrics = {'status': 'completed', 'scope': metadata['scope'], 'n_rollouts': len(rows),
               'evaluation_seeds': metadata['test_env_seeds'], 'mean_score': float(scores.mean()),
               'score_std': float(scores.std(ddof=1)) if len(scores)>1 else None,
               'mean_score_bootstrap_95ci': np.quantile(bootstrap, [.025, .975]).tolist() if len(scores)>1 else None,
               'score_definition': 'official Runner mean(max_t clip(coverage_t/0.95,0,1))',
               'mean_final_score': float(np.mean([r['final_score'] for r in rows])),
               'total_environment_steps': sum(r['steps'] for r in rows),
               'evaluation_seconds': elapsed, 'invalid_events': policy.invalid_events,
               'query_count': len(policy.query_records), 'bank_entries': len(bank.embeddings),
               'encoder_parameters': sum(p.numel() for p in model.parameters()),
               'checkpoint_bytes': (artifacts/'encoder_final.pt').stat().st_size,
               'bank_bytes': (artifacts/'memory_bank.pt').stat().st_size,
               'latency_ms': {k: {'mean': float(np.mean([r[k] for r in policy.query_records])),
                                  'p50': float(np.median([r[k] for r in policy.query_records])),
                                  'p95': float(np.quantile([r[k] for r in policy.query_records], .95))}
                              for k in ['encoder_ms', 'search_ms', 'extraction_ms']},
               'video_paths_by_seed': {str(k): v for k, v in videos.items() if v is not None},
               'retrieval_full_path': str(heavy_eval/'retrieval_full.jsonl'),
               'bootstrap_seed': e['bootstrap_seed'], 'bootstrap_samples': e['bootstrap_samples']}
    save_json(output, metrics, immutable=True)
    metadata.update(status='completed', stage='evaluated', mean_score=metrics['mean_score'],
                    completed_at=now(), evaluation_seconds=elapsed)
    event(run_dir, 'evaluation', 'completed', mean_score=metrics['mean_score'])
    (run_dir/'report.md').write_text(
        f"# {metadata['run_id']}\n\nStatus: completed; scope: {metadata['scope']}.\n\n"
        f"Official mean score: {metrics['mean_score']:.6f}; SD: {metrics['score_std']}; "
        f"95% bootstrap CI: {metrics['mean_score_bootstrap_95ci']}; n={len(rows)}.\n\n"
        "Score = mean over frozen seeds of episode maximum clip(coverage/0.95,0,1). "
        "No binary success rate is introduced. Final score is the last actual reward.\n\n"
        f"Model seed {metadata['seeds']['model']}; split SHA256 {metadata['split_manifest_sha256']}; "
        f"code start commit {metadata['code_at_start']['commit']}.\n\n"
        f"Training time: {metadata.get('training_seconds', 0):.2f}s; evaluation time {elapsed:.2f}s. "
        f"Bank: {len(bank.embeddings)} entries; invalid events: {policy.invalid_events}.\n\n"
        "This is a single-model-seed pilot (or untrained integration smoke as labeled), "
        "not a multi-seed final comparison. Weights, bank, full traces and bounded videos are local ignored artifacts; "
        "their checksums are in artifact_manifest.json. Git alone does not back them up.\n", encoding='utf-8')
    print(json.dumps({'run_id': metadata['run_id'], 'mean_score': metrics['mean_score'],
                      'n_rollouts': len(rows), 'scope': metadata['scope']}), flush=True)


def dispatch(args, cfg, project):
    project = Path(project)
    if args.command == 'report':
        from pusht.report import report
        return report(project)
    run_dir = None
    metadata = None
    try:
        if args.command in ['train', 'smoke']:
            cfg, manifest, run_dir, artifacts, metadata = create_run(
                cfg, project, args.run_id, smoke=args.command == 'smoke')
            if args.command == 'train':
                train_model(cfg, manifest, run_dir, artifacts, metadata)
            else:
                model = fresh_model(cfg)
                path = artifacts/'encoder_final.pt'
                torch.save({'state_dict': model.state_dict(),
                            'split_manifest_sha256': metadata['split_manifest_sha256']}, path)
                metadata.update(checkpoint_sha256=sha256(path), training_seconds=0)
                model, data, bank = create_bank(cfg, manifest, run_dir, artifacts, metadata, model=model)
                evaluate(cfg, manifest, run_dir, artifacts, metadata, model, data, bank)
        else:
            cfg, manifest, run_dir, artifacts, metadata = existing_run(project, args.run_id)
            if args.command == 'bank':
                create_bank(cfg, manifest, run_dir, artifacts, metadata)
            elif args.command == 'evaluate':
                evaluate(cfg, manifest, run_dir, artifacts, metadata)
        artifact_manifest(project, run_dir, artifacts)
        save_json(run_dir/'run_metadata.json', metadata)
        registry(project, args.run_id, metadata, metadata.get('mean_score', ''))
    except Exception:
        if run_dir is not None and metadata is not None:
            metadata.update(status='failed', failed_at=now(), error=traceback.format_exc())
            save_json(run_dir/'run_metadata.json', metadata)
            event(run_dir, args.command, 'failed', error=metadata['error'])
            registry(project, args.run_id, metadata)
        raise
