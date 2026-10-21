"""Isolated runtime and provenance; never writes to either reference checkout."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
os.environ['WANDB_MODE'] = 'offline'
for key in ['TORCH_HOME', 'WANDB_DIR', 'WANDB_CACHE_DIR', 'WANDB_CONFIG_DIR',
            'MPLCONFIGDIR', 'NUMBA_CACHE_DIR']:
    os.environ[key] = str(HERE / 'artifacts' / 'cache' / key.lower())
sys.path.insert(0, str(PROJECT / 'src'))


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def git(root, *args):
    return subprocess.check_output(['git', '-c', f'safe.directory={Path(root).as_posix()}',
                                   '-c', f'core.excludesFile={PROJECT / ".gitignore"}',
                                   '-C', str(root), *args], text=True).strip()


def load():
    import yaml
    cfg = yaml.safe_load((HERE / 'config.yaml').read_text(encoding='utf-8'))
    base = yaml.safe_load((PROJECT / cfg['base_config']).read_text(encoding='utf-8'))
    manifest_path = PROJECT / cfg['split_manifest']
    if sha(manifest_path) != cfg['split_sha256']:
        raise ValueError('Frozen split changed')
    manifest = read(manifest_path)
    for key in cfg['evaluation']:
        if cfg['evaluation'][key] != base['evaluation'][key]:
            raise ValueError(f'Evaluation protocol diverges from original: {key}')
    if manifest['test_env_seeds'] != list(range(100000, 100050)):
        raise ValueError('Unexpected frozen evaluation seeds')
    if set(manifest['train_subsets']['30']) & set(manifest['heldout_validation_episode_ids']):
        raise ValueError('Train/validation overlap')
    sys.path.insert(0, str(Path(base['paths']['dp_source'])))
    return cfg, base, manifest


def provenance(cfg, base, manifest):
    from importlib.metadata import version
    from pusht.repro import tree_fingerprint
    fp = tree_fingerprint(base['paths']['dataset'])
    if fp['sha256'] != manifest['dataset_fingerprint']:
        raise ValueError('Dataset content changed')
    return {'created_at': now(), 'project_commit_at_start': git(PROJECT, 'rev-parse', 'HEAD'),
            'project_status_at_start': git(PROJECT, 'status', '--porcelain=v1'),
            'config_sha256': sha(HERE / 'config.yaml'),
            'base_config_sha256': sha(PROJECT / cfg['base_config']),
            'split_manifest_sha256': cfg['split_sha256'], 'dataset': fp,
            'train_episode_ids': manifest['train_subsets']['30'],
            'validation_episode_ids': manifest['heldout_validation_episode_ids'],
            'evaluation_seeds': manifest['test_env_seeds'],
            'code_files': {p.name: sha(p) for p in sorted(HERE.glob('*.py'))},
            'references': {k: {'path': base['paths'][k],
                               'commit': git(base['paths'][k], 'rev-parse', 'HEAD'),
                               'dirty_status': git(base['paths'][k], 'status', '--porcelain=v1')}
                           for k in ['alt_source', 'dp_source']},
            'input': 'privileged current pusher_xy + block_xy + block_theta; no network',
            'training': None, 'model_seed': None, 'python': sys.version,
            'hardware': {'device': 'CPU only', 'processor': __import__('platform').processor()},
            'timing_method': 'time.perf_counter; includes synchronous state-read IPC separately',
            'packages': {key: version(key) for key in ['numpy', 'torch', 'gym', 'zarr', 'pymunk', 'pygame', 'opencv-python', 'dill']}}


def stats(x):
    import numpy as np
    x = np.asarray(x, dtype=float)
    return {'mean': float(x.mean()), 'median': float(np.median(x)),
            'p05': float(np.quantile(x, .05)), 'p95': float(np.quantile(x, .95)),
            'max': float(x.max())}
