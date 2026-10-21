"""Reproducibility helpers. All paths are supplied by configuration."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_text(value):
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'


def save_json(path, value, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json_text(value)
    if immutable and path.exists():
        if path.read_text(encoding='utf-8') != content:
            raise ValueError(f'Refusing to overwrite immutable file: {path}')
        return
    path.write_text(content, encoding='utf-8')


def tree_fingerprint(root):
    """Full bytes: sorted relative POSIX path + NUL + file SHA256 + LF."""
    root = Path(root)
    h = hashlib.sha256()
    files = sorted(p for p in root.rglob('*') if p.is_file())
    total = 0
    for p in files:
        h.update((p.relative_to(root).as_posix() + '\0' + sha256(p) + '\n').encode())
        total += p.stat().st_size
    return {'sha256': h.hexdigest(), 'file_count': len(files), 'bytes': total,
            'algorithm': 'SHA256(sorted(relative_posix_path + NUL + file_sha256 + LF))',
            'full_byte_hash_computed': True}


def git(root, *args, check=True):
    root = str(Path(root).resolve())
    result = subprocess.run(['git', '-c', f'safe.directory={root}', '-C', root, *args],
                            capture_output=True, text=True,
                            env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'})
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()


def provenance(root):
    return {'path': str(Path(root).resolve()), 'commit': git(root, 'rev-parse', 'HEAD'),
            'branch': git(root, 'branch', '--show-current'),
            'dirty_status': git(root, 'status', '--porcelain=v1'),
            'remote': git(root, 'remote', 'get-url', 'origin'),
            'tracked_diff_sha256': hashlib.sha256(git(root, 'diff', 'HEAD').encode()).hexdigest()}


def configure_runtime(project):
    project = Path(project).resolve()
    sys.dont_write_bytecode = True
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    for key, folder in {'TORCH_HOME': 'torch', 'WANDB_DIR': 'wandb',
                        'WANDB_CACHE_DIR': 'wandb-cache', 'WANDB_CONFIG_DIR': 'wandb-config',
                        'MPLCONFIGDIR': 'matplotlib', 'NUMBA_CACHE_DIR': 'numba'}.items():
        os.environ[key] = str(project / 'artifacts' / 'cache' / folder)
    os.environ['WANDB_MODE'] = 'offline'
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'


def load_config(path, project):
    import yaml
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    for key, value in cfg['paths'].items():
        p = Path(value)
        cfg['paths'][key] = str((Path(project) / p).resolve() if not p.is_absolute() else p)
    cfg['split']['manifest'] = str((Path(project) / cfg['split']['manifest']).resolve())
    return cfg


def load_split(path, dataset=None, scale=None):
    import numpy as np
    p = Path(path)
    expected = p.with_suffix('.sha256').read_text().strip().split()[0]
    if sha256(p) != expected:
        raise ValueError('Frozen split checksum mismatch')
    m = json.loads(p.read_text(encoding='utf-8'))
    val, pool = m['heldout_validation_episode_ids'], m['ordered_train_pool_episode_ids']
    ids = val + pool
    if len(ids) != len(set(ids)) or sorted(ids) != list(range(m['episode_count_total'])):
        raise ValueError('Split overlap, duplicate or missing episodes')
    for n, subset in m['train_subsets'].items():
        if subset is not None and (len(subset) != int(n) or subset != pool[:int(n)]):
            raise ValueError('Training subsets must be exact nested prefixes')
    if scale is not None and m['train_subsets'].get(str(scale)) is None:
        raise ValueError(f'{scale}: NOT_FEASIBLE_WITH_CURRENT_DATA')
    if dataset is not None:
        if tree_fingerprint(dataset)['sha256'] != m['dataset_fingerprint']:
            raise ValueError('Dataset full-content fingerprint changed')
        import zarr
        ends = zarr.open(str(dataset), mode='r')['meta/episode_ends'][:]
        if not np.array_equal(ends, m['episode_ends']):
            raise ValueError('Episode boundaries changed')
    return m
