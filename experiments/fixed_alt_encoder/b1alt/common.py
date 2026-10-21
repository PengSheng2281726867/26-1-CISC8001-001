"""Path resolution and frozen-contract checks; never generate a split."""
import json
from pathlib import Path
import yaml
from pusht.repro import load_config, load_split, sha256
from pusht.experiment import validate_method, verify_references

EXPERIMENT = Path(__file__).resolve().parents[1]
PROJECT = EXPERIMENT.parents[1]
DEFAULT_CONFIG = EXPERIMENT / 'configs/b1_pose_alt30_seed42.yaml'


def config(path=DEFAULT_CONFIG):
    settings = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    cfg = load_config(PROJECT / settings['baseline_config'], PROJECT)
    baseline_dir = PROJECT / 'experiments/runs' / settings['baseline_run']
    baseline = yaml.safe_load((baseline_dir / 'resolved_config.yaml').read_text())
    if cfg != baseline:
        raise ValueError('Current baseline config differs from original ALT-30 resolved config')
    if settings['b1'] != {
        'pose_hidden_dim': 64, 'pose_output_dim': 4, 'lambda_pose': 1.0,
        'target': '[block_x/256-1, block_y/256-1, sin(theta), cos(theta)]',
        'pose_loss': 'average_two_views_xy_MSE_plus_sincos_MSE',
        'checkpoint_selection': 'final_epoch'}:
        raise ValueError('This pilot has a fixed B1 definition; changes need a separate experiment')
    cfg['b1'] = settings['b1']
    cfg['diagnostics'] = settings['diagnostics']
    cfg['baseline_run'] = settings['baseline_run']
    cfg['artifacts_root'] = str(PROJECT / settings['artifacts_root'])
    cfg['baseline_config_sha256'] = sha256(baseline_dir / 'resolved_config.yaml')
    return cfg


def contract(cfg):
    m = load_split(cfg['split']['manifest'], cfg['paths']['dataset'], 30)
    validate_method(cfg, m)
    verify_references(cfg, PROJECT)
    baseline_dir = PROJECT / 'experiments/runs' / cfg['baseline_run']
    ref = json.loads((baseline_dir / 'dataset_manifest_ref.json').read_text())
    baseline_meta = json.loads((baseline_dir / 'run_metadata.json').read_text())
    actual = {'path': cfg['split']['manifest'], 'sha256': sha256(cfg['split']['manifest']),
              'dataset_fingerprint': m['dataset_fingerprint'],
              'train_episode_ids': m['train_subsets']['30'],
              'validation_episode_ids': m['heldout_validation_episode_ids']}
    if actual != ref or m['test_env_seeds'] != baseline_meta['test_env_seeds']:
        raise ValueError('Frozen data/seed contract differs from original ALT-30')
    if len(actual['train_episode_ids']) != 30 or len(actual['validation_episode_ids']) != 20:
        raise ValueError('Unexpected demonstration counts')
    if len(m['test_env_seeds']) != 50:
        raise ValueError('Expected all 50 frozen test seeds')
    return m, actual
