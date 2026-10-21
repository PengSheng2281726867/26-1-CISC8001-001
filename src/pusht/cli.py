"""Project command line; every failure is preserved with traceback and command."""
import argparse
import json
from pathlib import Path
import sys
import traceback
import urllib.request
from pusht.repro import load_config, now, save_json, sha256


def prepare_weights(cfg, project):
    path = Path(cfg['paths']['pretrained'])
    path.parent.mkdir(parents=True, exist_ok=True)
    url = 'https://download.pytorch.org/models/resnet18-f37072fd.pth'
    if not path.exists():
        partial = path.with_suffix('.download')
        print('Downloading official ImageNet ResNet18 weights...', flush=True)
        with urllib.request.urlopen(url, timeout=60) as response, partial.open('wb') as out:
            while block := response.read(1024 * 1024):
                out.write(block)
        if not sha256(partial).startswith('f37072fd'):
            raise ValueError('Official pretrained weight checksum prefix mismatch')
        partial.rename(path)
    if not sha256(path).startswith('f37072fd'):
        raise ValueError('Existing pretrained weights failed checksum')
    save_json(Path(project)/'results/pretrained_weights.json',
              {'url': url, 'sha256': sha256(path), 'bytes': path.stat().st_size,
               'weights': 'ResNet18_Weights.IMAGENET1K_V1', 'path': str(path)}, immutable=True)
    print('Verified ImageNet weights:', sha256(path), flush=True)


def main(project):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['audit', 'prepare-weights', 'check'])
    parser.add_argument('--config', default=str(Path(project)/'configs/alt_pusht_base.yaml'))
    parser.add_argument('--run-id')
    parser.add_argument('--scale', type=int)
    parser.add_argument('--model-seed', type=int)
    args = parser.parse_args()
    cfg = load_config(args.config, project)
    if args.scale is not None:
        cfg['training']['scale'] = args.scale
    if args.model_seed is not None:
        cfg['training']['model_seed'] = args.model_seed
        cfg['training']['shuffle_seed'] = args.model_seed + 1
        cfg['training']['augmentation_seed'] = args.model_seed + 2
    try:
        if args.command == 'audit':
            from pusht.audit import audit
            audit(cfg, project)
        elif args.command == 'prepare-weights':
            prepare_weights(cfg, project)
        elif args.command == 'check':
            import unittest
            import os
            os.environ['ALT_CONFIG'] = str(args.config)
            suite = unittest.defaultTestLoader.discover(str(Path(project)/'tests'))
            result = unittest.TextTestRunner(verbosity=2).run(suite)
            save_json(Path(project)/'results/core_validation.json', {'checked_at': now(), 'status': 'passed' if result.wasSuccessful() else 'failed', 'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'failure_details': [str(test)+': '+detail for test, detail in result.failures+result.errors], 'device': cfg['training']['device'], 'pretrained_sha256': sha256(cfg['paths']['pretrained'])})
            if not result.wasSuccessful():
                raise RuntimeError(f'Core gate failed: {len(result.failures)} failures, {len(result.errors)} errors')
        else:
            from pusht.experiment import dispatch
            dispatch(args, cfg, project)
    except Exception:
        folder = Path(project)/'docs'
        folder.mkdir(exist_ok=True)
        with (folder/'failure_log.md').open('a', encoding='utf-8') as f:
            f.write('\n## '+now()+'\n\nCommand: `'+ ' '.join(sys.argv)+'`\n\n')
            f.write('Config: `'+args.config+'`; model seed: '+str(cfg['training']['model_seed'])+'\n\n```text\n')
            f.write(traceback.format_exc()+'```\n')
        raise
