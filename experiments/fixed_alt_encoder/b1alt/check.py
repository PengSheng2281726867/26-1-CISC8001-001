"""Persist unit-test report and a one-seed untrained official integration smoke."""
import copy
import io
import json
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from alt.memory import build_bank
from pusht.data import PushTData
from pusht.policy import ALTPolicy
from pusht.repro import now, save_json
from b1alt.common import EXPERIMENT, contract
from b1alt.experiment import fresh


def run(cfg, check_id='phase1'):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', check_id):
        raise ValueError('Use a fresh safe check ID')
    output = EXPERIMENT/'results'/f'{check_id}_tests.json'
    if output.exists():
        raise ValueError('Test report exists; preserve old report before choosing another check ID')
    suite = unittest.defaultTestLoader.discover(str(EXPERIMENT/'tests'), pattern='test_*.py')
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    print(stream.getvalue(), flush=True)
    report = {'time': now(), 'tests_run': result.testsRun, 'failures': len(result.failures),
              'errors': len(result.errors), 'output': stream.getvalue(), 'status': 'passed' if result.wasSuccessful() else 'failed'}
    save_json(output, report, immutable=True)
    if not result.wasSuccessful():
        raise RuntimeError('B1 gate tests failed; inspect preserved report')
    manifest, _ = contract(cfg)
    model = fresh(cfg).eval()
    data = PushTData(cfg['paths']['dataset'])
    bank = build_bank(model, data, manifest['train_subsets']['30'], {'scope': 'untrained_smoke'})
    policy = ALTPolicy(model, bank, data, manifest['test_env_seeds'][:1])
    settings = copy.deepcopy(cfg)
    settings['evaluation']['video_count'] = 0
    sys.path.insert(0, cfg['paths']['dp_source'])
    from pusht.rollout import run_official
    calls = []
    hook = model.pose_head.register_forward_hook(lambda *args: calls.append(1))
    rows, videos, elapsed = run_official(policy, settings, manifest['test_env_seeds'][:1], EXPERIMENT/'artifacts'/f'{check_id}_smoke')
    hook.remove()
    assert not calls and len(rows) == 1 and policy.invalid_events == 0
    save_json(EXPERIMENT/'results'/f'{check_id}_smoke.json', {'status': 'passed', 'scope': 'untrained integration only; not B1 result',
        'rows': rows, 'seconds': elapsed, 'query_count': len(policy.query_records), 'pose_head_calls': len(calls),
        'query_samples': policy.query_records[:2], 'only_first_frozen_seed': True}, immutable=True)
    print(json.dumps({'unit_tests': 'passed', 'smoke': 'passed', 'pose_head_calls': 0}), flush=True)
