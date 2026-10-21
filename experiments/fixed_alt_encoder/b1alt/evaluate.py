"""Reuse original official evaluation function with explicit B1 model and bank."""
import json
from pusht.data import PushTData
from pusht.experiment import evaluate as official_evaluate
from pusht.repro import sha256, save_json
from b1alt.experiment import load_model, load_bank


def evaluate(cfg, manifest, rd, heavy, meta):
    offline = json.loads((rd/'offline_diagnostics.json').read_text())
    bank_check = json.loads((rd/'bank_validation.json').read_text())
    if offline['status'] != 'completed' or bank_check['status'] != 'passed':
        raise ValueError('Offline gate must pass before formal rollouts')
    if offline['checkpoint_sha256'] != meta['checkpoint_sha256'] or offline['bank_sha256'] != meta['bank_sha256']:
        raise ValueError('Offline/model/bank hashes differ')
    model, bank = load_model(cfg, heavy, meta), load_bank(heavy, meta)
    calls = []
    hook = model.pose_head.register_forward_hook(lambda *args: calls.append(1))
    try:
        official_evaluate(cfg, manifest, rd, heavy, meta, model=model,
                          data=PushTData(cfg['paths']['dataset']), bank=bank)
    finally:
        hook.remove()
    if calls:
        raise ValueError('Pose head executed during closed-loop inference')
    save_json(rd/'inference_isolation.json', {'pose_head_forward_calls': 0,
        'allowed_input_keys': ['image', 'agent_pos'], 'official_evaluator_reused': True,
        'checkpoint_sha256': meta['checkpoint_sha256'], 'bank_sha256': meta['bank_sha256'],
        'n_rollouts': len(meta['test_env_seeds'])}, immutable=True)
