"""Real-observation policy history/API and instrumented-environment parity."""
import os
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
from alt.encoder import FusionEncoder
from alt.memory import MemoryBank
from pusht.data import ObservationFrames, PushTData
from pusht.policy import ALTPolicy
from pusht.repro import load_config, load_split

PROJECT = Path(__file__).resolve().parents[1]
CFG = load_config(os.environ.get('ALT_CONFIG', PROJECT/'configs/alt_pusht_base.yaml'), PROJECT)
sys.path.insert(0, CFG['paths']['dp_source'])
from pusht.rollout import ObservedImageEnv
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv


class PolicyGate(unittest.TestCase):
    def test_latest_frame_only_reset_and_original_action_units(self):
        torch.set_num_threads(4)
        split = load_split(CFG['split']['manifest'])
        data = PushTData(CFG['paths']['dataset'])
        frames = ObservationFrames(data, split['train_subsets']['30'][:1])
        model = FusionEncoder.from_pretrained(CFG['paths']['pretrained']).to(CFG['training']['device']).eval()
        items = [frames[i] for i in [0, 10, 20, 30]]
        image = torch.stack([x[0] for x in items]).to(CFG['training']['device'])
        pos = torch.stack([x[1] for x in items]).to(CFG['training']['device'])
        with torch.inference_mode():
            vectors = model(image, pos).cpu()
        indices = frames.indices[[0, 10, 20, 30]]
        eps = np.searchsorted(data.ends, indices, side='right')
        bank = MemoryBank(vectors, eps, indices-data.starts[eps], indices, {})
        policy = ALTPolicy(model, bank, data)
        policy.reset()
        current = {'image': image[2:3, None], 'agent_pos': pos[2:3, None]}
        history = {'image': torch.stack([torch.zeros_like(image[2:3]), image[2:3]], dim=1),
                   'agent_pos': torch.stack([torch.zeros_like(pos[2:3]), pos[2:3]], dim=1)}
        a = policy.predict_action(current)['action']
        b = policy.predict_action(history)['action']
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        self.assertEqual(a.shape, (1, 8, 2))
        np.testing.assert_array_equal(a[0].cpu(), data.action_chunk(int(eps[2]), 20)[0])
        self.assertGreater(policy.query_records[-1]['top1_cosine'], 0.99999)
        self.assertEqual(policy.query_records[-1]['current_step'], 8)
        policy.reset()
        self.assertEqual(policy.current_step, 0)
        self.assertIsNone(policy.previous_match)
        with self.assertRaisesRegex(ValueError, 'only current'):
            policy.predict_action({**current, 'block_pose': torch.zeros(1, 1, 3)})

    def test_instrumentation_preserves_official_transitions(self):
        left, right = PushTImageEnv(legacy=True), ObservedImageEnv(legacy=True)
        try:
            for env in [left, right]:
                env.seed(100000)
            a, b = left.reset(), right.reset()
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
            for step in range(8):
                action = np.array([200.+step, 300.-step], dtype=np.float32)
                a, b = left.step(action), right.step(action)
                for key in a[0]:
                    np.testing.assert_array_equal(a[0][key], b[0][key])
                self.assertEqual(a[1], b[1])
                self.assertEqual(a[2], b[2])
                self.assertEqual(right.base_terminated, b[2])
                self.assertEqual(right.base_steps, step+1)
        finally:
            left.close()
            right.close()
