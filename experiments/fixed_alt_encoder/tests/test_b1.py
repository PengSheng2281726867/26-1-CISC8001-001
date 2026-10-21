"""Data/gradient/isolation/API checks on real frozen Push-T frames."""
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from alt.encoder import FusionEncoder, make_augmentation
from alt.memory import build_bank, MemoryBank
from pusht.data import PushTData, ObservationFrames
from pusht.experiment import configure_torch
from pusht.policy import ALTPolicy
from b1alt.common import EXPERIMENT, config, contract
from b1alt.encoder import B1FusionEncoder
from b1alt.data_adapter import PoseFrames, pose_target
from b1alt.loss import losses


class B1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = config()
        cls.manifest, _ = contract(cls.cfg)
        cls.device = configure_torch(cls.cfg)
        cls.data = PushTData(cls.cfg['paths']['dataset'])
        cls.episodes = cls.manifest['train_subsets']['30']

    def fresh(self):
        configure_torch(self.cfg)
        return B1FusionEncoder.from_pretrained(self.cfg['paths']['pretrained']).to(self.device)

    def samples(self, n=4):
        f = PoseFrames(self.data, self.episodes[:1])
        return [torch.stack([f[i][j] for i in range(n)]).to(self.device) for j in range(3)]

    def test_shared_initialization_exact_and_inference_head_disabled(self):
        configure_torch(self.cfg)
        baseline = FusionEncoder.from_pretrained(self.cfg['paths']['pretrained']).to(self.device).eval()
        b1 = self.fresh().eval()
        for name, v in baseline.state_dict().items():
            self.assertTrue(torch.equal(v, b1.state_dict()[name]), name)
        image, pos, _ = self.samples()
        calls = []
        hook = b1.pose_head.register_forward_hook(lambda *args: calls.append(1))
        with torch.inference_mode():
            a, b = baseline(image, pos), b1(image, pos)
        hook.remove()
        self.assertEqual(calls, [])
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(b.shape, (4, 128))
        torch.testing.assert_close(b.norm(dim=-1), torch.ones(4, device=self.device))

    def test_pose_head_independent_of_numeric_pusher_and_single_image_encode(self):
        model = self.fresh().eval()
        image, pos, _ = self.samples()
        count = []
        hook = model.image_encoder.register_forward_hook(lambda *args: count.append(1))
        with torch.inference_mode():
            z1, p1 = model.forward_with_pose(image, pos)
            z2, p2 = model.forward_with_pose(image, torch.full_like(pos, 500))
        hook.remove()
        self.assertEqual(len(count), 2)
        self.assertTrue(torch.equal(p1, p2))
        self.assertFalse(torch.equal(z1, z2))

    def test_pose_only_gradient_reaches_backbone_projection_and_head(self):
        model = self.fresh().train()
        image, pos, target = self.samples()
        before = model.pose_head[-1].weight.detach().clone()
        _, pred = model.forward_with_pose(image, pos)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        for param in [model.image_encoder.conv1.weight, model.image_encoder.fc.weight, model.pose_head[-1].weight]:
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())
            self.assertGreater(float(param.grad.abs().sum()), 0)
        self.assertIsNone(model.position_encoder[0].weight.grad)
        self.assertIsNone(model.fusion[0].weight.grad)
        torch.optim.Adam(model.parameters(), lr=1e-4).step()
        self.assertFalse(torch.equal(before, model.pose_head[-1].weight))

    def test_exact_loss_weighting_and_periodic_target(self):
        target = pose_target([[256, 512, 0], [0, 256, 2*np.pi]])
        torch.testing.assert_close(target, torch.tensor([[0., 1., 0., 1.], [-1., 0., 0., 1.]]), atol=2e-7, rtol=0)
        z = torch.nn.functional.normalize(torch.arange(8, dtype=torch.float32).reshape(2, 4)+1, dim=-1)
        p1, p2 = torch.zeros(2, 4), torch.full((2, 4), 2.)
        out = losses(z, z, p1, p2, torch.ones(2, 4))
        self.assertEqual(float(out['xy']), 1.)
        self.assertEqual(float(out['angle']), 1.)
        self.assertEqual(float(out['pose']), 2.)
        torch.testing.assert_close(out['total'], out['contrast']+2)

    def test_labels_first_middle_last_all_episodes_and_preprocessing(self):
        frames = PoseFrames(self.data, self.episodes)
        ordinary = ObservationFrames(self.data, self.episodes)
        state = self.data.root['data/state'][:]
        for ep in self.episodes:
            offsets = np.flatnonzero((frames.indices >= self.data.starts[ep]) & (frames.indices < self.data.ends[ep]))
            for i in offsets[[0, len(offsets)//2, -1]]:
                image, pos, target = frames[i]
                source = int(frames.indices[i])
                torch.testing.assert_close(image, ordinary[i][0], rtol=0, atol=0)
                torch.testing.assert_close(pos, torch.tensor(state[source, :2]))
                torch.testing.assert_close(target, pose_target(state[source, 2:5]))
                np.testing.assert_array_equal(self.data.action_chunk(ep, source-self.data.starts[ep])[0][0], self.data.actions[source])
        self.assertFalse(set(self.episodes) & set(self.manifest['heldout_validation_episode_ids']))

    def test_two_view_real_minibatch_backward_finite(self):
        model = self.fresh().train()
        torch.manual_seed(44)
        frames = PoseFrames(self.data, self.episodes[:1], make_augmentation(self.cfg['augmentation']))
        a, b, pos, target = [torch.stack([frames[i][j] for i in range(4)]).to(self.device) for j in range(4)]
        self.assertFalse(torch.equal(a, b))
        z1, p1 = model.forward_with_pose(a, pos)
        z2, p2 = model.forward_with_pose(b, pos)
        out = losses(z1, z2, p1, p2, target)
        self.assertTrue(all(torch.isfinite(x) for x in out.values()))
        out['total'].backward()
        self.assertGreater(float(model.image_encoder.conv1.weight.grad.norm()), 0)

    def test_baseline_bank_policy_latest_frame_reset_and_no_pose_access(self):
        model = self.fresh().eval()
        bank = build_bank(model, self.data, self.episodes[:1], {'test': True}, batch_size=32)
        (EXPERIMENT/'artifacts').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=EXPERIMENT/'artifacts') as tmp:
            path = Path(tmp)/'bank.pt'
            bank.save(path)
            bank = MemoryBank.load(path)
        # Closed-loop policy receives a proxy without the original zarr/state labels.
        class ActionsOnly:
            starts = self.data.starts
            positions = self.data.positions
            action_chunk = self.data.action_chunk
        policy = ALTPolicy(model, bank, ActionsOnly(), [100000, 100001])
        policy.reset()
        f = ObservationFrames(self.data, self.episodes[:1])
        image, pos = f[4]
        obs = {'image': torch.stack([torch.zeros_like(image), image])[None],
               'agent_pos': torch.stack([torch.zeros_like(pos), pos])[None]}
        def forbidden(*args):
            raise AssertionError('Pose head must not execute at inference')
        hook = model.pose_head.register_forward_hook(forbidden)
        action = policy.predict_action(obs)['action']
        hook.remove()
        self.assertEqual(action.shape, (1, 8, 2))
        q = policy.query_records[-1]
        np.testing.assert_array_equal(action.cpu()[0], self.data.action_chunk(q['matched_episode_id'], q['local_frame_index'])[0])
        self.assertEqual(q['agent_pos'], pos.tolist())
        for ep, local in [(self.episodes[0], 0), (self.episodes[0], int(self.data.ends[self.episodes[0]]-self.data.starts[self.episodes[0]]-1))]:
            chunk, idx = self.data.action_chunk(ep, local)
            self.assertTrue(np.all(idx < self.data.ends[ep]))
            self.assertEqual(chunk.shape, (8, 2))
        policy.reset()
        self.assertEqual(policy.current_step, 0)
        self.assertIsNone(policy.previous_match)
        with self.assertRaises(ValueError):
            policy.predict_action({**obs, 'block_pose': torch.zeros(1, 3)})


if __name__ == '__main__':
    unittest.main()
