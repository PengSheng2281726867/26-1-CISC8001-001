"""Unit checks for the isolated Lift encoder, loss, preprocessing, and retrieval."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from alt_lift.data import ContrastiveViews, ProprioNormalizer, action_chunk, canonicalize_proprio
from alt_lift.encoder import VisualProprioEncoder
from alt_lift.loss import nt_xent_loss
from alt_lift.memory import CosineTop1, MemoryBank
from alt_lift.policy import ActionRetrievalPolicy


class AltCoreTests(unittest.TestCase):
    def test_quaternion_normalization_and_sign_are_stable(self):
        values = np.zeros((2, 9), dtype=np.float32)
        values[:, 3:7] = [[0, 0, 0, 2], [0, 0, 0, -4]]
        canonical = canonicalize_proprio(values)
        np.testing.assert_allclose(canonical[:, 3:7], [[0, 0, 0, 1], [0, 0, 0, 1]])

    def test_augmentation_views_keep_shape_and_numeric_state(self):
        rng = np.random.default_rng(2)
        raw = np.concatenate([rng.random((3, 3), dtype=np.float32), np.tile([0, 0, 0, 1], (3, 1)), rng.random((3, 2), dtype=np.float32)], axis=1)
        store = SimpleNamespace(
            agentview=rng.integers(0, 256, (3, 84, 84, 3), dtype=np.uint8),
            wrist=rng.integers(0, 256, (3, 84, 84, 3), dtype=np.uint8),
            proprio=raw,
            __len__=lambda self: 3,
        )
        # Dataset.__len__ uses len(store), so expose a minimal class instead of a plain namespace.
        class FakeStore:
            def __len__(self):
                return 3
        fake = FakeStore()
        fake.agentview, fake.wrist, fake.proprio = store.agentview, store.wrist, store.proprio
        normalizer = ProprioNormalizer.fit(raw)
        dataset = ContrastiveViews(fake, normalizer, {"brightness": 0.1, "contrast": 0.1, "saturation": 0.05, "hue": 0.02})
        view_a, view_b, prop = dataset[1]
        self.assertEqual(tuple(view_a.shape), (2, 3, 84, 84))
        self.assertEqual(tuple(view_b.shape), (2, 3, 84, 84))
        self.assertEqual(tuple(prop.shape), (9,))
        self.assertTrue(torch.isfinite(view_a).all())

    def test_shared_resnet_embedding_and_finite_nt_xent_backward(self):
        torch.manual_seed(3)
        model = VisualProprioEncoder("image_proprio", pretrained_path=None)
        # Both camera slots are routed through the single image_encoder module.
        self.assertEqual(sum(1 for name in model._modules if "encoder" in name and name.startswith("image")), 1)
        self.assertNotIn("agent_encoder", model._modules)
        self.assertNotIn("wrist_encoder", model._modules)
        images_a = torch.randn(2, 2, 3, 32, 32)
        images_b = torch.randn(2, 2, 3, 32, 32)
        proprio = torch.randn(2, 9)
        z_a, z_b = model.encode_pair(images_a, images_b, proprio)
        self.assertEqual(tuple(z_a.shape), (2, 128))
        self.assertTrue(torch.allclose(z_a.norm(dim=1), torch.ones(2), atol=1e-5))
        loss = nt_xent_loss(z_a, z_b, temperature=0.4)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.image_encoder.layer1[0].conv1.weight.grad)
        self.assertIsNotNone(model.proprio_encoder[0].weight.grad)
        self.assertIsNotNone(model.fusion[0].weight.grad)

    def test_memory_self_query_and_stable_top1(self):
        embeddings = np.zeros((3, 128), dtype=np.float32)
        embeddings[0, 0], embeddings[1, 1], embeddings[2, 0] = 1.0, 1.0, 1.0
        bank = MemoryBank(
            embeddings=embeddings,
            episode_ids=np.asarray(["demo_1", "demo_2", "demo_3"]),
            local_frame_indices=np.asarray([0, 0, 0], dtype=np.int32),
            source_indices=np.asarray([0, 1, 2], dtype=np.int64),
            proprio=np.zeros((3, 9), dtype=np.float32),
        )
        row, score = CosineTop1(bank, torch.device("cpu")).query(torch.nn.functional.pad(torch.tensor([1.0, 0.0]), (0, 126)))
        self.assertEqual(row, 0)  # deterministic first-row tie break
        self.assertAlmostEqual(score, 1.0, places=6)

    def test_action_chunks_pad_inside_episode_without_crossing(self):
        first = np.arange(21, dtype=np.float32).reshape(3, 7)
        chunk = action_chunk(first, 2, 4)
        self.assertEqual(chunk.shape, (4, 7))
        np.testing.assert_array_equal(chunk[0], first[2])
        np.testing.assert_array_equal(chunk[1:], np.repeat(first[-1:], 3, axis=0))
        with self.assertRaises(ValueError):
            action_chunk(first, 3, 1)

    def test_policy_returns_original_action_and_logs_only_current_inputs(self):
        class FixedEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
            def encode(self, images, prop):
                self.seen_shapes = (tuple(images.shape), tuple(prop.shape))
                return torch.nn.functional.pad(torch.tensor([[1.0, 0.0]], device=images.device), (0, 126))

        bank = MemoryBank(
            embeddings=np.pad(np.asarray([[1, 0], [0, 1]], dtype=np.float32), ((0, 0), (0, 126))),
            episode_ids=np.asarray(["demo_1", "demo_2"]),
            local_frame_indices=np.asarray([1, 0], dtype=np.int32),
            source_indices=np.asarray([1, 2], dtype=np.int64),
            proprio=np.zeros((2, 9), dtype=np.float32),
        )
        actions = {"demo_1": np.asarray([[0] * 7, [0.2, 0, 0, 0, 0, 0, -1]], dtype=np.float32), "demo_2": np.zeros((1, 7), dtype=np.float32)}
        norm = ProprioNormalizer(np.zeros(9, dtype=np.float32), np.ones(9, dtype=np.float32))
        policy = ActionRetrievalPolicy(FixedEncoder(), bank, norm, actions, torch.device("cpu"))
        obs = {
            "agentview_image": np.zeros((84, 84, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((84, 84, 3), dtype=np.uint8),
            "robot0_eef_pos": np.zeros(3, dtype=np.float32),
            "robot0_eef_quat": np.asarray([0, 0, 0, 1], dtype=np.float32),
            "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
            "object": np.asarray([999], dtype=np.float32),  # deliberately ignored
            "reward": 999,
        }
        action, info = policy.act(obs)
        np.testing.assert_array_equal(action, actions["demo_1"][1])
        self.assertEqual(info["episode_id"], "demo_1")
        self.assertEqual(policy.model.seen_shapes, ((1, 2, 3, 84, 84), (1, 9)))
        policy.reset()
        self.assertEqual(policy.query_count, 0)


if __name__ == "__main__":
    unittest.main()
