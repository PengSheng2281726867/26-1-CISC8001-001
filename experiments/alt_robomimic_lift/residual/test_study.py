"""Check quaternion convention, episode masking, action alignment, and gripper bypass."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from study import Head, correct, features, online, pairs


class ResidualTests(unittest.TestCase):
    def setUp(self):
        self.prop = np.zeros((2, 9), dtype=np.float32)
        self.prop[:, 6] = 1
        self.z = np.eye(2, 128, dtype=np.float32)
        self.bank = SimpleNamespace(embeddings=self.z, proprio=self.prop,
                                    episode_ids=np.array(["A", "B"]),
                                    local_frame_indices=np.array([1, 0]))

    def test_leave_episode_out_and_same_index_label(self):
        commands = {"A": np.zeros((2, 7)), "B": np.zeros((1, 7))}
        commands["A"][1, 0], commands["B"][0, 0] = 0.5, -0.25
        p = SimpleNamespace(bank=self.bank, actions_by_episode=commands)
        x, label, rows = pairs(self.bank, p, commands)
        np.testing.assert_array_equal(rows, [1, 0])
        np.testing.assert_allclose(label[:, 0], [0.75, -0.75])
        self.assertEqual(x.shape, (2, 143))
        np.testing.assert_allclose(x[:, -7], [-0.25, 0.5])

    def test_world_rotation_and_quaternion_sign(self):
        query = self.prop[:1].copy()
        reference = Rotation.from_euler("x", 80, degrees=True)
        change = Rotation.from_euler("y", 0.3)
        self.prop[0, 3:7] = reference.as_quat()
        query[0, 3:7] = -(change * reference).as_quat()
        x = features(self.z[:1], query, self.bank, [0], np.zeros((1, 7)))
        np.testing.assert_allclose(x[0, 131:134], [0, 0.3, 0], atol=1e-6)

    def test_initial_identity_and_gripper_bypass(self):
        prior = np.array([[0.4, -0.5, 0, 0.1, 0.5, 0, -1.]])
        with torch.no_grad():
            zero = Head()(torch.randn(1, 143)).numpy()
        np.testing.assert_array_equal(correct(prior, zero), prior)
        np.testing.assert_array_equal(correct(prior, np.full((1, 6), 3)), [[1, 1, 1, 1, 1, 1, -1]])

    def test_numeric_head_receives_only_relative_state_and_action(self):
        prior = np.arange(7, dtype=np.float32) / 10
        def retrieved(embedding, include_embedding):
            return prior, dict(bank_row=0, query_embedding=embedding, query_proprio=self.prop[0])
        policy = SimpleNamespace(bank=self.bank, act=retrieved)
        head = Head(15)
        seen = []
        head.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
        online(policy, head)([self.z[0], -self.z[0]])
        expected = np.tile(np.r_[np.zeros(8), prior], (2, 1))
        np.testing.assert_array_equal(seen[0].numpy(), expected)


if __name__ == "__main__":
    unittest.main()
