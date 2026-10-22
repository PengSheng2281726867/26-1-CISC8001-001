"""Path portability checks for Windows-authored run manifests consumed in WSL."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from alt_lift.repro import resolve_config_path
sys.path.insert(0, str(HERE / "scripts"))
from evaluate_alt_lift import reset_to_frozen_seed, _array_hash, _policy_obs_hash, classify_episode_end
import numpy as np


class RuntimePathTests(unittest.TestCase):
    def test_episode_end_classification_handles_fixed_horizon(self):
        self.assertEqual(classify_episode_end(False, 1000, 1000), (False, True))
        self.assertEqual(classify_episode_end(True, 999, 1000), (True, False))
        self.assertEqual(classify_episode_end(True, 1000, 1000), (False, True))

    def test_relative_windows_separators_resolve_inside_experiment(self):
        resolved = resolve_config_path("artifacts\\runs\\lift30\\memory_bank.npz", base=HERE)
        self.assertEqual(resolved, HERE / "artifacts" / "runs" / "lift30" / "memory_bank.npz")

    def test_windows_absolute_paths_are_understood_in_active_runtime(self):
        resolved = resolve_config_path("D:\\Data\\lift\\image.hdf5", base=HERE)
        if sys.platform.startswith("linux"):
            self.assertEqual(resolved, Path("/mnt/d/Data/lift/image.hdf5"))
        else:
            self.assertEqual(str(resolved).lower(), "d:\\data\\lift\\image.hdf5")

    def test_seeded_reset_requires_exact_state_and_policy_observation(self):
        class FakeEnv:
            def reset(self):
                return {"agentview_image": np.zeros((2, 2, 3), dtype=np.uint8), "robot0_eye_in_hand_image": np.ones((2, 2, 3), dtype=np.uint8), "robot0_eef_pos": np.zeros(3, dtype=np.float32), "robot0_eef_quat": np.asarray([0, 0, 0, 1], dtype=np.float32), "robot0_gripper_qpos": np.zeros(2, dtype=np.float32)}
            def get_state(self):
                return {"states": np.asarray([0.0, 1.0], dtype=np.float64)}

        env = FakeEnv()
        obs = env.reset()
        fixture = {"sim_state_sha256": _array_hash(env.get_state()["states"]), "policy_observation_sha256": _policy_obs_hash(obs)}
        reset_obs = reset_to_frozen_seed(env, 100000, fixture)
        self.assertEqual(_policy_obs_hash(reset_obs), fixture["policy_observation_sha256"])
        fixture["sim_state_sha256"] = "mismatch"
        with self.assertRaisesRegex(RuntimeError, "state mismatch"):
            reset_to_frozen_seed(env, 100000, fixture)


if __name__ == "__main__":
    unittest.main()
