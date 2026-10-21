import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load
CFG, BASE, MANIFEST = load()
import numpy as np
from geometry_bank import GeometryBank, components, wrap
from oracle_runner import OracleImageEnv, OracleVectorTrace
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv


def make_env():
    env = OracleImageEnv(legacy=True, render_size=96)
    env.seed(12345)
    return MultiStepWrapper(env, n_obs_steps=1, n_action_steps=8, max_episode_steps=300)


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = GeometryBank(BASE['paths']['dataset'], MANIFEST['train_subsets']['30'])

    def test_periodic_distance_and_weights(self):
        q = [1, 2, 3, 4, .01]
        s = np.array([[1, 2, 3, 4, 2*np.pi-.01], [513, 2, 3, 516, .01+np.pi]])
        np.testing.assert_allclose(components(q, s)[0], [0, 0, (.02/np.pi)**2], atol=1e-15)
        np.testing.assert_allclose(components(q, s)[1], [1, 1, 1], atol=1e-15)
        self.assertAlmostEqual(abs(wrap(2*np.pi+.1)), .1)

    def test_source_exact_and_all_action_boundaries(self):
        bank = self.bank
        self.assertEqual(bank.states.shape, (3806, 5))
        self.assertEqual(set(bank.episode_id), set(MANIFEST['train_subsets']['30']))
        self.assertFalse(set(bank.episode_id) & set(MANIFEST['heldout_validation_episode_ids']))
        source = bank.data.root['data/state'][:]
        np.testing.assert_array_equal(bank.states, source[bank.source_index])
        for i in range(len(bank.states)):
            chunk, idx = bank.action_chunk(i)
            e = bank.episode_id[i]
            expected = np.minimum(bank.source_index[i]+np.arange(8), bank.data.ends[e]-1)
            np.testing.assert_array_equal(idx, expected)
            np.testing.assert_array_equal(chunk, bank.data.root['data/action'].oindex[idx])
            self.assertEqual(chunk.shape, (8, 2))

    def test_self_retrieval_and_exact_tie_order(self):
        for i, s in enumerate(self.bank.states):
            j, distance, _ = self.bank.query(s)
            self.assertEqual(distance, 0.)
            expected = int(np.flatnonzero(np.all(self.bank.states == s, axis=1))[0])
            self.assertEqual(j, expected)
        bank = self.bank
        original = bank.states
        try:
            bank.states = np.repeat(original[:1], 2, axis=0)
            self.assertEqual(bank.query(original[0])[0], 0)
        finally:
            bank.states = original


class StateTests(unittest.TestCase):
    def test_read_only_adapter_transition_parity(self):
        for seed in [12345, 98765]:
            plain, oracle = PushTImageEnv(legacy=True), OracleImageEnv(legacy=True)
            try:
                plain.seed(seed)
                oracle.seed(seed)
                a, b = plain.reset(), oracle.reset()
                for key in a:
                    np.testing.assert_array_equal(a[key], b[key])
                for target in [np.array([256., 256.]), np.array([180., 330.])]:
                    for _ in range(8):
                        a, b = plain.step(target), oracle.step(target)
                        for key in a[0]:
                            np.testing.assert_array_equal(a[0][key], b[0][key])
                        self.assertEqual(a[1:3], b[1:3])
                        for key in a[3]:
                            np.testing.assert_array_equal(a[3][key], b[3][key])
                        snap = oracle.oracle_snapshot
                        np.testing.assert_array_equal(snap['state'], snap['observation_state'])
                        self.assertTrue(0 <= snap['state'][4] < 2*np.pi)
            finally:
                plain.close()
                oracle.close()

    def test_spawn_async_state_sync_reset_and_eight_actions(self):
        env = AsyncVectorEnv([make_env], context='spawn')
        class Recorder:
            query_records = [{'current_step': 0}]
            def finish_query(self, executed):
                self.executed = executed
        policy = Recorder()
        proxy = OracleVectorTrace(env, [12345], policy, 300, .0001)
        try:
            obs = proxy.reset()
            snap = proxy.read_current_state(obs['agent_pos'][0, -1])
            self.assertEqual(snap['step'], 0)
            obs, _, _, _ = proxy.step(np.full((1, 8, 2), 256., dtype=np.float32))
            snap = proxy.read_current_state(obs['agent_pos'][0, -1])
            self.assertEqual(snap['step'], 8)
            self.assertEqual(policy.executed, 8)
            with self.assertRaises(AssertionError):
                proxy.read_current_state(obs['agent_pos'][0, -1]+1)
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
