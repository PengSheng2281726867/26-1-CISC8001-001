"""Read-only privileged adapter around the unchanged official Runner.run.

Pattern follows src/pusht/rollout.py. DP subclasses are used under its MIT
license (see repository licenses/DP-MIT.txt); no DP source is changed.
"""
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
import diffusion_policy.env_runner.pusht_image_runner as runner_module


class OracleImageEnv(PushTImageEnv):
    def _state(self):
        return np.asarray([*self.agent.position, *self.block.position,
                           self.block.angle % (2*np.pi)], dtype=np.float64)

    def _get_obs(self):
        observation = super()._get_obs()
        self.observation_state_snapshot = self._state()
        np.testing.assert_array_equal(observation['agent_pos'], self.observation_state_snapshot[:2])
        return observation

    @property
    def oracle_snapshot(self):
        # A synchronous property read performs no simulation step and reads no reward.
        state = self._state()
        np.testing.assert_array_equal(state, self.observation_state_snapshot)
        return {'state': state, 'observation_state': self.observation_state_snapshot.copy(),
                'step': self.base_steps}

    def reset(self):
        self.base_steps = 0
        self.base_terminated = False
        return super().reset()

    def step(self, action):
        result = super().step(action)
        self.base_steps += 1
        self.base_terminated = bool(result[2])
        return result


class OracleVectorTrace:
    def __init__(self, env, seeds, policy, max_steps, sync_atol):
        self.env, self.seeds, self.policy = env, seeds, policy
        self.max_steps, self.sync_atol = max_steps, sync_atol
        self.index, self.records, self.video_paths = -1, [], {}
        self.sync_checks = 0
        self.max_pusher_sync_error = 0.

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _capture(self, obs):
        snapshot = self.env.call('get_attr', 'oracle_snapshot')[0]
        np.testing.assert_array_equal(snapshot['state'], snapshot['observation_state'])
        np.testing.assert_allclose(obs['agent_pos'][0, -1], snapshot['state'][:2],
                                   atol=self.sync_atol, rtol=0)
        self.latest_snapshot = snapshot
        self.latest_agent_pos = obs['agent_pos'][0, -1].copy()

    def read_current_state(self, observed_pusher):
        snapshot = self.env.call('get_attr', 'oracle_snapshot')[0]
        np.testing.assert_array_equal(snapshot['state'], self.latest_snapshot['state'])
        if snapshot['step'] != self.latest_snapshot['step']:
            raise ValueError('Oracle and observation timestamps disagree')
        np.testing.assert_array_equal(observed_pusher, self.latest_agent_pos.astype(observed_pusher.dtype))
        error = float(np.max(np.abs(observed_pusher-snapshot['state'][:2])))
        if error > self.sync_atol:
            raise ValueError('Oracle pusher disagrees with observation')
        self.sync_checks += 1
        self.max_pusher_sync_error = max(self.max_pusher_sync_error, error)
        return {'state': snapshot['state'], 'step': snapshot['step'],
                'seed': self.seeds[self.index], 'pusher_max_abs_error': error}

    def reset(self):
        self.index += 1
        self.started = time.perf_counter()
        obs = self.env.reset()
        self._capture(obs)
        return obs

    def step(self, action):
        result = self.env.step(action)
        self._capture(result[0])
        steps = int(self.latest_snapshot['step'])
        self.policy.finish_query(steps-self.policy.query_records[-1]['current_step'])
        if bool(np.all(result[2])):
            rewards = self.env.call('get_attr', 'reward')[0]
            terminated = bool(self.env.call('get_attr', 'base_terminated')[0])
            assert steps == len(rewards)
            self.records.append({'evaluation_seed': self.seeds[self.index],
                                 'official_episode_score': float(np.max(rewards)),
                                 'final_score': float(rewards[-1]), 'steps': steps,
                                 'terminated': terminated,
                                 'truncated': bool(steps >= self.max_steps and not terminated),
                                 'runtime_seconds': time.perf_counter()-self.started})
        return result

    def render(self):
        paths = self.env.render()
        self.video_paths[self.seeds[self.index]] = paths[0]
        return paths


def run_official(policy, cfg, seeds, artifact_dir):
    e = cfg['evaluation']
    if seeds != list(range(100000, 100050)) or list(policy.bank.weights) != [1., 1., 1.]:
        raise ValueError('Formal protocol/weights are frozen')
    Path(artifact_dir).mkdir(parents=True, exist_ok=True)
    with patch.object(runner_module, 'PushTImageEnv', OracleImageEnv):
        runner = runner_module.PushTImageRunner(
            output_dir=str(artifact_dir), n_train=0, n_train_vis=0,
            n_test=50, n_test_vis=4, test_start_seed=100000,
            legacy_test=e['legacy_test'], max_steps=e['max_steps'],
            n_obs_steps=1, n_action_steps=8, n_envs=1, fps=e['fps'],
            past_action=False, tqdm_interval_sec=30)
    if runner.env_seeds != seeds:
        runner.env.close()
        raise ValueError('Official Runner seed mismatch')
    proxy = OracleVectorTrace(runner.env, seeds, policy, e['max_steps'], cfg['diagnostics']['position_sync_atol'])
    policy.state_provider = proxy.read_current_state
    runner.env = proxy
    try:
        t0 = time.perf_counter()
        official = runner.run(policy)  # no changes to upstream loop or scoring
        elapsed = time.perf_counter()-t0
        assert len(proxy.records) == 50
        for row in proxy.records:
            assert row['official_episode_score'] == float(official[f"test/sim_max_reward_{row['evaluation_seed']}"])
        assert abs(np.mean([r['official_episode_score'] for r in proxy.records])-official['test/mean_score']) < 1e-12
        return proxy, elapsed
    finally:
        proxy.env.close()
