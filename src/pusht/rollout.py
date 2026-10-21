"""Instrumentation around the unmodified official PushTImageRunner.run method.

ObservedImageEnv only records base done/step count. Vector proxy only observes
results. Neither changes transitions, actions, rewards, seeds or termination.
"""
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
import diffusion_policy.env_runner.pusht_image_runner as runner_module


class ObservedImageEnv(PushTImageEnv):
    def reset(self):
        self.base_terminated = False
        self.base_steps = 0
        return super().reset()

    def step(self, action):
        result = super().step(action)
        self.base_terminated = bool(result[2])
        self.base_steps += 1
        return result


class VectorTrace:
    def __init__(self, env, seeds, policy, max_steps):
        self.env, self.seeds, self.policy, self.max_steps = env, seeds, policy, max_steps
        self.index = -1
        self.records = []
        self.video_paths = {}

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        self.index += 1
        self.start = time.perf_counter()
        return self.env.reset()

    def step(self, action):
        result = self.env.step(action)
        steps = int(self.env.call('get_attr', 'base_steps')[0])
        last_query = self.policy.query_records[-1]
        last_query['executed_actions'] = steps - last_query['current_step']
        if bool(np.all(result[2])):
            rewards = self.env.call('get_attr', 'reward')[0]
            terminated = bool(self.env.call('get_attr', 'base_terminated')[0])
            assert steps == len(rewards)
            self.records.append({'evaluation_seed': self.seeds[self.index],
                                 'official_episode_score': float(np.max(rewards)),
                                 'final_score': float(rewards[-1]), 'steps': steps,
                                 'terminated': terminated,
                                 'truncated': bool(steps >= self.max_steps and not terminated),
                                 'runtime_seconds': time.perf_counter()-self.start})
        return result

    def render(self):
        paths = self.env.render()
        self.video_paths[self.seeds[self.index]] = paths[0]
        return paths


def run_official(policy, cfg, seeds, artifact_dir):
    if seeds != list(range(seeds[0], seeds[0]+len(seeds))):
        raise ValueError('Current official Runner adapter requires contiguous frozen seeds')
    Path(artifact_dir).mkdir(parents=True, exist_ok=True)
    e = cfg['evaluation']
    with patch.object(runner_module, 'PushTImageEnv', ObservedImageEnv):
        runner = runner_module.PushTImageRunner(
            output_dir=str(artifact_dir), n_train=0, n_train_vis=0,
            n_test=len(seeds), n_test_vis=min(e['video_count'], len(seeds)),
            test_start_seed=seeds[0], legacy_test=e['legacy_test'],
            max_steps=e['max_steps'], n_obs_steps=1, n_action_steps=8,
            n_envs=1, fps=e['fps'], past_action=False, tqdm_interval_sec=30)
    if runner.env_seeds != seeds:
        raise ValueError('Runner did not resolve frozen seed list')
    proxy = VectorTrace(runner.env, seeds, policy, e['max_steps'])
    runner.env = proxy
    try:
        started = time.perf_counter()
        official = runner.run(policy)  # exact upstream closed-loop/scoring implementation
        elapsed = time.perf_counter()-started
        if len(proxy.records) != len(seeds):
            raise ValueError('Missing evaluated episodes')
        for row in proxy.records:
            expected = official[f"test/sim_max_reward_{row['evaluation_seed']}"]
            if row['official_episode_score'] != float(expected):
                raise ValueError('Independent telemetry disagrees with official Runner score')
        if not np.isclose(np.mean([r['official_episode_score'] for r in proxy.records]),
                          official['test/mean_score'], rtol=0, atol=1e-12):
            raise ValueError('Aggregate score mismatch')
        return proxy.records, proxy.video_paths, elapsed
    finally:
        proxy.env.close()
