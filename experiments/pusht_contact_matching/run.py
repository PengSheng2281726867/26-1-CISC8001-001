"""Compare two contact-aware retrieval rules using the unchanged Push-T Runner."""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ORACLE = HERE.parent / 'geometry_oracle_alt'
sys.path.insert(0, str(ORACLE))
from common import now, provenance, read, save, sha
from geometry_bank import components, wrap
from geometry_policy import GeometryPolicy
from run_horizon_ablation import write_csv

# The upstream experiment has an entry point named run.py too.
from run import setup


def local_pusher(states):
    offset = states[..., :2] - states[..., 2:4]
    theta = states[..., 4]
    c, s = np.cos(theta), np.sin(theta)
    return np.stack((c * offset[..., 0] + s * offset[..., 1],
                     -s * offset[..., 0] + c * offset[..., 1]), axis=-1)


class ContactBank:
    def __init__(self, base, settings, mode, outline, radius):
        self.base, self.settings, self.mode = base, settings, mode
        self.points = np.asarray(outline, dtype=float)
        self.starts = self.points[:-1]
        self.vectors = np.diff(self.points, axis=0)
        self.lengths = np.sum(self.vectors**2, axis=1)
        self.radius = radius
        self.local = local_pusher(base.states.astype(float))
        self.edges, self.gaps = self.geometry(self.local)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def geometry(self, local):
        offset = local[..., None, :] - self.starts
        fraction = np.clip(np.sum(offset * self.vectors, axis=-1) / self.lengths, 0, 1)
        distance = np.linalg.norm(offset - fraction[..., None] * self.vectors, axis=-1)
        return distance.argmin(axis=-1), distance.min(axis=-1) - self.radius

    def rank(self, state):
        local = local_pusher(state)
        edge, gap = self.geometry(local)
        delta = np.stack((np.linalg.norm(self.states[:, 2:4] - state[2:4], axis=1),
                          np.abs(np.rad2deg(wrap(self.states[:, 4] - state[4]))),
                          np.linalg.norm(self.local - local, axis=1),
                          np.abs(self.gaps - gap)), axis=-1)
        fields = ('block_position', 'angle_degrees', 'local_pusher', 'gap')
        scores = ((delta / [self.settings['scales'][f] for f in fields]) ** 2).sum(axis=1)
        eligible = (self.edges == edge) & (delta <= [self.settings['gate'][f] for f in fields]).all(axis=1)
        ranking = scores
        if self.mode == 'gated' and eligible.any():
            ranking = np.where(eligible, scores, np.inf)
        return delta, eligible, ranking

    def query(self, state):
        delta, eligible, scores = self.rank(state)
        index = int(scores.argmin())
        self.last = {'eligible_count': int(eligible.sum()), 'selected_passes_gate': bool(eligible[index]),
                     'fallback': bool(self.mode == 'gated' and not eligible.any()),
                     'local_pusher_error': float(delta[index, 2]), 'gap_error': float(delta[index, 3]),
                     'same_edge': bool(self.edges[index] == self.geometry(local_pusher(state))[0])}
        # Preserve the original policy's generic logging fields as old-distance diagnostics.
        terms = components(state, self.states[index], self.position_scale)
        return index, float(terms.sum()), terms


class ContactPolicy(GeometryPolicy):
    def __init__(self, bank, sink, settings):
        super().__init__(bank, sink=sink)
        self.settings = settings
        self.cases = []

    def predict_action(self, obs):
        result = super().predict_action(obs)
        row = self.query_records[-1]
        row.update(self.bank.last)
        if (row['evaluation_seed'] in self.settings['case_seeds']
                and row['current_step'] in self.settings['case_steps']):
            state = np.array(row['current_state'])
            delta, eligible, scores = self.bank.rank(state)
            old = self.bank.base.query(state)[0]
            selected = row['bank_index']
            order = np.argsort(scores, kind='stable')
            candidates = [selected, *[int(i) for i in order if i != selected and np.isfinite(scores[i])]]
            indices = [old, *candidates[:self.settings['candidate_images']]]
            frame = np.rint(obs['image'][0, -1].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            cases = []
            for index in indices:
                cases.append({'episode': int(self.bank.episode_id[index]), 'frame': int(self.bank.frame_index[index]),
                              'source': int(self.bank.source_index[index]), 'passes_gate': bool(eligible[index]),
                              'errors': delta[index].tolist(), 'state': self.bank.states[index].tolist()})
            self.cases.append((frame, dict(row), cases))
        return result


def panels(output, cases, bank):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output.mkdir(exist_ok=False)
    for frame, row, candidates in cases:
        images = [frame] + [bank.data.root['data/img'][c['source']].astype(np.uint8) for c in candidates]
        names = ['Current', 'Old rule', 'New match', 'Next candidate']
        fig, axes = plt.subplots(1, len(images), figsize=(3.1 * len(images), 4.2), squeeze=False)
        for i, (ax, image) in enumerate(zip(axes[0], images)):
            ax.imshow(image, interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(names[i])
            else:
                c = candidates[i - 1]
                block, angle, local, gap = c['errors']
                ax.set_title(f"{names[i]} | demo {c['episode']}, frame {c['frame']}")
                ax.set_xlabel(f"T: {block:.0f} units, {angle:.1f} deg\n"
                              f"Local pusher: {local:.0f}, gap: {gap:.0f}\n"
                              f"Passes gate: {'yes' if c['passes_gate'] else 'no'}")
        fig.suptitle(f"Seed {row['evaluation_seed']} | step {row['current_step']} | "
                     f"eligible {row['eligible_count']} | fallback {row['fallback']}")
        fig.tight_layout()
        stem = output / f"seed{row['evaluation_seed']}_step{row['current_step']:03d}"
        fig.savefig(stem.with_suffix('.png'), dpi=105, bbox_inches='tight')
        plt.close(fig)
        save(stem.with_suffix('.json'), {'query': row, 'candidates': candidates})


def offline(banks, manifest):
    rows = {name: [] for name in ('old', *banks)}
    relative = banks['relative']
    data = relative.data
    for episode in manifest['heldout_validation_episode_ids']:
        for state in data.root['data/state'][data.starts[episode]:data.ends[episode]].astype(float):
            delta, eligible, _ = relative.rank(state)
            matches = {'old': relative.base.query(state)[0]}
            matches.update({name: bank.query(state)[0] for name, bank in banks.items()})
            edge = relative.geometry(local_pusher(state))[0]
            for name, index in matches.items():
                rows[name].append((bool(eligible.any()), bool(eligible[index]), bool(relative.edges[index] == edge),
                                   *delta[index]))
    return {name: {'frames': len(values), 'has_candidate': int(np.sum(np.array(values)[:, 0])),
                   'selected_passes': int(np.sum(np.array(values)[:, 1])),
                   'same_edge': int(np.sum(np.array(values)[:, 2])),
                   'mean_errors': np.mean(np.array(values)[:, 3:], axis=0).tolist()}
            for name, values in rows.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    cfg, base, manifest, bank = setup()
    from diffusion_policy.env.pusht.pusht_env import PushTEnv
    from oracle_runner import run_official
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    settings = read(HERE / 'config.json')
    artifact = HERE / 'artifacts' / args.run_id
    output = HERE / 'results' / args.run_id
    artifact.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    env = PushTEnv()
    env.seed(0)
    env.reset()
    outline = unary_union([Polygon([tuple(point) for point in shape.get_vertices()])
                           for shape in env.block.shapes]).exterior.coords
    radius = next(iter(env.agent.shapes)).radius
    banks = {mode: ContactBank(bank, settings, mode, outline, radius) for mode in ('relative', 'gated')}
    env.close()
    np.testing.assert_allclose(local_pusher(np.array([11., 23., 11., 20., np.pi/2])), [3., 0.], atol=1e-12)
    prov = provenance(cfg, base, manifest)
    prov['study_config_sha256'] = sha(HERE / 'config.json')
    prov['study_code_sha256'] = sha(__file__)
    seeds = manifest['test_env_seeds']
    with (ORACLE / 'results/eval_rollouts.csv').open(newline='', encoding='utf-8') as stream:
        old = {int(row['evaluation_seed']): float(row['official_episode_score']) for row in csv.DictReader(stream)}
    scores_old = np.array([old[seed] for seed in seeds])
    indices = np.random.default_rng(2026).integers(0, len(seeds), (10000, len(seeds)))
    result = {'status': 'running', 'run_id': args.run_id, 'started': now(), 'provenance': prov,
              'settings': settings, 'old_mean': float(scores_old.mean()),
              'offline': offline(banks, manifest), 'rollouts': {}}
    save(output / 'summary.json', result)
    print(result['offline'], flush=True)
    for name, contact_bank in banks.items():
        folder = artifact / name
        folder.mkdir()
        with (folder / 'queries.jsonl').open('x', encoding='utf-8') as stream:
            policy = ContactPolicy(contact_bank, stream, settings)
            proxy, elapsed = run_official(policy, cfg, seeds, folder)
        if [r['evaluation_seed'] for r in proxy.records] != seeds:
            raise ValueError('Runner seeds do not match the frozen order')
        for row in policy.query_records:
            action, idx = bank.action_chunk(row['bank_index'])
            np.testing.assert_array_equal(action, row['commands'])
            np.testing.assert_array_equal(idx, row['action_source_indices'])
            if not 1 <= row['executed_actions'] <= 8:
                raise ValueError('Invalid executed chunk length')
        write_csv(output / f'{name}_rollouts.csv', proxy.records)
        panels(output / name, policy.cases, bank)
        scores = np.array([r['official_episode_score'] for r in proxy.records])
        diff = scores - scores_old
        result['rollouts'][name] = {
            'n': len(scores), 'mean': float(scores.mean()), 'sd': float(scores.std(ddof=1)),
            'mean_final': float(np.mean([r['final_score'] for r in proxy.records])),
            'delta_vs_old': float(diff.mean()), 'paired95': np.quantile(diff[indices].mean(axis=1), [.025, .975]).tolist(),
            'queries': len(policy.query_records),
            'with_candidate': sum(row['eligible_count'] > 0 for row in policy.query_records),
            'fallback': sum(row['fallback'] for row in policy.query_records),
            'same_edge': sum(row['same_edge'] for row in policy.query_records),
            'query_trace_sha256': sha(folder / 'queries.jsonl'),
            'case_count': len(policy.cases), 'seconds': elapsed,
        }
        print(name, result['rollouts'][name], flush=True)
        save(output / 'summary.json', result)
    result.update(status='completed', finished=now())
    save(output / 'summary.json', result)


if __name__ == '__main__':
    main()
