"""Paired one-chunk action transfer from replayed Push-T oracle snapshots."""

import argparse
import csv
from importlib.metadata import version
from inspect import getfile
import json
from pathlib import Path
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent
ORACLE = HERE.parent / "geometry_oracle_alt"
sys.path.insert(0, str(ORACLE))
from common import PROJECT, git, now, read, save, sha
from geometry_bank import wrap
from run import setup


QUERY_STEPS = tuple(range(0, 300, 40))
ANGLE_DEADZONE = np.deg2rad(2)
SOURCE_RECORDS = {
    "oracle": ORACLE / "results",
    "visual": PROJECT / "experiments/runs/alt_030_split42_model42_pilot_v1",
}


def rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def align_actions(actions, source, current):
    return (actions - source[2:4]) @ rotation(current[4] - source[4]).T + current[2:4]


def snapshot(env):
    return np.r_[env._get_obs(), tuple(env.agent.velocity), tuple(env.block.velocity),
                 env.agent.angular_velocity, env.block.angular_velocity]


class FirstContact:
    def __init__(self, env):
        from shapely.geometry import Polygon
        from shapely.ops import unary_union

        self.env = env
        outline = unary_union([Polygon([tuple(p) for p in poly.get_vertices()])
                               for poly in env.block.shapes])
        vertices = np.asarray(outline.exterior.coords)
        self.edges = sorted(tuple(sorted((tuple(a), tuple(b))))
                            for a, b in zip(vertices[:-1], vertices[1:]))
        self.edge_points = np.asarray(self.edges)
        self.edge_vectors = self.edge_points[:, 1] - self.edge_points[:, 0]
        self.edge_lengths_squared = (self.edge_vectors**2).sum(axis=1)
        self.step = None
        self.first = None
        self.delegate = env.collision_handeler.post_solve
        env.collision_handeler.post_solve = self.post_solve

    def post_solve(self, arbiter, space, data):
        self.delegate(arbiter, space, data)
        if self.step is None or self.first is not None:
            return
        a, b = [shape.body for shape in arbiter.shapes]
        if {a, b} != {self.env.agent, self.env.block}:
            return
        contacts = arbiter.contact_point_set.points
        if not contacts:
            return
        point = contacts[0].point_a if a == self.env.block else contacts[0].point_b
        local = np.asarray(tuple(self.env.block.world_to_local(point)))
        offset = local - self.edge_points[:, 0]
        fraction = np.clip((offset * self.edge_vectors).sum(axis=1) / self.edge_lengths_squared, 0, 1)
        residual = offset - fraction[:, None] * self.edge_vectors
        edge = int(np.argmin((residual**2).sum(axis=1)))
        self.first = {"step": self.step, "local_point": local.tolist(), "edge": self.edges[edge]}


def reset(seed, legacy):
    from diffusion_policy.env.pusht.pusht_env import PushTEnv

    env = PushTEnv(legacy=legacy)
    env.seed(seed)
    env.reset()
    return env


def execute(env, chunk, detector=None):
    before = env._get_obs()
    rewards = []
    done = False
    for index, command in enumerate(chunk, start=1):
        if detector is not None:
            detector.step = index
        _, reward, done, _ = env.step(command)
        rewards.append(float(reward))
        if done:
            break
    if detector is not None:
        detector.step = None
    after = env._get_obs()
    return {"steps": len(rewards), "done": bool(done), "rewards": rewards,
            "best_score": max(rewards), "final_score": rewards[-1],
            "block_motion": float(np.linalg.norm(after[2:4] - before[2:4])),
            "angle_change": float(wrap(after[4] - before[4])),
            "first_contact": detector.first if detector is not None else None}


def verify_transform():
    source = np.array([0, 0, 10, 20, .3])
    current = np.array([0, 0, 90, 80, .3 + np.pi / 2])
    actions = np.array([[12., 20.], [10., 23.]])
    np.testing.assert_allclose(align_actions(actions, source, source), actions)
    np.testing.assert_allclose(align_actions(actions, source, current), [[90, 82], [87, 80]])
    np.testing.assert_allclose(align_actions(align_actions(actions, source, current), current, source), actions)


def run_pairs(seed, queries, official, bank, indices_by_source, source_policy, legacy, max_steps):
    baseline = reset(seed, legacy)
    detector = FirstContact(baseline)
    prefix = []
    rewards = []
    pairs = []
    for query in queries:
        step = query["current_step"]
        if step != len(prefix):
            raise ValueError("Noncontiguous replay prefix")
        current = baseline._get_obs()
        if source_policy == "oracle":
            np.testing.assert_allclose(current, query["current_state"], atol=1e-8, rtol=0)
        else:
            np.testing.assert_array_equal(current[:2].astype(np.float32), query["agent_pos"])
        chunk = np.array(query["commands"], dtype=np.float32)
        length = min(len(chunk), max_steps - step)
        selected = step in QUERY_STEPS
        if selected:
            index = indices_by_source[query["source_index"]]
            if source_policy == "oracle" and (index != query["bank_index"] or bank.query(current)[0] != index):
                raise ValueError("Recorded match differs from frozen geometric retrieval")
            bank_chunk, indices = bank.action_chunk(index)
            np.testing.assert_array_equal(chunk, bank_chunk)
            np.testing.assert_array_equal(indices, query["action_source_indices"])
            source = bank.states[index].astype(np.float64)
            if source_policy == "oracle":
                np.testing.assert_array_equal(source, query["matched_state"])
            else:
                np.testing.assert_array_equal(source[:2], query["matched_agent_pos"])
            recorded_episode = query.get("matched_episode", query.get("matched_episode_id"))
            recorded_frame = query.get("local_frame", query.get("local_frame_index"))
            if bank.episode_id[index] != recorded_episode or bank.frame_index[index] != recorded_frame:
                raise ValueError("Recorded demonstration episode or frame differs from the frozen bank")
            branch = reset(seed, legacy)
            try:
                for command in prefix:
                    if branch.step(command)[2]:
                        raise ValueError("Branch prefix terminated before selected snapshot")
                initial = snapshot(baseline)
                np.testing.assert_array_equal(initial, snapshot(branch))
                transformed = align_actions(chunk[:length], source, current)
                bounded = np.clip(transformed, branch.action_space.low, branch.action_space.high)
                aligned = execute(branch, bounded.astype(np.float32), FirstContact(branch))
            finally:
                branch.close()
            detector.first = None
        original = execute(baseline, chunk[:length], detector if selected else None)
        if original["steps"] != query["executed_actions"]:
            raise ValueError("Control executed steps differ from recorded replay")
        prefix.extend(chunk[:original["steps"]])
        rewards.extend(original["rewards"])
        if selected:
            following = int(bank.source_index[index]) + length
            end = bank.data.ends[bank.episode_id[index]]
            expert_turn = None
            if following < end:
                next_angle = bank.data.root["data/state"][following, 4]
                expert_turn = float(wrap(next_angle - source[4]))
            pairs.append({
                "seed": seed, "step": step, "source_index": int(bank.source_index[index]),
                "episode": int(bank.episode_id[index]), "frame": int(bank.frame_index[index]),
                "initial_snapshot": initial.tolist(), "expert_state": source.tolist(),
                "source_commands": chunk[:length].tolist(), "aligned_commands": bounded.tolist(),
                "original": original, "aligned": aligned, "expert_turn": expert_turn,
                "clipped_commands": int(np.any(transformed != bounded, axis=1).sum()),
            })
    baseline.close()
    if len(prefix) != int(official["steps"]):
        raise ValueError("Baseline episode length differs from official run")
    np.testing.assert_allclose([max(rewards), rewards[-1]],
                               [float(official["official_episode_score"]), float(official["final_score"])],
                               atol=1e-10, rtol=0)
    return pairs


def summarize(pairs, seeds):
    rows = {}
    for name in ("original", "aligned"):
        results = [pair[name] for pair in pairs]
        turns = [pair for pair in pairs if pair["expert_turn"] is not None
                 and pair["original"]["steps"] == pair["aligned"]["steps"] == 8
                 and abs(pair["expert_turn"]) >= ANGLE_DEADZONE]
        rows[name] = {
            "contact_count": sum(result["first_contact"] is not None for result in results),
            "mean_best_score": float(np.mean([result["best_score"] for result in results])),
            "mean_final_score": float(np.mean([result["final_score"] for result in results])),
            "mean_block_motion": float(np.mean([result["block_motion"] for result in results])),
            "expert_turn_comparable": len(turns),
            "same_turn_direction": int(sum(pair["expert_turn"] * pair[name]["angle_change"] > 0
                                           and abs(pair[name]["angle_change"]) >= ANGLE_DEADZONE
                                           for pair in turns)),
        }
    deltas = np.array([pair["aligned"]["best_score"] - pair["original"]["best_score"] for pair in pairs])
    # Resample whole episodes so nearby branch snapshots are not treated as independent.
    grouped = [deltas[[pair["seed"] == seed for pair in pairs]] for seed in seeds]
    rng = np.random.default_rng(2026)
    indices = rng.integers(0, len(seeds), (10000, len(seeds)))
    totals = np.array([values.sum() for values in grouped])
    sizes = np.array([len(values) for values in grouped])
    boot = totals[indices].sum(axis=1) / sizes[indices].sum(axis=1)
    edges = [(pair["original"]["first_contact"], pair["aligned"]["first_contact"]) for pair in pairs]
    return {"pairs": len(pairs), "modes": rows,
            "aligned_minus_original_best_score": float(deltas.mean()),
            "seed_cluster_bootstrap_95ci": np.quantile(boot, [.025, .975]).tolist(),
            "improved": int((deltas > 1e-9).sum()), "regressed": int((deltas < -1e-9).sum()),
            "clipped_pairs": sum(pair["clipped_commands"] > 0 for pair in pairs),
            "both_contact_different_edge": sum(a is not None and b is not None and a["edge"] != b["edge"]
                                               for a, b in edges),
            "only_aligned_contact": sum(a is None and b is not None for a, b in edges),
            "only_original_contact": sum(a is not None and b is None for a, b in edges)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source", choices=SOURCE_RECORDS, default="oracle")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    verify_transform()
    cfg, _, manifest, bank = setup()
    from diffusion_policy.env.pusht.pusht_env import PushTEnv

    indices_by_source = {int(frame): index for index, frame in enumerate(bank.source_index)}
    source_dir = SOURCE_RECORDS[args.source]
    artifacts = read(source_dir / "artifact_manifest.json")
    if isinstance(artifacts, dict):
        artifacts = artifacts["files"]
    source = next(entry for entry in artifacts if Path(entry["path"]).name == "retrieval_full.jsonl")
    trace = PROJECT / source["path"]
    if sha(trace) != source["sha256"]:
        raise ValueError("Archived source trace checksum mismatch")
    with (source_dir / "eval_rollouts.csv").open(newline="", encoding="utf-8") as stream:
        official = {int(row["evaluation_seed"]): row for row in csv.DictReader(stream)}
    queries = {seed: [] for seed in manifest["test_env_seeds"]}
    with trace.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            queries[row["evaluation_seed"]].append(row)
    seeds = list(queries)[:1] if args.smoke else list(queries)
    output = HERE / "artifacts" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    pairs = []
    start = time.perf_counter()
    for seed in seeds:
        pairs.extend(run_pairs(seed, queries[seed], official[seed], bank, indices_by_source, args.source,
                               cfg["evaluation"]["legacy_test"], cfg["evaluation"]["max_steps"]))
    pair_path = output / "pairs.json"
    save(pair_path, pairs)
    report = summarize(pairs, seeds)
    report.update(status="completed", smoke=args.smoke, run_id=args.run_id, seeds=seeds, source_policy=args.source,
                  query_steps=list(QUERY_STEPS), source_trace_sha256=sha(trace),
                  official_scores_sha256=sha(source_dir / "eval_rollouts.csv"),
                  split_sha256=cfg["split_sha256"], dataset_fingerprint=manifest["dataset_fingerprint"],
                  code_sha256=sha(__file__), pair_trace_sha256=sha(pair_path), completed_at=now(),
                  project_head=git(PROJECT, "rev-parse", "HEAD"), environment_source_sha256=sha(getfile(PushTEnv)),
                  wall_seconds=time.perf_counter() - start,
                  versions={name: version(name) for name in ("numpy", "pymunk", "gym", "shapely")},
                  horizon=8, expert_turn_deadzone_degrees=2)
    save(output / "summary.json", report)
    if not args.smoke:
        tracked = HERE / "results" / f"{args.run_id}.json"
        if tracked.exists():
            raise FileExistsError(tracked)
        save(tracked, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
