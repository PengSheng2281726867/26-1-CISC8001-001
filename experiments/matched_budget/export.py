"""Verify and export compact fixed-budget results from completed simulator runs."""
import csv
import itertools
from pathlib import Path
import sys

import numpy as np

from study_protocol import PROTOCOL_PATH, compatible_hashes

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "lift_box_dp/src"))
from common import read_json, sha256, write_json
from compare import verify_actions

LIFT_FIELDS = ("ever_success", "hold_success", "final_success")
PUSHT_FIELDS = ("official_episode_score", "final_score", "terminated")


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def value(text):
    if text in ("True", "False"):
        return float(text == "True")
    return float(text)


def load(task, method, scale, seed, design, protocol):
    if task == "lift":
        directory = HERE.parent / f"lift_box_dp/artifacts/matched_lift_{method}_{scale}_{seed}"
    else:
        directory = ROOT / f"artifacts/matched_budget/runs/pusht_{method}_{scale}_{seed}"
    metadata = read_json(directory / "completed.json")
    expected = {"task": task, "method": method, "scale": scale, "model_seed": seed,
                "optimizer_steps": design["updates"], "batch_size": protocol["batch_size"],
                "checkpoint_selection": protocol["checkpoint_selection"], "smoke": False}
    if (any(metadata[key] != expected[key] for key in expected)
            or metadata["protocol_sha256"] not in compatible_hashes(task, protocol)):
        raise ValueError(f"Unexpected condition metadata: {directory}")
    config = read_json(directory / "config.json")
    training_path = directory / ("training_metrics.csv" if task == "pusht" and method == "alt" else "training.csv")
    training = read_rows(training_path)
    key = {"training_metrics.csv": "optimizer_steps", "training.csv": "update" if task == "pusht" else "steps"}
    if int(training[-1][key[training_path.name]]) != design["updates"]:
        raise ValueError(f"Unexpected final optimizer count: {directory}")
    metadata.update(local_artifacts=directory.relative_to(ROOT).as_posix(), resolved_config=config,
                    training_csv_sha256=sha256(training_path), final_training_row=training[-1])
    if task == "lift":
        path = directory / f"fixed_validation_e{metadata['checkpoint_batch_epoch']:03d}"
        evaluation = read_json(path / "protocol.json")
        rows = read_rows(path / "rollouts.csv")
        seeds = list(range(design["validation_seed_start"], design["validation_seed_start"] + design["validation_count"]))
        if (evaluation["horizon"] != design["validation_horizon"] or evaluation["seeds"] != seeds
                or evaluation["initial_states_sha256"] != metadata["fixture_hashes"]):
            raise ValueError(f"Unexpected Lift protocol: {directory}")
        verify_actions(path / "actions.csv", rows, evaluation)
        metadata["actions_sha256"] = sha256(path / "actions.csv")
        fields, seed_key = LIFT_FIELDS, "seed"
    else:
        path = directory / "test"
        rows = read_rows(path / "rollouts.csv")
        manifest = read_json(HERE.parent / "splits/pusht_v7_split42.json")
        seeds = manifest["test_env_seeds"]
        if any(not 1 <= int(row["steps"]) <= manifest["max_steps"]
               or not 0 <= float(row["final_score"]) <= float(row["official_episode_score"]) <= 1
               or (row["truncated"] == row["terminated"]) for row in rows):
            raise ValueError(f"Invalid Push-T rollout: {directory}")
        if method == "dp":
            record = read_json(directory / "run.json")
            if record["selected_update"] != design["updates"] or record["config"]["checkpoints"] != [design["updates"]]:
                raise ValueError(f"Unexpected checkpoint selection: {directory}")
            metadata["upstream_files"] = record["upstream_files"]
        fields, seed_key = PUSHT_FIELDS, "evaluation_seed"
    if [int(row[seed_key]) for row in rows] != seeds:
        raise ValueError(f"Incomplete or misordered evaluation: {directory}")
    results = {key: np.array([value(row[key]) for row in rows]) for key in fields}
    summary = read_json(path / "metrics.json")
    expected_means = ([summary[key]["rate"] for key in fields] if task == "lift" else
                      [summary["mean_score"], summary["mean_final_score"], summary["successes"] / len(seeds)])
    np.testing.assert_allclose([result.mean() for result in results.values()], expected_means, atol=1e-12, rtol=0)
    metadata["rollouts_sha256"] = sha256(path / "rollouts.csv")
    metadata["metrics"] = summary
    return metadata, rows, results


def describe(values, resamples):
    # Evaluation seeds are paired and reused across training replicates.
    # The interval conditions on the three fitted models per method and scale.
    per_seed = values.mean(axis=1)
    by_start = values.mean(axis=0)
    return {"mean": float(per_seed.mean()), "per_model_seed": per_seed.tolist(),
            "model_seed_range": [float(per_seed.min()), float(per_seed.max())],
            "paired_env_bootstrap95": np.quantile(by_start[resamples].mean(axis=1), [.025, .975]).tolist()}


def main():
    protocol = read_json(PROTOCOL_PATH)
    output = HERE / "evidence"
    output.mkdir(exist_ok=True)
    manifests, comparisons = [], {}
    for task, fields in (("lift", LIFT_FIELDS), ("pusht", PUSHT_FIELDS)):
        design = protocol[task]
        scales = design["scales"]
        data, rollouts, runs = {}, [], []
        for method, scale, seed in itertools.product(("alt", "dp"), scales, protocol["model_seeds"]):
            metadata, rows, results = load(task, method, scale, seed, design, protocol)
            manifests.append(metadata)
            data[method, scale, seed] = results
            identity = {"method": method, "demonstrations": scale, "model_seed": seed}
            rollouts.extend({**identity, **row} for row in rows)
            runs.append({**identity, "optimizer_steps": design["updates"], "evaluation_episodes": len(rows),
                         **{key: float(results[key].mean()) for key in fields}})
        ids = {}
        for scale in scales:
            relevant = [run for run in manifests if run["task"] == task and run["scale"] == scale]
            ids[scale] = relevant[0]["train_episode_ids"]
            if len(ids[scale]) != scale or any(run["train_episode_ids"] != ids[scale] for run in relevant):
                raise ValueError(f"Demonstration sets differ in {task} at scale {scale}")
        if any(ids[larger][:smaller] != ids[smaller] for smaller, larger in zip(scales, scales[1:])):
            raise ValueError(f"Demonstrations are not nested in {task}")
        count = runs[0]["evaluation_episodes"]
        resamples = np.random.default_rng(2026).integers(count, size=(10000, count))
        by_model = {(method, scale, key): np.stack([data[method, scale, seed][key]
                    for seed in protocol["model_seeds"]])
                    for method, scale, key in itertools.product(("alt", "dp"), scales, fields)}
        summary = {"scales": {}, "scale_changes": {}}
        for scale in scales:
            summary["scales"][scale] = {key: {
                "alt": describe(by_model["alt", scale, key], resamples),
                "dp": describe(by_model["dp", scale, key], resamples),
                "dp_minus_alt": describe(by_model["dp", scale, key] - by_model["alt", scale, key], resamples),
            } for key in fields}
        for smaller, larger in itertools.combinations(scales, 2):
            summary["scale_changes"][f"{smaller}_to_{larger}"] = {key: {
                method: describe(by_model[method, larger, key] - by_model[method, smaller, key], resamples)
                for method in ("alt", "dp")
            } for key in fields}
        write_rows(output / f"{task}_rollouts.csv", rollouts)
        write_rows(output / f"{task}_models.csv", runs)
        comparisons[task] = summary
    write_json(output / "manifests.json", manifests)
    write_json(output / "comparison.json", {"model_seeds": protocol["model_seeds"], **comparisons})


if __name__ == "__main__":
    main()
