"""Pair DP and ALT outcomes by frozen reset seed."""
import argparse
import csv
import json
import statistics

import numpy as np

from common import ROOT, contract, read_json, sha256, write_json


def verify_actions(path, rows, protocol):
    cfg = protocol["config"]
    state = {int(row["seed"]): {"steps": 0, "hold": 0, "max": 0, "first": None, "reward": 0}
             for row in rows}
    with path.open(newline="", encoding="utf-8") as stream:
        for entry in csv.DictReader(stream):
            entry_state = state[int(entry["seed"])]
            step = int(entry["step"])
            success = int(entry["above_threshold"])
            reward = float(entry["reward"])
            action = np.asarray(json.loads(entry["action"]))
            if step != entry_state["steps"] + 1 or success not in (0, 1) or reward != success:
                raise ValueError("Invalid step or sparse reward in control trace")
            if action.shape != (7,) or not np.isfinite(action).all() or np.max(np.abs(action)) > 1.00001:
                raise ValueError("Invalid controller action in trace")
            entry_state["steps"] = step
            entry_state["reward"] = reward
            if success and entry_state["first"] is None:
                entry_state["first"] = step
            entry_state["hold"] = entry_state["hold"] + 1 if success else 0
            entry_state["max"] = max(entry_state["max"], entry_state["hold"])
    for row in rows:
        value = state[int(row["seed"])]
        if (value["steps"] != int(row["steps"])
                or value["steps"] != protocol["horizon"]
                or value["first"] != (int(row["first_success_step"]) if row["first_success_step"] else None)
                or value["max"] != int(row["longest_hold_steps"])
                or value["hold"] != int(row["terminal_hold_steps"])
                or value["reward"] != float(row["final_reward"])
                or bool(value["hold"]) != (row["final_success"] == "True")
                or (value["first"] is not None) != (row["ever_success"] == "True")
                or (value["max"] >= cfg["hold_steps"]) != (row["hold_success"] == "True")):
            raise ValueError(f"Recorded metrics differ from the action trace at seed {row['seed']}")


def load(run, evaluation, policy, expected):
    path = ROOT / "artifacts" / run / evaluation
    protocol = read_json(path / "protocol.json")
    if protocol["policy"] != policy or protocol["smoke"] or protocol["execution_horizon"] != 1:
        raise ValueError("Unexpected evaluation protocol")
    with (path / "rollouts.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if [int(row["seed"]) for row in rows] != expected:
        raise ValueError("Evaluation is partial or differs from the frozen seed order")
    verify_actions(path / "actions.csv", rows, protocol)
    summary = read_json(path / "metrics.json")
    if summary["n_rollouts"] != len(rows):
        raise ValueError("Summary count differs from the completed rollouts")
    for key in ("ever_success", "final_success", "hold_success"):
        if sum(row[key] == "True" for row in rows) != summary[key]["count"]:
            raise ValueError(f"Summary count differs from per-seed records: {key}")
    summary["file_sha256"] = {name: sha256(path / name) for name in
                              ("protocol.json", "rollouts.csv", "actions.csv", "metrics.json")}
    return rows, protocol, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("alt_run")
    parser.add_argument("dp_run")
    parser.add_argument("evaluation_id")
    args = parser.parse_args()
    cfg, alt_path, _, split, _ = contract()
    settings = read_json(alt_path / split["evaluation_config"]["path"])
    seeds = settings["test_seeds"]
    alt, a_protocol, a_summary = load(args.alt_run, args.evaluation_id, "alt", seeds)
    dp, d_protocol, d_summary = load(args.dp_run, args.evaluation_id, "dp", seeds)
    for key in ("config", "horizon", "initial_states_sha256", "evaluation_config_sha256"):
        if a_protocol[key] != d_protocol[key]:
            raise ValueError(f"Evaluation protocol differs: {key}")
    if a_protocol["config"] != cfg:
        raise ValueError("Results differ from the frozen configuration")
    generator = np.random.default_rng(42)
    resamples = generator.integers(0, len(seeds), size=(10000, len(seeds)))
    comparison = {"alt_run": args.alt_run, "dp_run": args.dp_run,
                  "evaluation": args.evaluation_id, "seeds": seeds,
                  "alt": a_summary, "dp": d_summary, "paired": {}}
    for key in ("ever_success", "final_success", "hold_success"):
        differences = np.asarray([int(d[key] == "True") - int(a[key] == "True") for a, d in zip(alt, dp)])
        comparison["paired"][key] = {
            "dp_minus_alt": float(differences.mean()),
            "bootstrap95": np.quantile(differences[resamples].mean(axis=1), [0.025, 0.975]).tolist(),
            "dp_only": int((differences == 1).sum()),
            "alt_only": int((differences == -1).sum()),
        }
    for method, rows in (("alt", alt), ("dp", dp)):
        steps = [int(row["first_success_step"]) for row in rows if row["first_success_step"]]
        comparison[method]["median_first_hit_among_successes"] = statistics.median(steps) if steps else None
    with (alt_path / "results/per_seed_scores.csv").open(newline="", encoding="utf-8") as stream:
        historical = {int(row["eval_seed"]): row for row in csv.DictReader(stream)}
    comparison["alt"]["historical_outcome_discrepancy_seeds"] = [
        int(row["seed"]) for row in alt
        if (row["ever_success"], row["first_success_step"], float(row["final_reward"])) !=
           (historical[int(row["seed"])]["success"], historical[int(row["seed"])]["steps_to_success"],
            float(historical[int(row["seed"])]["final_reward"]))
    ]
    comparison["compare_source_sha256"] = sha256(__file__)
    output = ROOT / "experiments/exp_001_lift30" / f"comparison_{args.dp_run}_{args.evaluation_id}.json"
    if output.exists():
        raise FileExistsError(output)
    write_json(output, comparison)
    print(output)


if __name__ == "__main__":
    main()
