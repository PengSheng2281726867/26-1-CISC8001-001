"""Verify both selected test runs and write a compact paired Lift-10 result."""
import argparse

import numpy as np

from common import ROOT, read_json, sha256, write_json
from epoch_study import STUDY_DIR, study
from evaluate_epochs import check_trace


def load(run_id, method, cfg, split, horizon):
    run = ROOT / "artifacts" / run_id
    selection = read_json(run / "selection.json")
    if selection["policy"] != method:
        raise ValueError("Unexpected policy")
    epoch = selection["selected_epoch"]
    result = run / f"test_e{epoch:03d}"
    protocol = read_json(result / "protocol.json")
    if (protocol["config"] != cfg or protocol["policy"] != method
            or protocol["seeds"] != split["test_env_seeds"] or protocol["horizon"] != horizon
            or protocol["initial_states_sha256"] != split["evaluation_initial_state_contract"]["sha256"]):
        raise ValueError("Test evaluation protocol mismatch")
    rows = check_trace(result, protocol)
    metrics = read_json(result / "metrics.json")
    for metric in ("ever_success", "hold_success", "final_success"):
        if sum(row[metric] == "True" for row in rows) != metrics[metric]["count"]:
            raise ValueError("Test summary differs from the rollout records")
    return rows, {
        "run_id": run_id, "selection": selection, "test": metrics,
        "training": read_json(run / "training.json"),
        "files_sha256": {name: sha256(result / name) for name in
                         ("metrics.json", "rollouts.csv", "actions.csv", "protocol.json")},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("alt_run")
    parser.add_argument("dp_run")
    args = parser.parse_args()
    cfg, alt, _, split, _ = study()
    horizon = read_json(alt / split["evaluation_config"]["path"])["horizon"]
    a_rows, a = load(args.alt_run, "alt", cfg, split, horizon)
    d_rows, d = load(args.dp_run, "dp", cfg, split, horizon)
    result = {"config": cfg, "alt": a, "dp": d, "paired": {}}
    rng = np.random.default_rng(42)
    resampled = rng.integers(0, len(a_rows), size=(10000, len(a_rows)))
    for metric in ("ever_success", "hold_success", "final_success"):
        delta = np.array([int(d[metric] == "True") - int(a[metric] == "True") for a, d in zip(a_rows, d_rows)])
        result["paired"][metric] = {
            "dp_minus_alt": float(delta.mean()),
            "bootstrap95": np.quantile(delta[resampled].mean(axis=1), [0.025, 0.975]).tolist(),
            "dp_only": int((delta > 0).sum()), "alt_only": int((delta < 0).sum()),
        }
    output = STUDY_DIR / f"comparison_{args.alt_run}_{args.dp_run}.json"
    if output.exists():
        raise FileExistsError(output)
    write_json(output, result)
    print(output)


if __name__ == "__main__":
    main()
