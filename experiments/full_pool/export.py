"""Verify full-pool checkpoint selection and archive small reproducible results."""
import csv
from pathlib import Path
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "lift_box_dp/src"))
from common import ROOT as LIFT, read_json, sha256, write_json
from compare import verify_actions
from full_pool import full_pool
from matched_budget import fixtures

LIFT_FIELDS = ("ever_success", "hold_success", "final_success")
PUSHT_FIELDS = ("official_episode_score", "final_score", "terminated")


def verify_hash(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"Artifact checksum mismatch: {path}")


def scored_rows(path, expected_seeds, seed_key, keys):
    with (path / "rollouts.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if [int(row[seed_key]) for row in rows] != expected_seeds:
        raise ValueError(f"Missing or misordered eval seeds: {path}")
    columns = {}
    for key in keys:
        columns[key] = np.array([float(row[key] == "True") if row[key] in ("True", "False")
                                 else float(row[key]) for row in rows])
    return rows, columns


def checksummed_artifacts(run, epochs, saved, use_bank):
    for epoch in epochs:
        record = saved[str(epoch)]
        stem = f"epoch{epoch:03d}"
        verify_hash(run / f"{stem}.pt", record["checkpoint_sha256"])
        if use_bank:
            verify_hash(run / f"{stem}_bank.npz", record["bank_sha256"])


def lift(method):
    cfg, alt, _, split, _ = full_pool()
    design, valid_seeds, _, hashes = fixtures(split, alt, cfg)
    settings = read_json(alt / split["evaluation_config"]["path"])
    run = LIFT / f"artifacts/full_pool_lift_{method}_v1"
    complete = read_json(run / "completed.json")
    selection = read_json(run / "selection.json")
    saved = read_json(run / "checkpoints.json")
    epochs = cfg[f"{method}_epochs"]
    if (complete["status"] != "completed" or complete["method"] != method
            or read_json(run / "config.json") != cfg or sorted(map(int, saved)) != epochs
            or selection["validation_seeds"] != valid_seeds or selection["fixture_hashes"] != hashes
            or selection["selection_order"] != cfg["selection_order"]):
        raise ValueError(f"Unexpected Lift training or evaluation protocol: {run}")
    verify_hash(run / "selection.json", complete["selection_sha256"])
    checksummed_artifacts(run, epochs, saved, method == "alt")

    def read_case(path, seeds, horizon, fixture_ref, epoch):
        protocol = read_json(path / "protocol.json")
        if (protocol["config"] != cfg or protocol["policy"] != method or protocol["epoch"] != epoch
                or protocol["seeds"] != seeds or protocol["horizon"] != horizon
                or protocol["initial_states_sha256"] != fixture_ref or protocol["checkpoint"] != saved[str(epoch)]):
            raise ValueError(f"Invalid Lift rollout protocol: {path}")
        rows, columns = scored_rows(path, seeds, "seed", LIFT_FIELDS)
        verify_actions(path / "actions.csv", rows, protocol)
        scores = read_json(path / "metrics.json")
        if scores["n_rollouts"] != len(rows) or any(scores[key]["count"] != columns[key].sum() for key in LIFT_FIELDS):
            raise ValueError(f"Lift metrics do not match rollouts: {path}")
        return rows, columns, scores

    validation_rows, checkpoints = [], {}
    for epoch in epochs:
        path = run / f"validation_e{epoch:03d}"
        rows, _, scores = read_case(path, valid_seeds, design["validation_horizon"], hashes, epoch)
        if scores != selection["validation_scores"][str(epoch)]:
            raise ValueError(f"Lift selection references different validation records: {path}")
        validation_rows.extend({"epoch": epoch, **row} for row in rows)
        checkpoints[epoch] = {**scores, **saved[str(epoch)]}
    rank = cfg["selection_order"][:-1]
    best = max(checkpoints, key=lambda epoch: tuple(checkpoints[epoch][key]["count"] for key in rank) + (-epoch,))
    if selection["selected_epoch"] != best:
        raise ValueError("Recorded Lift checkpoint does not maximize the validation criterion")
    path = run / f"test_e{best:03d}"
    rows, columns, scores = read_case(path, settings["test_seeds"], settings["horizon"],
                                      split["evaluation_initial_state_contract"]["sha256"], best)
    training = complete["training"]
    if training["train_ids"] != cfg["train_ids"] or training["steps"] != saved[str(epochs[-1])]["steps"]:
        raise ValueError("Lift training metadata differs from the final checkpoint")
    metadata = {"config": cfg, "training": training, "selection_order": cfg["selection_order"],
                "validation": checkpoints, "selected_epoch": best,
                "selected_updates": saved[str(best)]["steps"], "test": scores,
                "test_protocol": read_json(path / "protocol.json"), "fixture_hashes": hashes,
                "evaluation_source_sha256": complete["eval_source_sha256"],
                "test_actions_sha256": sha256(path / "actions.csv")}
    return metadata, run / "training.csv", validation_rows, rows, columns


def pusht(method):
    manifest = read_json(HERE.parent / "splits/pusht_v7_split42.json")
    dp = read_json(HERE / "pusht_dp.json")
    alt = read_json(HERE / "pusht_alt.json")
    run = ROOT / ("artifacts/full_pool/pusht_alt_v1" if method == "alt"
                  else "artifacts/pusht_dp/full_pool_upper_v1/186")
    record = read_json(run / "run.json")
    valid = read_json(run / "validation.json")
    seed_start, seed_count = dp["validation_seed_start"], dp["validation_count"]
    seeds = list(range(seed_start, seed_start + seed_count))
    if (record["status"] != "completed" or record["smoke"] or record["validation_seeds"] != seeds
            or record["train_episodes"] != manifest["ordered_train_pool_episode_ids"]
            or record["test_seeds"] != manifest["test_env_seeds"]
            or record["dataset_fingerprint"] != manifest["dataset_fingerprint"]):
        raise ValueError(f"Unexpected Push-T training or eval protocol: {run}")
    config = read_json(run / "config.json") if method == "alt" else record["config"]
    if method == "dp" and config != dp:
        raise ValueError("DP training config differs from the frozen full Push-T pool")
    if method == "alt" and (config["training"]["checkpoint_epochs"] != alt["checkpoint_epochs"]
                            or config["training"]["num_workers"] != alt["loader_workers"]):
        raise ValueError("ALT training config differs from the frozen full Push-T pool")

    def read_case(path, expected_seeds, scores):
        rows, values = scored_rows(path, expected_seeds, "evaluation_seed", PUSHT_FIELDS)
        if any(not 1 <= int(row["steps"]) <= manifest["max_steps"] or row["terminated"] == row["truncated"]
               or not 0 <= float(row["final_score"]) <= float(row["official_episode_score"]) <= 1 for row in rows):
            raise ValueError(f"Invalid Push-T rollout: {path}")
        np.testing.assert_allclose([values[key].mean() for key in PUSHT_FIELDS],
                                   [scores["mean_score"], scores["mean_final_score"], scores["successes"] / len(rows)],
                                   atol=1e-12, rtol=0)
        return rows, values

    checkpoints = alt["checkpoint_epochs"] if method == "alt" else dp["checkpoints"]
    if sorted(map(int, valid)) != checkpoints:
        raise ValueError("Unexpected Push-T validation checkpoints")
    validation_rows = []
    for point in checkpoints:
        label = f"epoch{point:03d}" if method == "alt" else str(point)
        weights = f"{label}.pt" if method == "alt" else f"ema_{label}.pt"
        stats = valid[str(point)]
        verify_hash(run / weights, stats["checkpoint_sha256"])
        if method == "alt":
            verify_hash(run / f"{label}_bank.pt", stats["bank_sha256"])
        rows, _ = read_case(run / f"validation_{label}", seeds, stats)
        validation_rows.extend({"checkpoint": point, **row} for row in rows)
    field = "selected_epoch" if method == "alt" else "selected_update"
    best = max(checkpoints, key=lambda point: (valid[str(point)]["mean_score"], -point))
    if record[field] != best:
        raise ValueError("Recorded Push-T checkpoint does not maximize the validation criterion")
    rows, columns = read_case(run / "test", manifest["test_env_seeds"], record["test"])
    metadata = {**record, "config": config, "validation": valid,
                "selected_updates": valid[str(best)]["optimizer_steps"] if method == "alt" else best}
    return metadata, run / ("training_metrics.csv" if method == "alt" else "training.csv"), validation_rows, rows, columns


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    output = HERE / "evidence"
    output.mkdir(exist_ok=True)
    for task, read in (("lift", lift), ("pusht", pusht)):
        alt, dp = read("alt"), read("dp")
        paired = {}
        n = len(alt[3])
        bootstrap = np.random.default_rng(2026).integers(n, size=(10000, n))
        for metric, alt_scores in alt[4].items():
            delta = dp[4][metric] - alt_scores
            paired[metric] = {"dp_minus_alt": float(delta.mean()),
                              "paired_env_bootstrap95": np.quantile(delta[bootstrap].mean(axis=1), [.025, .975]).tolist()}
        metadata = {"methods": {}, "paired": paired, "n_test_starts": n}
        for method, (record, training, validation, test, _) in (("alt", alt), ("dp", dp)):
            metadata["methods"][method] = record
            name = f"{task}_{method}"
            shutil.copyfile(training, output / f"{name}_training.csv")
            write_rows(output / f"{name}_validation.csv", validation)
            write_rows(output / f"{name}_test.csv", test)
        write_json(output / f"{task}_comparison.json", metadata)


if __name__ == "__main__":
    main()
