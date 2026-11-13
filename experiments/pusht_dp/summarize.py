"""Check matched Push-T evidence, export small logs, and summarize paired comparisons."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from pusht.repro import load_split, save_json, sha256

SPLIT = ROOT / "experiments/splits/pusht_v7_split42.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_rollouts(path, seeds):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(r["evaluation_seed"]) for r in rows] == seeds
    peak = np.array([float(r["official_episode_score"]) for r in rows])
    end = np.array([float(r["final_score"]) for r in rows])
    success = [r["terminated"] == "True" for r in rows]
    for row, score, final, terminated in zip(rows, peak, end, success):
        steps = int(row["steps"])
        assert 0 <= final <= score <= 1
        assert 1 <= steps <= 300 and (steps == 300 or terminated)
        assert (row["truncated"] == "True") == (not terminated)
        if terminated:
            assert score == final == 1
    return peak, {"n": len(rows), "mean_score": float(peak.mean()),
                  "mean_final_score": float(end.mean()), "successes": sum(success)}


def export_run(source, destination, split, config_path):
    run = read(source / "run.json")
    seeds = split["test_env_seeds"]
    assert run["status"] == "completed"
    assert run["split_sha256"] == sha256(SPLIT)
    assert run["dataset_fingerprint"] == split["dataset_fingerprint"]
    assert run["heldout_episodes"] == split["heldout_validation_episode_ids"]
    assert run["test_seeds"] == seeds and not set(run["validation_seeds"]) & set(seeds)
    assert run["config_sha256"] == sha256(config_path) and run["config"] == read(config_path)
    assert run["train_episodes"] == split["ordered_train_pool_episode_ids"][:run["scale"]]
    dp_path = source / "test/rollouts.csv"
    scores, metrics = check_rollouts(dp_path, seeds)
    assert metrics["mean_score"] == run["test"]["mean_score"]
    checks = read(source / "validation.json")
    selected = run["selected_update"]
    assert selected == max(map(int, checks), key=lambda u: (checks[str(u)]["mean_score"], -u))
    destination.mkdir(parents=True, exist_ok=False)
    for step, entry in checks.items():
        assert sha256(source / f"ema_{step}.pt") == entry["checkpoint_sha256"]
        rollout = source / f"validation_{step}/rollouts.csv"
        _, val = check_rollouts(rollout, run["validation_seeds"])
        assert np.isclose(val["mean_score"], entry["mean_score"], atol=1e-12, rtol=0)
        shutil.copyfile(rollout, destination / f"validation_{step}_rollouts.csv")
    for filename in ("run.json", "training.csv", "validation.json"):
        shutil.copyfile(source / filename, destination / filename)
    shutil.copyfile(dp_path, destination / "test_rollouts.csv")
    heavy = sorted(source.glob("ema_*.pt")) + sorted(source.glob("test/media/*.mp4"))
    assert len(heavy) == len(checks) + 4
    save_json(destination / "artifact_manifest.json", {
        "run_directory": (source.relative_to(ROOT) if source.is_relative_to(ROOT) else source).as_posix(),
        "files": [{"path": file.relative_to(source).as_posix(),
                   "bytes": file.stat().st_size, "sha256": sha256(file)} for file in heavy],
    })
    metrics.update(selected_update=selected, training_seconds=checks[str(selected)]["training_seconds"],
                   total_training_seconds=run["total_training_seconds"],
                   equivalent_epochs=checks[str(selected)]["equivalent_epochs"],
                   sd=run["test"]["sd"], parameters=run["policy_parameters"],
                   evaluation_seconds=run["test"]["runtime_seconds"])
    return scores, metrics


def summarize(source, destination):
    source = source.resolve()
    split = load_split(SPLIT)
    seeds = split["test_env_seeds"]
    rng = np.random.default_rng(2026)
    draws = rng.integers(0, len(seeds), size=(10000, len(seeds)))
    dp_scores, alt_scores, scale_results = {}, {}, {}
    for scale in (30, 100):
        source_scale = source / str(scale)
        alt = ROOT / "experiments/runs" / f"alt_{scale:03}_split42_model42_pilot_v1"
        run = read(source_scale / "run.json")
        alt_run = read(alt / "run_metadata.json")
        assert alt_run["status"] == "completed"
        assert run["split_sha256"] == alt_run["split_manifest_sha256"]
        assert run["dataset_fingerprint"] == alt_run["dataset_fingerprint"]
        assert run["train_episodes"] == alt_run["dataset_manifest_ref"]["train_episode_ids"]
        dp_scores[scale], dp = export_run(source_scale, destination / str(scale),
                                         split, HERE / "config.json")
        alt_scores[scale], alt_metrics = check_rollouts(alt / "eval_rollouts.csv", seeds)
        assert np.isclose(alt_metrics["mean_score"], read(alt / "eval_metrics.json")["mean_score"])
        alt_metrics.update(updates=alt_run["optimizer_steps"], training_seconds=alt_run["training_seconds"])
        scale_results[scale] = {"dp": dp, "alt": alt_metrics}

    def paired(new, old):
        delta = new - old
        return {"difference": float(delta.mean()),
                "ci95": np.quantile(delta[draws].mean(axis=1), [0.025, 0.975]).tolist()}

    geometry = ROOT / "experiments/pusht_residual/experiments/network_study/evidence"
    for name in ("aligned", "round3"):
        _, metrics = check_rollouts(geometry / f"test_{name}_rollouts.csv", seeds)
        reference = read(geometry / "test_summary.json")[name]
        assert np.isclose(metrics["mean_score"], reference["mean"])
    save_json(destination / "comparison.json", {
        "scales": scale_results, "bootstrap_seed": 2026, "bootstrap_draws": len(draws),
        "dp_minus_alt_30": paired(dp_scores[30], alt_scores[30]),
        "dp_minus_alt_100": paired(dp_scores[100], alt_scores[100]),
        "dp_100_minus_dp_30": paired(dp_scores[100], dp_scores[30]),
    })


def summarize_long(source, destination, reference):
    split = load_split(SPLIT)
    seeds = split["test_env_seeds"]
    scores, results = {}, {}
    config_path = HERE / "long_config.json"
    for scale in read(config_path)["scales"]:
        run_dir = source.resolve() / str(scale)
        assert read(run_dir / "run.json")["code_sha256"] == sha256(HERE / "run.py")
        scores[scale], results[scale] = export_run(run_dir, destination / str(scale), split, config_path)
    old, _ = check_rollouts(HERE / "evidence/100/test_rollouts.csv", seeds)
    rng = np.random.default_rng(2026)
    draws = rng.integers(0, len(seeds), size=(10000, len(seeds)))

    def paired(delta):
        return {"difference": float(delta.mean()),
                "ci95": np.quantile(delta[draws].mean(axis=1), [0.025, 0.975]).tolist()}

    ref = read(reference / "reference.json")
    assert ref["test_seeds"] == seeds
    _, ref_metrics = check_rollouts(reference / "rollouts.csv", seeds)
    assert ref_metrics["mean_score"] == ref["metrics"]["mean_score"]
    target = destination / "official"
    target.mkdir()
    for filename in ("reference.json", "rollouts.csv"):
        shutil.copyfile(reference / filename, target / filename)
    save_json(destination / "comparison.json", {
        "scales": results, "official_pretrained": ref_metrics,
        "dp_100_long_minus_short": paired(scores[100] - old),
        "dp_186_minus_100_long": paired(scores[186] - scores[100]),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Local artifact run directory")
    parser.add_argument("destination", type=Path, help="New evidence directory")
    parser.add_argument("--official-reference", type=Path, help="Use the long study and its upstream reference")
    args = parser.parse_args()
    if args.destination.exists():
        parser.error("Evidence directory must be new")
    if args.official_reference:
        summarize_long(args.source, args.destination, args.official_reference)
    else:
        summarize(args.source, args.destination)
