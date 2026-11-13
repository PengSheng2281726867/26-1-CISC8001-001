"""Independently verify frozen seeds, action traces, and aggregate statistics."""
import csv
import json

import numpy as np

from study import CFG, ROOT, ORACLE, load, sha
from geometry_bank import GeometryBank


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    cfg, base, manifest = load()
    bank = GeometryBank(base["paths"]["dataset"], manifest["train_subsets"]["30"])
    data = bank.data
    seeds = manifest["test_env_seeds"]
    rng = np.random.default_rng(CFG["bootstrap_seed"])
    draws = rng.integers(0, len(seeds), (CFG["bootstrap_draws"], len(seeds)))
    output = ROOT / "artifacts/residual_study_verification"
    output.mkdir(exist_ok=False)
    counts = {}
    all_scores = {}
    original = {}
    for n in (1, 2, 3):
        directory = ROOT / "artifacts" / f"round{n}_evaluate"
        training = ROOT / "artifacts" / f"round{n}_train"
        report = read(directory / "summary.json")
        assert report["checkpoint_sha256"] == sha(training / "residual.pt")
        assert read(training / "run.json")["status"] == read(directory / "run.json")["status"] == "completed"
        for stage, source, names in (
            ("train", training, ("run.json", "validation.json", "curve.json")),
            ("evaluate", directory, ("run.json", "summary.json", "raw.csv", "aligned.csv", "residual.csv")),
        ):
            for name in names:
                (output / f"round{n}_{stage}_{name}").write_bytes((source / name).read_bytes())
        score = {}
        for mode in ("raw", "aligned", "residual"):
            with (directory / f"{mode}.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            assert [int(r["evaluation_seed"]) for r in rows] == seeds
            steps = {seed: 0 for seed in seeds}
            actions = directory / mode / "queries.jsonl"
            assert sha(actions) == report[mode]["query_sha256"]
            magnitude = []
            with actions.open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    seed, episode, frame = row["evaluation_seed"], row["episode"], row["frame"]
                    assert episode in manifest["train_subsets"]["30"]
                    assert row["current_step"] == steps[seed] and 1 <= row["executed_actions"] <= 8
                    steps[seed] += row["executed_actions"]
                    source, idx = data.action_chunk(episode, frame)
                    np.testing.assert_array_equal(idx, row["source_indices"])
                    command = np.asarray(row["commands"])
                    assert command.shape == (8, 2) and np.isfinite(command).all()
                    assert np.all((0 <= command) & (command <= 512))
                    if mode == "raw":
                        np.testing.assert_array_equal(command, source)
                    if mode == "aligned":
                        q, m = np.asarray(row["state"]), np.asarray(row["matched_state"])
                        angle = q[4] - m[4]
                        c, s = np.cos(angle), np.sin(angle)
                        aligned = (source-m[2:4]) @ np.array([[c,s],[-s,c]]) + q[2:4]
                        np.testing.assert_allclose(command, aligned.clip(0, 512), atol=5e-5, rtol=0)
                    magnitude.append(row["residual_l2"])
                    if n == 3 and mode == "residual":
                        assert row["residual_l2"] <= 120.0001
            assert steps == {int(r["evaluation_seed"]): int(r["steps"]) for r in rows}
            score[mode] = np.array([float(r["official_episode_score"]) for r in rows])
            np.testing.assert_allclose(score[mode].mean(), report[mode]["mean"], atol=1e-12)
            np.testing.assert_allclose(np.mean([float(r["final_score"]) for r in rows]),
                                       report[mode]["mean_final"], atol=1e-12)
            assert len(magnitude) == report[mode]["queries"]
            counts[f"round{n}_{mode}"] = {"rollouts": len(rows), "queries": len(magnitude),
                                         "mean_residual_l2": float(np.mean(magnitude)),
                                         "p90_residual_l2": float(np.quantile(magnitude, .9))}
        diff = score["residual"]-score["aligned"]
        np.testing.assert_allclose(np.quantile(diff[draws].mean(axis=1), [.025,.975]),
                                   report["paired"]["residual_minus_aligned"]["ci95"], atol=1e-12)
        all_scores[n] = score
    for mode in ("raw","aligned"):
        np.testing.assert_allclose(all_scores[2][mode], all_scores[3][mode], atol=1e-12, rtol=0)
    with (ORACLE / "results/eval_rollouts.csv").open(newline="", encoding="utf-8") as stream:
        original = {int(r["evaluation_seed"]): float(r["official_episode_score"]) for r in csv.DictReader(stream)}
    np.testing.assert_allclose(all_scores[1]["raw"], [original[s] for s in seeds], atol=1e-12, rtol=0)
    diff = all_scores[3]["residual"]-all_scores[2]["residual"]
    counts["round3_minus_round2_residual"] = {
        "delta": float(diff.mean()), "ci95": np.quantile(diff[draws].mean(axis=1), [.025, .975]).tolist()}
    (output / "verification.json").write_text(json.dumps(counts, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
