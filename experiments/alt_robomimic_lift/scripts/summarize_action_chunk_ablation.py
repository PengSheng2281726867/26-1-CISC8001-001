#!/usr/bin/env python3
"""Regenerate the paired first-hit action-chunk summary from rollout records."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any


EVALUATIONS = {
    1: "evaluation01",
    3: "chunk03_firsthit_v1",
    5: "chunk05_firsthit_v1",
    7: "chunk07_firsthit_v1",
    9: "chunk09_firsthit_v1",
}
EVAL_Z = 1.959963984540054


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def wilson_interval(successes: int, count: int) -> list[float]:
    p = successes / count
    z2 = EVAL_Z * EVAL_Z
    denom = 1 + z2 / count
    center = (p + z2 / (2 * count)) / denom
    radius = EVAL_Z * math.sqrt(p * (1 - p) / count + z2 / (4 * count * count)) / denom
    return [center - radius, center + radius]


def exact_mcnemar_p(a_only: int, b_only: int) -> float:
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    lower_tail = sum(math.comb(discordant, k) for k in range(min(a_only, b_only) + 1)) / (2**discordant)
    return min(1.0, 2 * lower_tail)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_group(run_dir: Path, chunk: int, evaluation_id: str) -> dict[str, Any]:
    eval_dir = run_dir / "evaluations" / evaluation_id
    metrics = json.loads((eval_dir / "eval_metrics.json").read_text(encoding="utf-8"))
    resolved_path = eval_dir / "resolved_evaluation.json"
    resolved = json.loads(resolved_path.read_text(encoding="utf-8")) if resolved_path.exists() else {}
    rows = load_csv(eval_dir / "eval_rollouts.csv")
    keyed = {int(row["eval_seed"]): row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError(f"Duplicate evaluation seeds in {evaluation_id}")
    if len(rows) != 50:
        raise ValueError(f"Expected 50 rollouts in {evaluation_id}, found {len(rows)}")

    seeds = sorted(keyed)
    success_seeds = [seed for seed in seeds if parse_bool(keyed[seed]["success"])]
    first_hit_steps = [float(keyed[seed]["steps_to_success"]) for seed in success_seeds]
    steps = [float(keyed[seed]["steps"]) for seed in seeds]
    first_hit_or_cap_steps = [
        float(keyed[seed]["steps_to_success"])
        if parse_bool(keyed[seed]["success"]) else float(keyed[seed]["steps"])
        for seed in seeds
    ]
    success_count = len(success_seeds)
    if success_count != int(metrics["success_count"]):
        raise ValueError(f"Row success count differs from metrics in {evaluation_id}")
    if int(metrics["rollout_count"]) != len(rows):
        raise ValueError(f"Row count differs from metrics in {evaluation_id}")
    if not math.isclose(float(metrics["success_rate"]), success_count / len(rows), abs_tol=1e-12):
        raise ValueError(f"Metrics success rate differs from rollout rows in {evaluation_id}")
    if any(not math.isclose(a, b, abs_tol=1e-12) for a, b in zip(
        metrics["wilson_95_percent_interval"], wilson_interval(success_count, len(rows))
    )):
        raise ValueError(f"Metrics Wilson interval differs from recomputed interval in {evaluation_id}")
    if "retrieval_queries" in keyed[seeds[0]]:
        row_queries = sum(int(keyed[seed]["retrieval_queries"]) for seed in seeds)
        if row_queries != int(metrics["total_retrieval_queries"]):
            raise ValueError(f"Retrieval query total differs from rollout rows in {evaluation_id}")
    if "action_padding_count" in keyed[seeds[0]]:
        row_padding = sum(int(keyed[seed]["action_padding_count"]) for seed in seeds)
        if row_padding != int(metrics["action_padding_count"]):
            raise ValueError(f"Action padding total differs from rollout rows in {evaluation_id}")
    if chunk > 1:
        if any(int(keyed[seed]["action_chunk_length"]) != chunk for seed in seeds):
            raise ValueError(f"Rollout chunk length differs from requested K={chunk} in {evaluation_id}")
        if not metrics.get("stop_on_success", False):
            raise ValueError(f"First-hit stopping is disabled in {evaluation_id}")

    initial_hashes = {seed: keyed[seed]["initial_state_sha256"] for seed in seeds}
    return {
        "action_chunk_length": chunk,
        "evaluation_id": evaluation_id,
        "rows_by_seed": keyed,
        "seeds": seeds,
        "success_seeds": success_seeds,
        "initial_state_hashes": initial_hashes,
        "success_count": success_count,
        "rollout_count": len(rows),
        "success_rate": success_count / len(rows),
        "wilson_95_percent_interval": wilson_interval(success_count, len(rows)),
        "mean_steps_to_success": mean(first_hit_steps) if first_hit_steps else None,
        "median_steps_to_success": median(first_hit_steps) if first_hit_steps else None,
        "success_step_p25": percentile(first_hit_steps, 0.25),
        "success_step_p75": percentile(first_hit_steps, 0.75),
        "mean_environment_steps_per_rollout": mean(steps),
        "mean_first_hit_or_cap_steps_per_rollout": mean(first_hit_or_cap_steps),
        "total_environment_steps": int(sum(steps)),
        "mean_wall_seconds_per_rollout": mean(float(keyed[s]["wall_seconds"]) for s in seeds),
        "evaluation_wall_seconds": float(metrics["evaluation_wall_seconds"]),
        # The original K=1 CSV predates explicit query/padding columns. It queried
        # once per environment step; its padding count was not separately logged.
        "mean_retrieval_queries_per_rollout": mean(
            float(keyed[s].get("retrieval_queries") or keyed[s]["steps"]) for s in seeds
        ),
        "total_retrieval_queries": int(metrics.get("total_retrieval_queries", sum(steps))),
        "mean_action_padding_per_rollout": (
            mean(float(keyed[s]["action_padding_count"]) for s in seeds)
            if "action_padding_count" in keyed[seeds[0]] else None
        ),
        "total_action_padding_count": metrics.get("action_padding_count"),
        "mean_retrieval_episode_switches_per_rollout": mean(float(keyed[s]["retrieval_switches_episode"]) for s in seeds),
        "mean_similarity": mean(float(keyed[s]["mean_similarity"]) for s in seeds),
        "dataset_fingerprint": metrics["dataset_fingerprint"],
        "split_manifest_sha256": metrics["split_manifest_sha256"],
        "evaluation_seed_manifest_sha256": metrics["evaluation_seed_manifest_sha256"],
        "success_definition": metrics["success_definition"],
        "checkpoint_sha256": metrics.get("checkpoint_sha256", resolved.get("checkpoint_sha256")),
        "memory_bank_sha256": metrics.get("memory_bank_sha256", resolved.get("memory_bank_sha256")),
        "source_code_commit": resolved.get("evaluation_code_commit"),
        "evaluation_script_sha256": resolved.get("evaluation_script_sha256"),
        "ablation_plan_sha256": resolved.get("ablation_plan_sha256"),
        "raw_metrics_wilson_interval": metrics["wilson_95_percent_interval"],
    }


def paired_comparison(groups: dict[int, dict[str, Any]], a: int, b: int) -> dict[str, Any]:
    a_rows = groups[a]["rows_by_seed"]
    b_rows = groups[b]["rows_by_seed"]
    if sorted(a_rows) != sorted(b_rows):
        raise ValueError(f"Seed mismatch in paired comparison K={a} vs K={b}")
    a_only = b_only = both = neither = 0
    for seed in sorted(a_rows):
        a_success = parse_bool(a_rows[seed]["success"])
        b_success = parse_bool(b_rows[seed]["success"])
        if a_success and b_success:
            both += 1
        elif a_success:
            a_only += 1
        elif b_success:
            b_only += 1
        else:
            neither += 1
    return {
        "comparison": f"K={a} vs K={b}",
        "a_chunk_length": a,
        "b_chunk_length": b,
        "both_success": both,
        "a_only_success": a_only,
        "b_only_success": b_only,
        "neither_success": neither,
        "discordant_seeds": a_only + b_only,
        "success_rate_difference_b_minus_a": groups[b]["success_rate"] - groups[a]["success_rate"],
        "exact_two_sided_mcnemar_p": exact_mcnemar_p(a_only, b_only),
    }


def render_report(summary: dict[str, Any]) -> str:
    groups_by_k = {group["action_chunk_length"]: group for group in summary["groups"]}
    rows = []
    for group in summary["groups"]:
        low, high = group["wilson_95_percent_interval"]
        mean_success = group["mean_steps_to_success"]
        median_success = group["median_steps_to_success"]
        rows.append(
            "| {k} | {s}/{n} ({rate:.0%}) | {lo:.1%}–{hi:.1%} | {mean_hit:.1f} | {median_hit:.1f} | {hit_or_cap:.1f} | {mean_steps:.1f} | {queries:.1f} | {pad} | {wall:.1f} |".format(
                k=group["action_chunk_length"],
                s=group["success_count"],
                n=group["rollout_count"],
                rate=group["success_rate"],
                lo=low,
                hi=high,
                mean_hit=mean_success if mean_success is not None else float("nan"),
                median_hit=median_success if median_success is not None else float("nan"),
                hit_or_cap=group["mean_first_hit_or_cap_steps_per_rollout"],
                mean_steps=group["mean_environment_steps_per_rollout"],
                queries=group["mean_retrieval_queries_per_rollout"],
                pad=group["total_action_padding_count"] if group["total_action_padding_count"] is not None else "n/a",
                wall=group["evaluation_wall_seconds"],
            )
        )
    comparisons = []
    for item in summary["paired_comparisons"]:
        comparisons.append(
            "| {comparison} | {both} | {ao} | {bo} | {neither} | {diff:+.1%} | {p:.4f} |".format(
                comparison=item["comparison"],
                both=item["both_success"],
                ao=item["a_only_success"],
                bo=item["b_only_success"],
                neither=item["neither_success"],
                diff=item["success_rate_difference_b_minus_a"],
                p=item["exact_two_sided_mcnemar_p"],
            )
        )
    code_records = [f"K={k}: `{groups_by_k[k]['source_code_commit']}`" for k in (3, 5, 7, 9)]
    script_hashes = {groups_by_k[k]["evaluation_script_sha256"] for k in (3, 5, 7, 9)}
    plan_hashes = {groups_by_k[k]["ablation_plan_sha256"] for k in (3, 5, 7, 9)}
    script_hash = next(iter(script_hashes)) if len(script_hashes) == 1 else "MISMATCH"
    plan_hash = next(iter(plan_hashes)) if len(plan_hashes) == 1 else "MISMATCH"
    quantiles = "; ".join(
        f"K={g['action_chunk_length']}: {g['success_step_p25']:.1f}–{g['success_step_p75']:.1f}"
        for g in summary["groups"]
    )
    provenance = (
        "The evaluator source file was unchanged between runs. Recorded code commits: "
        + "; ".join(code_records)
        + f". Evaluator script SHA-256: `{script_hash}`; frozen plan SHA-256: `{plan_hash}`. "
        + "The repository HEAD advanced during the long evaluation sequence; all four evaluations record identical evaluator and plan hashes.\n\n"
    )
    header = """# ALT-Lift action-chunk first-hit ablation

## Protocol

This is a no-retraining ablation of the completed ALT-30 checkpoint and memory bank. Each K means one cosine top-1 query followed by up to K consecutive original 7D incremental OSC_POSE/Panda actions from the matched demonstration episode, then a fresh observation and query. The RoboSuite Lift predicate is checked after every individual action. The evaluator stops immediately the first time cube height is greater than the table offset by 0.04 m. The success signal is evaluator-only; the policy receives no reward or oracle state. Unsuccessful rollouts continue to the unchanged 1,000-step cap. Same-episode terminal padding is retained and counted.

All variants use the same checkpoint, bank, dataset and split fingerprints, reset fixtures, and 50 environment seeds (`100000`–`100049`). Pilot runs on seed `100000` are excluded. K=1 is the existing `evaluation01`: it ran every rollout for 1,000 steps, but its recorded first-success indicator and first-hit step are used as a prefix-equivalent reference. The mean first-hit/cap step count is derived for K=1 by using its recorded first-hit step for successes and 1,000 for failures; this is the counterfactual stopping cost, not a separately run early-stop evaluation. K=1's recorded mean executed steps, retrieval count, and runtime are from the full 1,000-step runs.

"""
    table_header = """## Results

| K actions per query | First-hit success | Wilson 95% interval | Mean steps to success | Median steps to success | Mean first-hit/cap steps per seed | Mean executed steps / rollout | Mean retrieval queries / actual rollout | Total padded actions | Evaluation wall time (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    paired_header = """
## Paired outcomes by reset seed

`a_only` means success under the first K but failure under the second; `b_only` means the reverse. The p-value is a two-sided exact McNemar test on discordant seeds; the comparisons are descriptive and exploratory, and no multiplicity correction is applied.

| Comparison | Both succeed | A only | B only | Neither | B − A success rate | Exact p |
|---|---:|---:|---:|---:|---:|---:|
"""
    interpretation = """
## Interpretation and limits

The best observed point estimate is K=5; the four chunk settings are close and their Wilson intervals overlap substantially. A single trained checkpoint was evaluated, so these intervals describe variation across simulator reset seeds only and do not measure training-seed uncertainty. First-hit completion says that Lift height was reached once; it does not claim the object was held or that the controller learned an endpoint. The action-chunk setting changes both open-loop duration and how often visual retrieval is refreshed, so this is an execution-horizon comparison, not a training-method comparison.

All 50 rollout rows and per-seed first-hit steps are tracked with each evaluation. Videos and detailed per-query action/retrieval traces are local, Git-ignored artifacts. The evaluator reports action-padding totals because repeated final demonstration actions are part of this unchanged retrieval boundary policy.

## Reproduction

From this experiment folder, run `scripts/summarize_action_chunk_ablation.py`. To reproduce the simulator measurements themselves, see the four frozen commands in `README.md`; the original local checkpoint and bank are required. Each evaluation stores its resolved setup and artifact checksums under `experiments/runs/alt_lift_ph30_image_proprio_seed42_v1/evaluations/`.
"""
    return (
        header
        + provenance
        + table_header
        + "\n".join(rows)
        + "\n\nSuccessful-rollout step quantiles (25th–75th percentile): "
        + quantiles
        + ".\n"
        + paired_header
        + "\n".join(comparisons)
        + "\n"
        + interpretation
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("experiments/runs/alt_lift_ph30_image_proprio_seed42_v1"),
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--report", type=Path, default=Path("docs/action_chunk_ablation.md"))
    args = parser.parse_args()

    groups = {k: load_group(args.run_dir, k, evaluation_id) for k, evaluation_id in EVALUATIONS.items()}
    seed_list = groups[1]["seeds"]
    if seed_list != list(range(100000, 100050)):
        raise ValueError("The fixed evaluation seed set differs from the frozen 100000..100049 contract")
    manifest_hashes = {g["evaluation_seed_manifest_sha256"] for g in groups.values()}
    split_hashes = {g["split_manifest_sha256"] for g in groups.values()}
    dataset_hashes = {g["dataset_fingerprint"] for g in groups.values()}
    if len(manifest_hashes) != 1 or len(split_hashes) != 1 or len(dataset_hashes) != 1:
        raise ValueError("Evaluation seed, split, or dataset fingerprints differ across groups")
    formal_groups = [groups[k] for k in (3, 5, 7, 9)]
    if len({g["evaluation_script_sha256"] for g in formal_groups}) != 1:
        raise ValueError("Evaluator script hash differs across formal chunk groups")
    if len({g["ablation_plan_sha256"] for g in formal_groups}) != 1:
        raise ValueError("Frozen ablation plan hash differs across formal chunk groups")
    if len({g["checkpoint_sha256"] for g in formal_groups}) != 1:
        raise ValueError("Checkpoint hash differs across formal chunk groups")
    if len({g["memory_bank_sha256"] for g in formal_groups}) != 1:
        raise ValueError("Memory bank hash differs across formal chunk groups")

    reference_states = groups[1]["initial_state_hashes"]
    for k, group in groups.items():
        if group["initial_state_hashes"] != reference_states:
            raise ValueError(f"Initial-state fixture hashes differ from K=1 in K={k}")

    comparisons = [
        paired_comparison(groups, 1, k) for k in (3, 5, 7, 9)
    ] + [paired_comparison(groups, 3, 5), paired_comparison(groups, 5, 7), paired_comparison(groups, 7, 9)]
    public_groups = [{key: value for key, value in g.items() if key not in {"rows_by_seed", "seeds", "success_seeds", "initial_state_hashes"}} for g in groups.values()]
    per_seed = {
        str(seed): {f"K{k}": parse_bool(groups[k]["rows_by_seed"][seed]["success"]) for k in groups}
        for seed in seed_list
    }
    summary = {
        "schema_version": 1,
        "run_id": "alt_lift_ph30_image_proprio_seed42_v1",
        "dataset_fingerprint": next(iter(dataset_hashes)),
        "split_manifest_sha256": next(iter(split_hashes)),
        "evaluation_seed_manifest_sha256": next(iter(manifest_hashes)),
        "evaluation_seeds": seed_list,
        "success_definition": next(iter(groups.values()))["success_definition"],
        "groups": public_groups,
        "paired_comparisons": comparisons,
        "per_seed_success": per_seed,
        "inference_scope": "one trained model seed; Wilson intervals describe evaluation-seed uncertainty only",
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "action_chunk_firsthit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    columns = [
        "action_chunk_length", "evaluation_id", "success_count", "rollout_count", "success_rate",
        "wilson_95_percent_low", "wilson_95_percent_high", "mean_steps_to_success",
        "median_steps_to_success", "success_step_p25", "success_step_p75",
        "mean_environment_steps_per_rollout", "mean_retrieval_queries_per_rollout",
        "mean_first_hit_or_cap_steps_per_rollout",
        "total_retrieval_queries", "total_action_padding_count",
        "mean_retrieval_episode_switches_per_rollout", "evaluation_wall_seconds",
    ]
    with (args.results_dir / "action_chunk_firsthit_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for group in public_groups:
            low, high = group["wilson_95_percent_interval"]
            writer.writerow({
                key: group[key] for key in columns if key in group
            } | {"wilson_95_percent_low": low, "wilson_95_percent_high": high})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
