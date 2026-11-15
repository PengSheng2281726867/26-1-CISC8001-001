"""Build the presentation figures and their tabular data from tracked results."""
import csv
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/presentation/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
import numpy as np

LIFT = ROOT / "experiments/lift_box_dp"
DP = ROOT / "experiments/pusht_dp"
RESIDUAL = ROOT / "experiments/pusht_residual/experiments"
COLORS = {"ALT": "#286799", "DP": "#A6462B"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def style(axis, limit=1.1):
    axis.set_ylim(0, limit)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#D7DFE6", linewidth=0.8)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#64748B")
    axis.tick_params(axis="both", length=0, pad=8)


def finish(figure, stem):
    figure.savefig(stem.with_suffix(".png"), dpi=200, facecolor="white")
    svg = stem.with_suffix(".svg")
    figure.savefig(svg, facecolor="white", metadata={"Date": None})
    lines = svg.read_text(encoding="utf-8").splitlines()
    svg.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
    plt.close(figure)


def fixed_budget():
    evidence = ROOT / "experiments/matched_budget/evidence"
    destination = HERE / "materials/matched_budget"
    destination.mkdir(parents=True, exist_ok=True)
    tasks = (
        ("lift", "Lift-PH", "24 validation starts, 250 steps",
         (("ever_success", "Ever lifted"), ("hold_success", "Held ≥1 s"), ("final_success", "Final step"))),
        ("pusht", "Image Push-T", "50 test starts, at most 300 steps",
         (("official_episode_score", "Peak coverage score"), ("final_score", "Final coverage score"))),
    )
    for task, title, evaluation, metrics in tasks:
        source = evidence / f"{task}_models.csv"
        rows = read_rows(source)
        updates, = {int(row["optimizer_steps"]) for row in rows}
        write_rows(destination / f"{task}.csv", [{**r, "source": source.relative_to(ROOT).as_posix()} for r in rows])
        scales = sorted({int(row["demonstrations"]) for row in rows})
        positions = np.arange(len(scales))
        figure, axes = plt.subplots(1, len(metrics), figsize=(11.4, 4.5), sharey=True)
        ceiling = 1.05 if task == "lift" else max(.5, max(float(row[key]) for row in rows for key, _ in metrics) + .05)
        for axis, (key, label) in zip(axes, metrics):
            for offset, method in ((-.09, "ALT"), (.09, "DP")):
                values = np.array([[float(row[key]) for row in rows if row["method"] == method.lower()
                                    and int(row["demonstrations"]) == scale] for scale in scales])
                axis.plot(positions + offset, values.mean(axis=1), "D-", color=COLORS[method],
                          linewidth=1.8, markersize=6, label=method)
                for index, measurements in enumerate(values):
                    points = positions[index] + offset + np.linspace(-.04, .04, len(measurements))
                    axis.scatter(points, measurements, color=COLORS[method], s=21, alpha=.55)
            style(axis, ceiling)
            axis.set_ylim(-.024 * ceiling, ceiling)
            axis.set_title(label, fontsize=13, weight="bold")
            axis.set_xticks(positions, scales)
            axis.set_xlabel("Demonstrations")
        axes[0].set_ylabel("Fraction of episodes" if task == "lift" else "Mean normalized coverage")
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper right", bbox_to_anchor=(.97, .98), frameon=False, ncol=2)
        figure.suptitle(f"{title} · {updates:,} optimizer updates per model", x=.08, y=.97,
                        ha="left", fontsize=17, weight="bold")
        figure.text(.08, .045, f"{evaluation}. Circles show individual training seeds and diamonds show their means.",
                    color="#475569")
        figure.subplots_adjust(left=.08, right=.97, bottom=.23, top=.77, wspace=.16)
        finish(figure, destination / task)


def lift():
    destination = HERE / "materials/lift"
    rows = []
    for demonstrations, protocol, horizon, cases in (
        (10, "formal test", 1000, [("ALT", 10, "alt10_epochs_v1/test_e010"),
                                 ("DP", 200, "dp10_epochs_v1/test_e200")]),
        (30, "formal test", 1000, [("ALT", 40, "alt30_seed42_v1/eval01"),
                                 ("DP", 200, "dp30_seed42_v1/eval01")]),
    ):
        for method, epoch, case in cases:
            source = LIFT / "evidence" / case / "rollouts.csv"
            rollouts = read_rows(source)
            assert len(rollouts) == 50 and all(int(r["steps"]) == horizon for r in rollouts)
            rows.append({"demonstrations": demonstrations, "method": method, "epoch": epoch,
                         "protocol": protocol, "horizon": horizon, "n": len(rollouts),
                         "ever": sum(r["ever_success"] == "True" for r in rollouts),
                         "hold": sum(r["hold_success"] == "True" for r in rollouts),
                         "final": sum(r["final_success"] == "True" for r in rollouts),
                         "source": source.relative_to(ROOT).as_posix()})
    for method in ("ALT", "DP"):
        source = (LIFT / "experiments/exp_004_lift100_epochs"
                  / f"validation_{method.lower()}100_epochs_v1.json")
        metrics = read_json(source)["epochs"]["40"]["combined24"]
        rows.append({"demonstrations": 100, "method": method, "epoch": 40,
                     "protocol": "validation", "horizon": 250, "n": 24,
                     "ever": metrics["ever_success"]["count"],
                     "hold": metrics["hold_success"]["count"],
                     "final": metrics["final_success"]["count"],
                     "source": source.relative_to(ROOT).as_posix()})
    write_rows(destination / "lift_outcomes.csv", rows)
    figure, axes = plt.subplots(1, 3, figsize=(12.8, 5.3), sharey=True)
    positions = np.arange(3)
    for j, (panel, count) in enumerate(zip(axes, (10, 30, 100))):
        for sign, method in ((-1, "ALT"), (1, "DP")):
            row = next(r for r in rows if r["demonstrations"] == count and r["method"] == method)
            values = [row[k] / row["n"] for k in ("ever", "hold", "final")]
            bars = panel.bar(positions + sign * .17, values, .32, color=COLORS[method], label=method)
            panel.bar_label(bars, labels=[str(row[k]) for k in ("ever", "hold", "final")], padding=4)
        suffix = "50 test episodes" if count < 100 else "24 validation episodes"
        panel.set_title(f"{count} demonstrations\n{suffix}", fontsize=13,
                        color="#667085" if count == 100 else "#182638")
        if count == 100:
            panel.set_facecolor("#F1F5F9")
        panel.set_xticks(positions, ["Ever lifted", "Held ≥1 s", "Final step"])
        style(panel, 1.12)
        if j == 0:
            panel.set_ylabel("Fraction of episodes")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right", bbox_to_anchor=(.97, .98), frameon=False, ncol=2)
    figure.suptitle("Lift-PH: reaching the height and keeping it", x=.071, y=.98, ha="left", weight="bold", fontsize=18)
    figure.text(.071, .04, "Bars show fractions; labels show counts.  Test: 1,000 steps  |  Validation: 250 steps", color="#475569")
    figure.subplots_adjust(left=.075, right=.975, bottom=.16, top=.74, wspace=.18)
    finish(figure, destination / "lift_outcomes")


def scores_figure(rows, destination, stem, title, footnote):
    write_rows(destination / (stem + ".csv"), rows)
    figure, axis = plt.subplots(figsize=(12.0, 5.2))
    positions = np.arange(len(rows))
    for offset, key, label, color in ((-.18, "peak", "Episode peak", "#286799"),
                                      (.18, "final", "Final step", "#BC632E")):
        bars = axis.bar(positions + offset, [r[key] for r in rows], .34, color=color, label=label)
        axis.bar_label(bars, fmt="%.3f", padding=4, fontsize=10)
    axis.set_xticks(positions, [r["label"] for r in rows], fontsize=10)
    axis.set_ylabel("Mean normalized coverage score")
    style(axis, 1.15)
    axis.legend(loc="upper left", bbox_to_anchor=(0, 1.05), ncol=2, frameon=False)
    figure.suptitle(title, x=.09, y=.98, ha="left", weight="bold", fontsize=18)
    figure.text(.09, .028, footnote, color="#475569")
    figure.subplots_adjust(left=.09, right=.98, bottom=.19, top=.82)
    finish(figure, destination / stem)


def visual():
    short = DP / "evidence/comparison.json"
    long = DP / "evidence_long/comparison.json"
    scales = read_json(short)["scales"]
    rows = []
    for scale in ("30", "100"):
        for model in ("alt", "dp"):
            result = scales[scale][model]
            label = f"{model.upper()} · {scale} demos" + ("\n5k updates" if model == "dp" else "")
            rows.append({"label": label, "n": result["n"], "peak": result["mean_score"],
                         "final": result["mean_final_score"], "successes": result["successes"],
                         "source": short.relative_to(ROOT).as_posix()})
    result = read_json(long)["scales"]["100"]
    rows.append({"label": "DP · 100 demos\n15k updates", "n": result["n"],
                 "peak": result["mean_score"], "final": result["mean_final_score"],
                 "successes": result["successes"], "source": long.relative_to(ROOT).as_posix()})
    scores_figure(rows, HERE / "materials/pusht_visual", "visual_scores",
                  "Push-T: RGB + pusher position", "50 shared test seeds per setting. Policy inputs are RGB and pusher position.")


def geometry():
    evidence = RESIDUAL / "network_study/evidence"
    raw = RESIDUAL / "residual_study/evidence/round3_evaluate_raw.csv"
    rows = []
    for label, source in (
        ("Replay as-is", raw),
        ("Pose alignment", evidence / "test_aligned_rollouts.csv"),
        ("Alignment +\nordinary residual", evidence / "test_round1_rollouts.csv"),
        ("Alignment +\ntapered residual", evidence / "test_round3_rollouts.csv"),
    ):
        rollouts = read_rows(source)
        assert [int(r["evaluation_seed"]) for r in rollouts] == list(range(100000, 100050))
        rows.append({"label": label, "n": len(rollouts),
                     "peak": np.mean([float(r["official_episode_score"]) for r in rollouts]),
                     "final": np.mean([float(r["final_score"]) for r in rollouts]),
                     "successes": sum(r["terminated"] == "True" for r in rollouts),
                     "source": source.relative_to(ROOT).as_posix()})
    scores_figure(rows[:2], HERE / "materials/pusht_geometry", "alignment",
                  "Object-frame alignment", "Same relative-geometry rule, with simulator ground-truth T pose. 50 shared test seeds.")
    scores_figure(rows[1:], HERE / "materials/pusht_geometry", "residual",
                  "How much should the network change?", "All three use simulator ground-truth T pose. 50 shared test seeds.")
    reference, tapered = [read_rows(ROOT / r["source"]) for r in (rows[1], rows[3])]
    delta = np.asarray([float(y["final_score"]) - float(x["final_score"])
                        for x, y in zip(reference, tapered)])
    draws = np.random.default_rng(2026).integers(0, len(delta), (10000, len(delta)))
    comparison = {"tapered_minus_aligned_final_mean": float(delta.mean()),
                  "paired_ci95": np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist(),
                  "eval_seeds": [int(r["evaluation_seed"]) for r in reference],
                  "bootstrap_seed": 2026, "n_resamples": 10000,
                  "sources": [rows[1]["source"], rows[3]["source"]]}
    (HERE / "materials/pusht_geometry/final_score_pairing.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8")


def diagrams():
    destination = HERE / "materials/methods"
    destination.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 3.7))
    axis.set(xlim=(0, 12), ylim=(0, 3.7))
    axis.axis("off")
    for y, method, stages in (
        (2.35, "ALT", ["Encode current\nobservation", "Find nearest\ntraining frame", "Replay retrieved\naction chunk"]),
        (1.00, "DP", ["Encode recent\nobservations", "Denoise an action\nsequence", "Execute the next\naction chunk"]),
    ):
        color = COLORS[method]
        axis.text(.15, y, method, va="center", color=color, weight="bold", fontsize=18)
        for index, label in enumerate(stages):
            x = 1.65 + index * 3.45
            axis.add_patch(FancyBboxPatch((x, y - .4), 2.55, .8,
                                         boxstyle="round,pad=.06", ec=color, fc="#F5F8FB", lw=1.5))
            axis.text(x + 1.275, y, label, va="center", ha="center", fontsize=12)
            if index < 2:
                axis.add_patch(FancyArrowPatch((x + 2.66, y), (x + 3.35, y),
                                               arrowstyle="-|>", mutation_scale=15, color=color))
    figure.text(.12, .06, "Conceptual overview. The executed chunk length follows each task protocol.", color="#475569")
    finish(figure, destination / "alt_vs_dp")

    figure, panels = plt.subplots(1, 3, figsize=(12, 4.25))
    outline = np.array([[-1.0, .72], [1, .72], [1, .36], [.2, .36], [.2, -.95],
                        [-.2, -.95], [-.2, .36], [-1, .36]])
    targets = np.array([[1.65, .17], [1.33, .30], [1.07, .42], [.85, .53]])
    offset = np.array([.32, -.17])
    theta = -.63
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    for index, (axis, title) in enumerate(zip(panels, ("Retrieved action",
                                                       "Object-frame alignment", "Learned residual"))):
        transform = (lambda points: points) if index == 0 else (lambda points: points @ rotation.T + offset)
        axis.add_patch(Polygon(transform(outline), closed=True, facecolor="#94A8BD", edgecolor="#526A81"))
        commands = transform(targets)
        axis.plot(commands[:, 0], commands[:, 1], "o-", color=COLORS["ALT"],
                  alpha=.75 if index == 2 else 1)
        if index == 2:
            corrections = np.linspace(1, 0, len(targets))[:, None] * np.array([-.05, .31])
            corrected = commands + corrections
            axis.plot(corrected[:, 0], corrected[:, 1], "o-", color=COLORS["DP"])
            for before, after in zip(commands[:-1], corrected[:-1]):
                axis.add_patch(FancyArrowPatch(before, after, arrowstyle="-|>", mutation_scale=10,
                                               color=COLORS["DP"], linewidth=1))
        axis.set(xlim=(-1.36, 2.13), ylim=(-1.40, 1.30), aspect="equal")
        axis.axis("off")
        axis.set_title(title, fontsize=13, weight="bold")
    figure.text(.08, .065, "Illustration. The experiment uses eight targets and simulator ground-truth T pose.", color="#475569")
    figure.subplots_adjust(left=.065, right=.955, top=.86, bottom=.15, wspace=.09)
    finish(figure, destination / "action_transfer")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path",
                         "svg.hashsalt": "course-presentation"})
    lift()
    visual()
    geometry()
    diagrams()
    fixed_budget()
