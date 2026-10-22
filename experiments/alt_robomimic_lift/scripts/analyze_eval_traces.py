#!/usr/bin/env python
"""Summarize fixed-run retrieval traces and sample the already-recorded videos."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import imageio.v2 as imageio

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from alt_lift.repro import json_write, resolve_config_path


def _vector(text):
    return np.asarray(json.loads(text), dtype=np.float64)


def _summarize(rows, result_by_seed):
    groups = defaultdict(list)
    for row in rows:
        groups[int(row["eval_seed"])].append(row)
    by_seed = []
    for seed in sorted(groups):
        current = groups[seed]
        actions = np.stack([_vector(row["actual_action"]) for row in current])
        pos_delta = np.stack([_vector(row["query_eef_pos"]) - _vector(row["matched_eef_pos"]) for row in current])
        episode_counts = Counter(row["matched_episode_id"] for row in current)
        frame_counts = Counter((row["matched_episode_id"], int(row["matched_local_frame_index"])) for row in current)
        scores = np.asarray([float(row["top1_cosine_similarity"]) for row in current])
        gripper = actions[:, 6]
        row = result_by_seed[seed]
        by_seed.append({
            "eval_seed": seed,
            "success": bool(row["success"]),
            "steps": int(row["steps"]),
            "steps_to_success": row["steps_to_success"],
            "query_count": len(current),
            "mean_top1_cosine": float(scores.mean()),
            "mean_eef_position_gap_m": float(np.linalg.norm(pos_delta, axis=1).mean()),
            "retrieval_episode_switches": int(sum(int(x["episode_switch"]) for x in current)),
            "retrieval_frame_jumps_over_8": int(sum(int(x["frame_jump_over_8"]) for x in current)),
            "same_frame_repeat_fraction": float(np.mean([int(x["same_frame_repeat"]) for x in current])),
            "unique_matched_episode_count": len(episode_counts),
            "unique_matched_frame_count": len(frame_counts),
            "most_retrieved_episode_id": episode_counts.most_common(1)[0][0],
            "most_retrieved_frame": {"episode_id": frame_counts.most_common(1)[0][0][0], "local_frame_index": frame_counts.most_common(1)[0][0][1], "count": frame_counts.most_common(1)[0][1]},
            "mean_abs_action_per_dimension": np.abs(actions).mean(axis=0).tolist(),
            "gripper_open_command_fraction": float(np.mean(gripper < -1e-6)),
            "gripper_close_command_fraction": float(np.mean(gripper > 1e-6)),
        })
    return by_seed


def _group_summary(rows):
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        "mean_top1_cosine": float(np.mean([row["mean_top1_cosine"] for row in rows])),
        "mean_eef_position_gap_m": float(np.mean([row["mean_eef_position_gap_m"] for row in rows])),
        "mean_same_frame_repeat_fraction": float(np.mean([row["same_frame_repeat_fraction"] for row in rows])),
        "mean_episode_switches": float(np.mean([row["retrieval_episode_switches"] for row in rows])),
        "mean_frame_jumps_over_8": float(np.mean([row["retrieval_frame_jumps_over_8"] for row in rows])),
        "mean_unique_matched_frames": float(np.mean([row["unique_matched_frame_count"] for row in rows])),
    }


def _video_contact_sheet(artifact_dir: Path, output: Path):
    seeds = (100000, 100025, 100049)
    sample_steps = (0, 200, 400, 600, 800, 1000)
    tile_w, tile_h, label_h = 168, 84, 18
    sheet = Image.new("RGB", (len(sample_steps) * tile_w, len(seeds) * (tile_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    sizes = {}
    for row, seed in enumerate(seeds):
        path = artifact_dir / "evaluations" / "evaluation01" / "videos" / ("seed_%d.mp4" % seed)
        if not path.is_file():
            raise FileNotFoundError("Missing predeclared rollout video: %s" % path)
        reader = imageio.get_reader(str(path))
        first = reader.get_data(0)
        sizes[str(seed)] = [int(first.shape[1]), int(first.shape[0])]
        if first.shape[:2] != (84, 168):
            raise ValueError("Unexpected side-by-side camera video size for seed %d: %r" % (seed, first.shape))
        for col, step in enumerate(sample_steps):
            frame = reader.get_data(step)
            sheet.paste(Image.fromarray(frame), (col * tile_w, row * (tile_h + label_h)))
            draw.text((col * tile_w + 2, row * (tile_h + label_h) + tile_h + 1), "seed %d step %d" % (seed, step), fill="black")
        reader.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evaluation-id", type=str, default="evaluation01")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "failure_diagnostics.json")
    parser.add_argument("--figure", type=Path, default=HERE / "results" / "figures" / "eval_video_contact_sheet.png")
    args = parser.parse_args()
    if args.output.exists() or args.figure.exists():
        raise FileExistsError("Refusing to overwrite trace diagnostics; choose new --output and --figure paths")
    run_dir = args.run_dir.resolve()
    artifact_root = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))["artifact_root"]
    artifact_dir = resolve_config_path(artifact_root)
    result = json.loads((run_dir / "evaluations" / args.evaluation_id / "eval_metrics.json").read_text(encoding="utf-8"))
    result_by_seed = {int(row["eval_seed"]): row for row in result["per_seed"]}
    trace_path = artifact_dir / "evaluations" / args.evaluation_id / "retrieval_log.csv"
    with trace_path.open("r", newline="", encoding="utf-8") as stream:
        trace_rows = list(csv.DictReader(stream))
    by_seed = _summarize(trace_rows, result_by_seed)
    successes = [row for row in by_seed if row["success"]]
    failures = [row for row in by_seed if not row["success"]]
    sizes = _video_contact_sheet(artifact_dir, args.figure)
    payload = {
        "status": "completed",
        "run_id": result["run_id"],
        "evaluation_id": result["evaluation_id"],
        "trace_rows": len(trace_rows),
        "failed_seeds": [row["eval_seed"] for row in failures],
        "successful_seeds": [row["eval_seed"] for row in successes],
        "success_group_retrieval_diagnostics": _group_summary(successes),
        "failure_group_retrieval_diagnostics": _group_summary(failures),
        "per_seed": by_seed,
        "predeclared_video_frame_sizes": sizes,
        "video_contact_sheet": str(args.figure),
        "failure_analysis_note": "Retrieval/action patterns are diagnostics, not causal explanations. No policy or evaluation settings were changed after observing outcomes.",
    }
    json_write(args.output, payload)
    print(json.dumps({k: v for k, v in payload.items() if k != "per_seed"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
