#!/usr/bin/env python
"""Run the frozen 50-condition RoboSuite Lift evaluation with the trained ALT policy."""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from pathlib import Path

import h5py
import numpy as np
import torch
import torchvision

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from alt_lift.data import PROPRIO_KEYS, LiftFrameStore, ProprioNormalizer
from alt_lift.encoder import VisualProprioEncoder
from alt_lift.memory import MemoryBank
from alt_lift.policy import ActionRetrievalPolicy
from alt_lift.repro import git_value, json_write, resolve_config_path, sha256_file


def _array_hash(values):
    arr = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(json.dumps(list(arr.shape)).encode("ascii"))
    digest.update(arr.tobytes())
    return digest.hexdigest()


def _policy_obs_hash(obs):
    digest = hashlib.sha256()
    keys = ("agentview_image", "robot0_eye_in_hand_image") + PROPRIO_KEYS
    for key in keys:
        arr = np.ascontiguousarray(obs[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(arr.dtype).encode("ascii"))
        digest.update(json.dumps(list(arr.shape)).encode("ascii"))
        digest.update(arr.tobytes())
    return digest.hexdigest()


def reset_to_frozen_seed(env, seed, fixture):
    """Recreate the audited seeded reset and verify state and policy pixels before acting."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    obs = env.reset()
    sim_state = np.asarray(env.get_state()["states"], dtype=np.float64)
    actual_state_hash = _array_hash(sim_state)
    if actual_state_hash != fixture["sim_state_sha256"]:
        raise RuntimeError("Frozen simulator state mismatch for seed %d" % int(seed))
    actual_obs_hash = _policy_obs_hash(obs)
    if actual_obs_hash != fixture["policy_observation_sha256"]:
        raise RuntimeError("Frozen policy observation mismatch for seed %d" % int(seed))
    return obs


def _wilson(successes, total, z=1.959963984540054):
    if total <= 0:
        return [None, None]
    p = successes / float(total)
    den = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / den
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / den
    return [max(0.0, center - half), min(1.0, center + half)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=HERE / "configs" / "alt_lift30.yaml")
    parser.add_argument("--paths-config", type=Path, default=HERE / "configs" / "runtime_paths.json")
    parser.add_argument("--eval-config", type=Path, default=HERE / "configs" / "evaluation.json")
    parser.add_argument("--initial-states", type=Path, default=HERE / "results" / "evaluation_initial_states.json")
    parser.add_argument("--evaluation-id", type=str, default="evaluation01")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "completed":
        raise ValueError("Training must complete before environment evaluation")
    artifact_path = run_dir / "artifact_manifest.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact_dir = resolve_config_path(metadata["artifact_root"])
    bank_path = artifact_dir / "memory_bank.npz"
    checkpoint_path = artifact_dir / "checkpoint_final.pt"
    if sha256_file(bank_path) != artifact["bank_sha256"] or sha256_file(checkpoint_path) != metadata["checkpoint_sha256"]:
        raise ValueError("Checkpoint or memory bank checksum mismatch")
    paths = json.loads(args.paths_config.read_text(encoding="utf-8"))
    path_config = __import__("yaml").safe_load(args.config.read_text(encoding="utf-8"))
    image_path = resolve_config_path(paths["image_hdf5"])
    low_dim_path = resolve_config_path(paths["low_dim_hdf5"])
    split_path = resolve_config_path(path_config["dataset"]["split_manifest"])
    split = json.loads(split_path.read_text(encoding="utf-8"))
    eval_config = json.loads(args.eval_config.read_text(encoding="utf-8"))
    fixtures = json.loads(args.initial_states.read_text(encoding="utf-8"))
    if sha256_file(split_path) != metadata["split_manifest_sha256"] or artifact["split_manifest_sha256"] != metadata["split_manifest_sha256"]:
        raise ValueError("The tracked split reference no longer matches this checkpoint and memory bank")
    if sha256_file(image_path) != split["source_files"]["image_hdf5"]["sha256"] or sha256_file(low_dim_path) != split["source_files"]["low_dim_hdf5"]["sha256"]:
        raise ValueError("Dataset source files changed after the run")
    if sha256_file(args.eval_config) != split["evaluation_config"]["sha256"]:
        raise ValueError("Evaluation settings changed after the split contract was frozen")
    if sha256_file(args.initial_states) != split["evaluation_initial_state_contract"]["sha256"]:
        raise ValueError("Evaluation reset fixtures changed after the split contract was frozen")
    seed_rows = {int(row["seed"]): row for row in fixtures["initial_conditions"]}
    seeds = [int(x) for x in eval_config["test_seeds"]]
    if seeds != [int(x) for x in fixtures["seed_contract"]["seeds"]]:
        raise ValueError("Evaluation seeds differ from the frozen reset fixture list")
    if metadata["evaluation_seed_manifest_sha256"] != sha256_file(args.initial_states):
        raise ValueError("Evaluation initial-state fixture checksum differs from the run contract")
    if list(metadata["train_episode_ids"]) != list(split["train_subsets"]["30"]):
        raise ValueError("Training episode IDs do not match the frozen 30-episode subset")
    if set(MemoryBank.load(bank_path).episode_ids.tolist()) != set(metadata["train_episode_ids"]):
        raise ValueError("Memory bank episode membership differs from the frozen training subset")
    with h5py.File(str(image_path), "r") as h5:
        data = h5["data"]
        train_ids = list(metadata["train_episode_ids"])
        actions_by_episode = {episode_id: np.asarray(data[episode_id]["actions"]) for episode_id in train_ids}
        env_meta = json.loads(data.attrs["env_args"])
        all_keys = sorted(data["demo_0"]["obs"].keys())
    rgb_keys = [key for key in all_keys if key.endswith("_image")]
    low_dim_keys = [key for key in all_keys if not key.endswith("_image")]

    # Set up the legacy RoboSuite environment exactly as in the Phase 0 validator.
    import robomimic
    import robosuite
    import mujoco_py
    from robomimic.utils import env_utils, obs_utils

    obs_utils.initialize_obs_utils_with_obs_specs({"obs": {"rgb": rgb_keys, "low_dim": low_dim_keys}})
    env_meta = json.loads(json.dumps(env_meta))
    env_meta["env_kwargs"].update({
        "ignore_done": False,
        "horizon": int(eval_config["horizon"]),
        "control_freq": int(eval_config["control_freq"]),
        "reward_shaping": bool(eval_config["reward_shaping"]),
    })
    env = env_utils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=True, use_image_obs=True)
    env.postprocess_visual_obs = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(1)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    normalizer = ProprioNormalizer(checkpoint["proprio_mean"].numpy(), checkpoint["proprio_std"].numpy())
    model = VisualProprioEncoder(checkpoint["variant"], pretrained_path=None)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device).eval()
    bank = MemoryBank.load(bank_path)
    policy = ActionRetrievalPolicy(model, bank, normalizer, actions_by_episode, device)

    # A short API-only check uses the first frozen initial condition without advancing it.
    first_seed = seeds[0]
    first_fixture = seed_rows[first_seed]
    obs = reset_to_frozen_seed(env, first_seed, first_fixture)
    smoke_action, smoke_info = policy.act(obs, chunk_length=1)
    if smoke_action.shape != (7,) or not np.isfinite(smoke_action).all() or smoke_info["episode_id"] not in train_ids:
        raise RuntimeError("One-step policy API smoke check failed")

    import imageio.v2 as imageio
    eval_dir = run_dir / "evaluations" / args.evaluation_id
    eval_artifact_dir = artifact_dir / "evaluations" / args.evaluation_id
    if eval_dir.exists() or eval_artifact_dir.exists():
        raise FileExistsError("Refusing to overwrite an evaluation attempt; choose a new --evaluation-id")
    if (HERE / "results" / "evaluation_summary.json").exists():
        raise FileExistsError("A completed evaluation summary already exists; preserve it and use a separate run")
    eval_dir.mkdir(parents=True)
    eval_artifact_dir.mkdir(parents=True)
    output_video_dir = eval_artifact_dir / "videos"
    output_video_dir.mkdir(parents=True, exist_ok=True)
    retrieval_path = eval_artifact_dir / "retrieval_log.csv"
    score_path = eval_dir / "eval_rollouts.csv"
    score_fields = ["run_id", "train_episode_count", "split_manifest_sha256", "train_seed", "eval_seed", "initial_state_sha256", "success", "steps_to_success", "steps", "terminated", "time_limit", "max_reward", "final_reward", "wall_seconds", "retrieval_switches_episode", "retrieval_jumps_over_8_frames", "repeated_same_frame_fraction", "mean_similarity", "mean_preprocess_ms", "mean_encoder_ms", "mean_search_ms", "mean_action_extraction_ms", "video_path"]
    retrieval_fields = ["eval_seed", "control_step", "matched_episode_id", "matched_local_frame_index", "matched_source_index", "top1_cosine_similarity", "query_eef_pos", "query_eef_quat", "query_gripper_qpos", "matched_eef_pos", "matched_eef_quat", "matched_gripper_qpos", "actual_action", "preprocess_ms", "encoder_ms", "search_ms", "action_extraction_ms", "episode_switch", "frame_jump_over_8", "same_frame_repeat"]
    rollout_rows = []
    retrieval_count = 0
    evaluation_start = time.perf_counter()
    attempts = metadata.setdefault("evaluation_attempts", [])
    attempt = {
        "evaluation_id": args.evaluation_id,
        "status": "running",
        "started_unix_time": time.time(),
        "evaluation_code_commit": git_value("rev-parse", "HEAD"),
        "evaluation_worktree_status_at_launch": git_value("status", "--short"),
    }
    attempts.append(attempt)
    json_write(metadata_path, metadata)
    try:
        with retrieval_path.open("w", newline="", encoding="utf-8") as rf, score_path.open("w", newline="", encoding="utf-8") as sf:
            retrieval_writer = csv.DictWriter(rf, fieldnames=retrieval_fields)
            score_writer = csv.DictWriter(sf, fieldnames=score_fields)
            retrieval_writer.writeheader()
            score_writer.writeheader()
            for episode_i, seed in enumerate(seeds):
                fixture = seed_rows[seed]
                policy.reset()
                obs = reset_to_frozen_seed(env, seed, fixture)
                save_video = seed in set(int(x) for x in eval_config.get("video_seeds", [100000, 100025, 100049]))
                video_path = output_video_dir / ("seed_%d.mp4" % seed) if save_video else None
                writer = imageio.get_writer(str(video_path), fps=int(eval_config["control_freq"]), codec="libx264", quality=7, macro_block_size=1) if save_video else None
                if writer is not None:
                    writer.append_data(np.concatenate((obs["agentview_image"], obs["robot0_eye_in_hand_image"]), axis=1))
                episode_start = time.perf_counter()
                prev_episode, prev_frame = None, None
                switch_count = jump_count = repeat_count = 0
                similarities, preprocess_times, encoder_times, search_times, action_times = [], [], [], [], []
                max_reward, final_reward, first_success_step = 0.0, 0.0, None
                ever_success = False
                done = False
                steps = 0
                while steps < int(eval_config["horizon"]) and not done:
                    action, query = policy.act(obs, chunk_length=1)
                    matched_episode = query["episode_id"]
                    matched_frame = int(query["local_frame_index"])
                    switched = prev_episode is not None and matched_episode != prev_episode
                    frame_jump = prev_episode == matched_episode and prev_frame is not None and abs(matched_frame - prev_frame) > 8
                    same_frame = prev_episode == matched_episode and prev_frame == matched_frame
                    switch_count += int(switched)
                    jump_count += int(frame_jump)
                    repeat_count += int(same_frame)
                    similarities.append(query["top1_cosine_similarity"])
                    preprocess_times.append(query["preprocess_seconds"])
                    encoder_times.append(query["encoder_seconds"])
                    search_times.append(query["search_seconds"])
                    action_times.append(query["action_extraction_seconds"])
                    command = np.asarray(action)
                    obs, reward, done, info = env.step(command)
                    steps += 1
                    final_reward = float(reward)
                    max_reward = max(max_reward, final_reward)
                    success = bool(env.is_success()["task"])
                    if success and not ever_success:
                        first_success_step = steps
                    ever_success = ever_success or success
                    qprop = query["query_proprio"]
                    mprop = query["matched_proprio"]
                    retrieval_writer.writerow({
                        "eval_seed": seed,
                        "control_step": steps - 1,
                        "matched_episode_id": matched_episode,
                        "matched_local_frame_index": matched_frame,
                        "matched_source_index": query["source_index"],
                        "top1_cosine_similarity": query["top1_cosine_similarity"],
                        "query_eef_pos": json.dumps(qprop[:3].tolist()),
                        "query_eef_quat": json.dumps(qprop[3:7].tolist()),
                        "query_gripper_qpos": json.dumps(qprop[7:9].tolist()),
                        "matched_eef_pos": json.dumps(mprop[:3].tolist()),
                        "matched_eef_quat": json.dumps(mprop[3:7].tolist()),
                        "matched_gripper_qpos": json.dumps(mprop[7:9].tolist()),
                        "actual_action": json.dumps(command.tolist()),
                        "preprocess_ms": query["preprocess_seconds"] * 1000.0,
                        "encoder_ms": query["encoder_seconds"] * 1000.0,
                        "search_ms": query["search_seconds"] * 1000.0,
                        "action_extraction_ms": query["action_extraction_seconds"] * 1000.0,
                        "episode_switch": int(switched),
                        "frame_jump_over_8": int(frame_jump),
                        "same_frame_repeat": int(same_frame),
                    })
                    retrieval_count += 1
                    if writer is not None:
                        writer.append_data(np.concatenate((obs["agentview_image"], obs["robot0_eye_in_hand_image"]), axis=1))
                    prev_episode, prev_frame = matched_episode, matched_frame
                    if steps % 100 == 0:
                        rf.flush()
                if writer is not None:
                    writer.close()
                row = {
                    "run_id": metadata["run_id"],
                    "train_episode_count": len(train_ids),
                    "split_manifest_sha256": metadata["split_manifest_sha256"],
                    "train_seed": metadata["hyperparameters"]["model_seed"],
                    "eval_seed": seed,
                    "initial_state_sha256": fixture["sim_state_sha256"],
                    "success": bool(ever_success),
                    "steps_to_success": first_success_step,
                    "steps": steps,
                    "terminated": bool(done and steps < int(eval_config["horizon"])),
                    "time_limit": bool(steps >= int(eval_config["horizon"]) and not done),
                    "max_reward": max_reward,
                    "final_reward": final_reward,
                    "wall_seconds": time.perf_counter() - episode_start,
                    "retrieval_switches_episode": switch_count,
                    "retrieval_jumps_over_8_frames": jump_count,
                    "repeated_same_frame_fraction": float(repeat_count / max(steps, 1)),
                    "mean_similarity": float(np.mean(similarities)),
                    "mean_preprocess_ms": float(np.mean(preprocess_times) * 1000.0),
                    "mean_encoder_ms": float(np.mean(encoder_times) * 1000.0),
                    "mean_search_ms": float(np.mean(search_times) * 1000.0),
                    "mean_action_extraction_ms": float(np.mean(action_times) * 1000.0),
                    "video_path": str(video_path) if video_path else "",
                }
                rollout_rows.append(row)
                score_writer.writerow(row)
                sf.flush()
                print("seed %d/%d success=%s steps=%d seconds=%.1f" % (seed, len(seeds), ever_success, steps, row["wall_seconds"]), flush=True)
    except Exception:
        attempt.update({"status": "failed", "error": traceback.format_exc(), "failed_unix_time": time.time()})
        json_write(metadata_path, metadata)
        raise
    finally:
        try:
            env.env.close()
        except Exception:
            try:
                env.close()
            except Exception:
                pass
    successes = int(sum(1 for row in rollout_rows if row["success"]))
    summary = {
        "status": "completed",
        "evaluation_id": args.evaluation_id,
        "run_id": metadata["run_id"],
        "train_episode_count": len(train_ids),
        "train_episode_ids": train_ids,
        "split_manifest_sha256": metadata["split_manifest_sha256"],
        "dataset_fingerprint": metadata["dataset_fingerprint"],
        "evaluation_seed_manifest_sha256": metadata["evaluation_seed_manifest_sha256"],
        "evaluation_seeds": seeds,
        "success_definition": eval_config["success_signal"],
        "success_count": successes,
        "rollout_count": len(rollout_rows),
        "success_rate": successes / float(len(rollout_rows)),
        "wilson_95_percent_interval": _wilson(successes, len(rollout_rows)),
        "model_seed_uncertainty_measured": False,
        "mean_steps": float(np.mean([row["steps"] for row in rollout_rows])),
        "mean_wall_seconds_per_rollout": float(np.mean([row["wall_seconds"] for row in rollout_rows])),
        "total_environment_steps": int(sum(row["steps"] for row in rollout_rows)),
        "total_retrieval_queries": retrieval_count,
        "mean_encoder_latency_ms": float(np.mean([row["mean_encoder_ms"] for row in rollout_rows])),
        "mean_search_latency_ms": float(np.mean([row["mean_search_ms"] for row in rollout_rows])),
        "mean_preprocess_latency_ms": float(np.mean([row["mean_preprocess_ms"] for row in rollout_rows])),
        "mean_action_extraction_latency_ms": float(np.mean([row["mean_action_extraction_ms"] for row in rollout_rows])),
        "episode_limit": int(eval_config["horizon"]),
        "control_freq_hz": int(eval_config["control_freq"]),
        "controller": env_meta["env_kwargs"]["controller_configs"],
        "environment_versions": {"robomimic": robomimic.__version__, "robosuite": robosuite.__version__, "mujoco_py": mujoco_py.__version__, "torch": torch.__version__, "torchvision": torchvision.__version__, "python": sys.version, "platform": platform.platform()},
        "torch_cpu_threads": torch.get_num_threads(),
        "artifact_paths": {"rollout_csv": str(score_path), "retrieval_csv": str(retrieval_path), "video_dir": str(output_video_dir)},
        "per_seed": rollout_rows,
        "evaluation_wall_seconds": time.perf_counter() - evaluation_start,
    }
    out = eval_dir / "eval_metrics.json"
    json_write(out, summary)
    eval_artifacts = [retrieval_path] + sorted(output_video_dir.glob("*.mp4"))
    eval_manifest = {
        "evaluation_id": args.evaluation_id,
        "artifacts": [
            {"path": str(path.relative_to(HERE)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in eval_artifacts
        ],
        "rollout_csv_path": str(score_path.relative_to(HERE)),
        "rollout_csv_sha256": sha256_file(score_path),
    }
    json_write(eval_dir / "artifact_manifest.json", eval_manifest)
    attempt.update({"status": "completed", "completed_unix_time": time.time(), "metrics_sha256": sha256_file(out), "artifact_manifest_sha256": sha256_file(eval_dir / "artifact_manifest.json")})
    metadata.update({"evaluation_status": "completed", "evaluation_success_count": successes, "evaluation_rollout_count": len(rollout_rows), "evaluation_success_rate": summary["success_rate"], "evaluation_metrics_path": str(out.relative_to(HERE)), "evaluation_metrics_sha256": sha256_file(out)})
    json_write(metadata_path, metadata)
    json_write(HERE / "results" / "evaluation_summary.json", summary)
    # Small durable score table mirrors the ignored detailed per-run trace.
    with (HERE / "results" / "per_seed_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=score_fields)
        writer.writeheader()
        writer.writerows(rollout_rows)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_seed"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
