"""Create the immutable, seed-42 Lift-PH data/evaluation contract once."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py


HERE = Path(__file__).resolve().parent.parent
DEFAULT_DATA_CONFIG = HERE / "configs" / "data_paths.json"
DEFAULT_OUTPUT = HERE / "configs" / "lift_ph_split42.json"
DEFAULT_EVAL_CONDITIONS = HERE / "results" / "evaluation_initial_states.json"
DEFAULT_RUN_CONFIG = HERE / "configs" / "alt_lift30.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value: str, base: Path) -> Path:
    if len(value) > 2 and value[1] == ":" and value[2] in ("\\", "/"):
        return Path("/mnt") / value[0].lower() / value[3:].replace("\\", "/")
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def demo_sort_key(episode_id: str) -> int:
    if not episode_id.startswith("demo_"):
        raise ValueError("Unexpected episode ID: %r" % episode_id)
    return int(episode_id.split("_", 1)[1])


def build_contract(
    episode_ids: list[str],
    image_path: Path,
    low_dim_path: Path,
    schema_fingerprint: str,
    source_mask_counts: dict[str, int],
    evaluation_conditions_path: Path,
    run_config_path: Path,
    split_seed: int = 42,
) -> dict[str, Any]:
    episode_ids = sorted(episode_ids, key=demo_sort_key)
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("Dataset episode IDs are not unique")
    if len(episode_ids) < 120:
        raise ValueError("Need at least 20 validation + 100 ordered training episodes")
    shuffled = list(episode_ids)
    random.Random(split_seed).shuffle(shuffled)
    heldout = shuffled[:20]
    ordered_train = shuffled[20:]
    if len(heldout) != 20 or len(ordered_train) < 100:
        raise ValueError("Not enough unique episodes for the configured future nested split")
    subsets = {str(n): ordered_train[:n] for n in (30, 50, 100)}
    if any(len(values) != int(scale) for scale, values in subsets.items()):
        raise ValueError("A requested split size is unavailable")
    if set(heldout) & set(ordered_train):
        raise ValueError("Validation and train episodes overlap")

    eval_conditions = json.loads(evaluation_conditions_path.read_text(encoding="utf-8"))
    if not eval_conditions["seed_contract"]["all_repeat_checks_passed"]:
        raise ValueError("Evaluation reset repeat checks did not pass")
    seeds = eval_conditions["seed_contract"]["seeds"]
    source_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "image_sha256": sha256_file(image_path),
                "low_dim_sha256": sha256_file(low_dim_path),
                "schema_fingerprint": schema_fingerprint,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema_version": 1,
        "contract_id": "lift_ph_split42_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": "robomimic_lift_ph_v0.2.0_image",
        "dataset_fingerprint": source_fingerprint,
        "dataset_schema_fingerprint": schema_fingerprint,
        "source_files": {
            "image_hdf5": {
                "path_from_repo_root": "artifacts/datasets/robomimic/lift/ph/image.hdf5",
                "bytes": image_path.stat().st_size,
                "sha256": sha256_file(image_path),
            },
            "low_dim_hdf5": {
                "path_from_repo_root": "artifacts/datasets/robomimic/lift/ph/low_dim.hdf5",
                "bytes": low_dim_path.stat().st_size,
                "sha256": sha256_file(low_dim_path),
            },
        },
        "episode_count_total": len(episode_ids),
        "split_seed": split_seed,
        "selection_algorithm": "Sort all demo IDs by numeric suffix, shuffle once with Python random.Random(42), reserve the first 20 for offline validation, and use the remaining order as the nested training pool. Exact IDs are authoritative; never regenerate a run split.",
        "heldout_validation_episode_ids": heldout,
        "ordered_train_pool_episode_ids": ordered_train,
        "train_subsets": subsets,
        "official_source_masks_audited_but_not_used_for_this_custom_split": {
            "train_count": int(source_mask_counts.get("train", 0)),
            "valid_count": int(source_mask_counts.get("valid", 0)),
            "selection_note": "The benchmark masks are preserved in dataset_audit.json. This experiment uses the required seed-42 permutation over all distinct source episodes; the custom held-out list is disjoint from every training subset."
        },
        "training_now": {
            "scale": 30,
            "episode_ids": subsets["30"],
            "run_id": "alt_lift_ph30_image_proprio_seed42_v1",
            "model_seed": 42,
            "dataloader_shuffle_seed": 43,
            "augmentation_seed": 44,
        },
        "test_env_seeds": seeds,
        "evaluation_initial_state_contract": {
            "path": "results/evaluation_initial_states.json",
            "sha256": sha256_file(evaluation_conditions_path),
            "repeat_checks_passed": bool(eval_conditions["seed_contract"]["all_repeat_checks_passed"]),
        },
        "evaluation_config": {
            "path": "configs/evaluation.json",
            "sha256": sha256_file(HERE / "configs" / "evaluation.json"),
        },
        "run_config": {
            "path": "configs/alt_lift30.yaml",
            "sha256": sha256_file(run_config_path),
        },
        "notes": "Only the 30-episode subset is trained in this study. The nested 50 and 100 lists are reserved for a future explicitly authorized study. Validation demos are offline diagnostics only and never enter the memory bank or normalizer fit.",
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--audit", type=Path, default=HERE / "results" / "dataset_audit.json")
    parser.add_argument("--eval-conditions", type=Path, default=DEFAULT_EVAL_CONDITIONS)
    parser.add_argument("--run-config", type=Path, default=DEFAULT_RUN_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Frozen split already exists; inspect and validate it instead of rewriting it")
    data_config = json.loads(args.data_config.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    image_path = resolve_path(data_config["image_hdf5"], HERE)
    low_dim_path = resolve_path(data_config["low_dim_hdf5"], HERE)
    with h5py.File(str(image_path), "r") as h5:
        episode_ids = [key for key in h5["data"].keys() if key.startswith("demo_")]
        source_mask_counts = {key: len(h5["mask"][key]) for key in h5["mask"]}
    for key, path in (("image_dataset", image_path), ("low_dim_dataset", low_dim_path)):
        expected = audit[key]["sha256"]
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError("Dataset hash changed since the audit: %s" % key)
    if len(episode_ids) != audit["image_dataset"]["episode_count"]:
        raise ValueError("Episode count changed since the audit")
    payload = build_contract(
        episode_ids=episode_ids,
        image_path=image_path,
        low_dim_path=low_dim_path,
        schema_fingerprint=audit["schema_fingerprint"],
        source_mask_counts=source_mask_counts,
        evaluation_conditions_path=args.eval_conditions,
        run_config_path=args.run_config,
        split_seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "dataset_fingerprint": payload["dataset_fingerprint"],
        "validation_count": len(payload["heldout_validation_episode_ids"]),
        "train_counts": {key: len(value) for key, value in payload["train_subsets"].items()},
        "training_ids": payload["training_now"]["episode_ids"],
        "test_seed_count": len(payload["test_env_seeds"]),
    }, indent=2))


if __name__ == "__main__":
    main()
