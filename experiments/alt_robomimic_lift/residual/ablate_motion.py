"""Compare translation-only and rotation-only corrections without fitting new weights."""
import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from study import (DESIGN, HERE, LIFT, Head, check_trace, load_base, online,
                   paired_counts, read_json, run_rollouts, sha256, write_json)


def main(source, run_id):
    local = LIFT / "artifacts/residual"
    parent = local / "lift100_numeric_v1"
    baseline = local / "lift100_residual_v1/baseline"
    training = read_json(parent / "training.json")
    epoch = read_json(parent / "selection.json")["epoch"]
    full = parent / f"test_e{epoch}"
    protocol = read_json(full / "protocol.json")
    baseline_protocol = read_json(baseline / "protocol.json")
    keys = ("seeds", "horizon", "fixture_hash", "config", "bank", "execution_horizon")
    if any(protocol[key] != baseline_protocol[key] for key in keys):
        raise ValueError("Existing control groups differ in protocol")
    check_trace(full, protocol)
    check_trace(baseline, baseline_protocol)
    policy, base, _, _, cfg = load_base(source)
    checkpoint = parent / f"epoch{epoch}.pt"
    digest = sha256(checkpoint)
    if (digest != training["checkpoints"][str(epoch)] or digest != protocol["checkpoint"]
            or base["bank_sha256"] != protocol["bank"] or cfg != protocol["config"]
            or base["checkpoint_sha256"] != baseline_protocol["checkpoint"]
            or base["checkpoint_sha256"] != training["base_checkpoint_sha256"]):
        raise ValueError("The frozen policy differs from the controls")
    head = Head(training["design"]["network"][0]).eval()
    head.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    output = local / run_id
    output.mkdir(exist_ok=True)
    groups = {"translation": [True]*3 + [False]*3, "rotation": [False]*3 + [True]*3}
    summary = {"baseline": read_json(baseline/"metrics.json"), "full": read_json(full/"metrics.json")}
    ratings = {}
    for name, mask in groups.items():
        out = output / name
        resolved = {**protocol, "correction_mask": mask, "encoder_sha256": base["checkpoint_sha256"],
                    "source_sha256": sha256(__file__), "inference_sha256": sha256(HERE/"study.py"),
                    "workers": DESIGN["workers"]}
        if out.exists():
            if read_json(out/"protocol.json") != resolved:
                raise ValueError("Existing run uses a different protocol")
            if not (out/"metrics.json").exists():
                out.rename(out.with_name(f"{name}_incomplete_{time.time_ns()}"))
        if not out.exists():
            out.mkdir()
            write_json(out/"protocol.json", resolved)
            print(f"Evaluating {name}", flush=True)
            run_rollouts(online(policy, head, np.asarray(mask)), cfg, LIFT, protocol["seeds"],
                         protocol["horizon"], DESIGN["workers"], out)
        check_trace(out, resolved)
        summary[name] = read_json(out/"metrics.json")
        ratings[name] = {"vs_alt": paired_counts(baseline,out,resolved), "vs_full": paired_counts(full,out,resolved)}
    ratings["rotation_vs_translation"] = paired_counts(output/"translation", output/"rotation", protocol)
    dest = HERE/"results"/run_id
    dest.mkdir(parents=True, exist_ok=False)
    for name in groups:
        for filename in ("rollouts.csv", "protocol.json"):
            shutil.copyfile(output/name/filename, dest/f"{name}_{filename}")
    manifest = []
    for name in groups:
        path = output/name/"actions.csv"
        manifest.append(dict(path=path.relative_to(LIFT).as_posix(), bytes=path.stat().st_size, sha256=sha256(path)))
    write_json(dest/"artifact_manifest.json", manifest)
    write_json(dest/"metrics.json", summary)
    write_json(dest/"paired.json", ratings)
    print({key: value["hold_success"]["count"] for key,value in summary.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alt100-source", required=True, type=Path)
    parser.add_argument("run_id")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        parser.error("Use a single directory name for the run")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    main(args.alt100_source, args.run_id)
