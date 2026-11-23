"""Measure Lift phase mismatches and test oracle reranking among the visual top five."""
import argparse
import csv
from pathlib import Path
import sys
import time

import h5py
import numpy as np
from PIL import Image, ImageDraw
import torch

from study import DESIGN, HERE, LIFT, check_trace, load_base, paired_counts, read_json, run_rollouts, sha256, write_json
from evaluate import Simulator

PHASES = ("no bilateral contact below threshold", "bilateral contact below threshold", "above height threshold")


class PhaseSimulator(Simulator):
    def __init__(self, alt, log):
        super().__init__(alt, log, entrypoint=HERE / "phase_simulator.py")

    def send(self, command, value):
        result = super().send(command, value)
        if command == "reset":
            self.seed, self.step = value, 0
            result.update(seed=self.seed, step=self.step)
        elif command == "step":
            self.step += 1
            result[0].update(seed=self.seed, step=self.step)
        return result


def make_gallery(path, obs, bank, candidates, query_phase, phases, selected, data):
    board = Image.new("RGB", (6*180, 216), "#f6f7fb")
    draw = ImageDraw.Draw(board)
    for i, row in enumerate([None, *candidates]):
        if row is None:
            source = obs
            line = f"Current t={obs['step']} | phase {query_phase}"
        else:
            episode, step = bank.episode_ids[row], bank.local_frame_indices[row]
            source = data["data"][episode]["obs"]
            line = f"#{i} {episode}:{step} | phase {phases[row]}"
        draw.text((i*180+5, 5), line, fill="#172134")
        if row is not None and row == selected:
            draw.text((i*180+5, 20), "Selected by phase rule", fill="#075f56")
        for j, camera in enumerate(("agentview_image","robot0_eye_in_hand_image")):
            pixels = source[camera] if row is None else source[camera][step]
            board.paste(Image.fromarray(pixels), (i*180+48, 34+j*88))
    board.save(path)


def actuator(policy, phases, filtered, trace, gallery, data, gallery_seeds):
    saved = set()
    @torch.inference_mode()
    def act(observations):
        actions = []
        for obs in observations:
            prior, info = policy.act(obs, include_embedding=True)
            q = torch.as_tensor(info["query_embedding"], device=policy.device).reshape(1,-1)
            scores = torch.mv(policy.search.embeddings, torch.nn.functional.normalize(q, dim=1).flatten())
            top = torch.argsort(scores, descending=True, stable=True)[:5].cpu().tolist()
            if top[0] != info["bank_row"]:
                # Preserve the actual baseline top-1 even in a near numerical tie.
                top = [info["bank_row"], *[row for row in top if row != info["bank_row"]]][:5]
            phase = obs["oracle_phase"]
            matches = [row for row in top if phases[row] == phase]
            choice = matches[0] if filtered and matches else top[0]
            trace.writerow([obs["seed"], obs["step"], phase, *top, choice])
            sample = (obs["seed"], phase)
            if not filtered and obs["seed"] in gallery_seeds and not phases[top[0]] == phase and sample not in saved:
                output = gallery / f"seed{sample[0]}_phase{phase}_step{obs['step']}.jpg"
                make_gallery(output, obs, policy.bank, top, phase, phases, matches[0] if matches else top[0], data)
                saved.add(sample)
            if choice != top[0]:
                prior = policy.actions_by_episode[policy.bank.episode_ids[choice]][policy.bank.local_frame_indices[choice]]
            actions.append(prior)
        return np.stack(actions)
    return act


def summarize_queries(path, phases, filtered, seeds, horizon):
    counts = {str(k): dict(queries=0, top1_mismatch=0, recovered_top5=0, changed=0) for k in range(3)}
    next_step = dict.fromkeys(seeds, 0)
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            seed, step, phase = (int(row[k]) for k in ("seed","step","phase"))
            top = [int(row[f"rank{i}"]) for i in range(1,6)]
            selected = int(row["selected"])
            matches = [r for r in top if phases[r] == phase]
            if step != next_step[seed] or selected != (matches[0] if filtered and matches else top[0]):
                raise ValueError("Incomplete or inconsistent phase trace")
            next_step[seed] += 1
            bucket = counts[str(phase)]
            mismatch = phases[top[0]] != phase
            bucket["queries"] += 1
            bucket["top1_mismatch"] += int(mismatch)
            bucket["recovered_top5"] += int(mismatch and bool(matches))
            bucket["changed"] += int(selected != top[0])
    if any(n != horizon for n in next_step.values()):
        raise ValueError("Phase trace step counts differ")
    return counts


def main(source, run_id, smoke):
    policy, base, split, image, cfg = load_base(source)
    root = LIFT / "artifacts/residual" / run_id
    root.mkdir(parents=True, exist_ok=True)
    label_path = root/"bank_phases.json"
    if not label_path.exists():
        simulator = PhaseSimulator(LIFT, root/"labels.log")
        try:
            labels = simulator.send("label_demonstrations", list(policy.actions_by_episode))
        finally:
            simulator.close()
        write_json(label_path, dict(labels, bank_sha256=base["bank_sha256"]))
    labels = read_json(label_path)
    if labels["bank_sha256"] != base["bank_sha256"] or set(labels["episodes"]) != set(policy.actions_by_episode):
        raise ValueError("Labels do not match the frozen bank")
    phases = np.asarray([labels["episodes"][key][step] for key,step in
                         zip(policy.bank.episode_ids, policy.bank.local_frame_indices)])
    bank_counts = np.bincount(phases, minlength=3).tolist()
    if not all(bank_counts):
        raise ValueError("A phase is absent from the entire training bank")
    print("Historical phase counts:", bank_counts, flush=True)
    settings = read_json(LIFT / split["evaluation_config"]["path"])
    seeds = settings["test_seeds"][:1] if smoke else settings["test_seeds"]
    horizon = 10 if smoke else settings["horizon"]
    common = dict(config=cfg,seeds=seeds,horizon=horizon,phase_definition=PHASES,
                  fixture_hash=split["evaluation_initial_state_contract"]["sha256"], top_k=5,
                  bank=base["bank_sha256"], encoder_sha256=base["checkpoint_sha256"],
                  labels_sha256=sha256(label_path), source_sha256=sha256(__file__),
                  simulator_sha256=sha256(HERE/"phase_simulator.py"), workers=min(len(seeds),DESIGN["workers"]))
    common["phase_definition"] = list(PHASES)
    summary = {"training_frames_by_phase": bank_counts, "historical_height_restore_max_abs_error_m": labels["max_cube_height_error_m"]}
    for name in ("original", "phase_filtered"):
        filtered = name == "phase_filtered"
        path = root/name
        protocol = dict(common, variant=name)
        if path.exists() and read_json(path/"protocol.json") != protocol:
            raise ValueError("Existing run protocol differs")
        if path.exists() and not (path/"metrics.json").exists():
            path.rename(path.with_name(f"{name}_incomplete_{time.time_ns()}"))
        if not path.exists():
            path.mkdir()
            write_json(path/"protocol.json",protocol)
            gallery = path/"gallery"
            gallery.mkdir()
            with (path/"queries.csv").open("x",newline="") as stream, h5py.File(image,"r") as data:
                writer = csv.writer(stream)
                writer.writerow(["seed","step","phase",*[f"rank{i}" for i in range(1,6)],"selected"])
                policy_fn = actuator(policy, phases, filtered, writer, gallery, data,
                                     set(seeds[:1] if smoke else (seeds[0],seeds[25],seeds[-1])))
                print(name, flush=True)
                run_rollouts(policy_fn,cfg,LIFT,seeds,horizon,common["workers"],path,simulator_factory=PhaseSimulator)
        check_trace(path,protocol)
        summary[name] = {"metrics":read_json(path/"metrics.json"),
                         "query_counts":summarize_queries(path/"queries.csv",phases,filtered,seeds,horizon)}
    summary["paired_filtered_minus_original"] = paired_counts(root/"original",root/"phase_filtered",common)
    if not smoke:
        result = HERE/"results"/run_id
        result.mkdir(parents=True,exist_ok=False)
        for name in ("original", "phase_filtered"):
            for filename in ("protocol.json", "rollouts.csv"):
                (result/f"{name}_{filename}").write_bytes((root/name/filename).read_bytes())
        manifests = [label_path] + [p for name in ("original","phase_filtered") for p in (root/name).rglob("*")
                                  if p.suffix == ".jpg" or p.name in ("queries.csv","actions.csv")]
        write_json(result/"artifact_manifest.json",[dict(path=p.relative_to(LIFT).as_posix(),bytes=p.stat().st_size,
                                                          sha256=sha256(p)) for p in manifests])
        write_json(result/"summary.json",summary)
    print(summary,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--alt100-source",required=True,type=Path)
    parser.add_argument("--smoke",action="store_true")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in (".",".."):
        parser.error("Use a single directory name for the run")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    main(args.alt100_source,args.run_id,args.smoke)
