"""Test object-relative geometry and a supervised waypoint residual on frozen Lift-100."""
import argparse
from functools import partial
from pathlib import Path
import shutil
import time

import h5py
import numpy as np
import torch
from torch import nn

from study import HERE, LIFT, check_trace, load_base, paired_counts, read_json, run_rollouts, sha256, write_json
from evaluate import Simulator

DESIGN = dict(waypoint_steps=3, action_meters=0.05, component_cap=0.15, epochs=20,
              learning_rate=0.001, batch_size=64, model_seed=42, shuffle_seed=43, workers=4,
              head_widths=[9,32,32,3])
GROUPS = ("phase_replay", "geometry", "geometry_mlp")


def geometry_rows(policy, image):
    horizon = DESIGN["waypoint_steps"]
    bank = policy.bank
    positions, goals, available, actions = [], [], [], []
    with h5py.File(image, "r") as source:
        for episode, commands in policy.actions_by_episode.items():
            obs = source["data"][episode]["obs"]
            ee = obs["robot0_eef_pos"][:]
            cube = obs["object"][:, :3]
            rows = np.flatnonzero(bank.episode_ids == episode)
            idx = bank.local_frame_indices[rows]
            np.testing.assert_allclose(bank.proprio[rows,:3], ee[idx], atol=1e-6, rtol=0)
            positions.extend(zip(rows, cube[idx]-ee[idx]))
            goals.extend(zip(rows, ee[np.minimum(idx+horizon,len(ee)-1)]-cube[idx]))
            available.extend(zip(rows, idx+horizon < len(ee)))
            actions.extend(zip(rows, commands[idx]))
    return tuple(np.asarray([value for _,value in sorted(table,key=lambda item:item[0])])
                 for table in (positions,goals,available,actions))


def head_model():
    return nn.Sequential(nn.Linear(9,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU(),nn.Linear(32,3))


def fit(policy, phase, offset, waypoint, available, action, output):
    bank = policy.bank
    candidates = np.flatnonzero(phase == 0)
    eligible_query = candidates[available[candidates] & (action[candidates,6] < 0)]
    refs = []
    for indices in np.array_split(eligible_query, max(1,len(eligible_query)//128)):
        scores = bank.embeddings[indices] @ bank.embeddings[candidates].T
        scores[bank.episode_ids[indices,None] == bank.episode_ids[candidates][None,:]] = -np.inf
        refs.extend(candidates[scores.argmax(1)])
    refs = np.asarray(refs)
    assert np.all(bank.episode_ids[eligible_query] != bank.episode_ids[refs])
    keep = available[refs] & (action[refs,6] < 0)
    query, refs = eligible_query[keep], refs[keep]
    if not len(query):
        raise ValueError("No cross-episode approach pairs")
    x = torch.tensor(np.column_stack((offset[query],offset[refs],waypoint[refs])),dtype=torch.float32)
    y = torch.tensor(waypoint[query]-waypoint[refs],dtype=torch.float32)
    torch.manual_seed(DESIGN["model_seed"])
    rng = torch.Generator().manual_seed(DESIGN["shuffle_seed"])
    head = head_model()
    opt = torch.optim.Adam(head.parameters(),lr=DESIGN["learning_rate"])
    # Train in decimeters to keep the task-space loss numerically well scaled.
    x, y = x*10, y*10
    for _ in range(DESIGN["epochs"]):
        for minibatch in torch.randperm(len(x),generator=rng).split(DESIGN["batch_size"]):
            loss = (head(x[minibatch])-y[minibatch]).square().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        rmse = (head(x)-y).square().mean().sqrt().item()/10
    torch.save(head.state_dict(),output/"head.pt")
    np.save(output/"training_pairs.npy",np.column_stack((query,refs)))
    return dict(pairs=len(query),target_zero_rmse_m=y.square().mean().sqrt().item()/10,
                train_rmse_m=rmse,checkpoint_sha256=sha256(output/"head.pt"),
                pairs_sha256=sha256(output/"training_pairs.npy"))


def translated_command(prior, delta_m):
    command = prior.copy()
    residual = delta_m/(DESIGN["waypoint_steps"]*DESIGN["action_meters"])
    cap = DESIGN["component_cap"]
    command[:3] = np.clip(prior[:3]+np.clip(residual,-cap,cap),-1,1)
    return command


def policy_for(policy, phase, offset, waypoint, available, actions, head, variant, counters, estimate_offset=None):
    indices = [np.flatnonzero(phase == k) for k in range(3)]
    embeddings = [policy.search.embeddings[torch.as_tensor(rows,device=policy.device)] for rows in indices]

    @torch.inference_mode()
    def act(observations):
        commands = []
        for obs in observations:
            _, info = policy.act(obs, include_embedding=True)
            k = obs["oracle_phase"]
            q = torch.as_tensor(info["query_embedding"], device=policy.device)
            row = indices[k][int(torch.argmax(torch.mv(embeddings[k],q)))]
            prior = actions[row]
            eligible = k == 0 and available[row] and prior[6] < 0
            counters["eligible_steps"] += int(eligible)
            if variant != "phase_replay" and eligible:
                if estimate_offset is None:
                    now = obs["oracle_cube_center"]-info["query_proprio"][:3]
                else:
                    now = estimate_offset(obs, info["query_proprio"])
                delta = now-offset[row]
                if variant == "geometry_mlp":
                    x = torch.tensor(np.concatenate((now,offset[row],waypoint[row]))*10,dtype=torch.float32)
                    delta = delta+head(x).numpy()/10
                command = translated_command(prior,delta)
                counters["modified_steps"] += int(np.any(command[:3]!=prior[:3]))
                commands.append(command)
            else:
                commands.append(prior)
        return np.stack(commands)
    return act


def main(source, run_id, smoke):
    policy, base, split, image, cfg = load_base(source)
    local = LIFT/"artifacts/residual"
    labels_path = local/"lift100_phase_v1/bank_phases.json"
    labels = read_json(labels_path)
    if labels["bank_sha256"] != base["bank_sha256"]:
        raise ValueError("Stage label bank differs from ALT-100")
    bank = policy.bank
    phase = np.array([labels["episodes"][key][i] for key,i in zip(bank.episode_ids,bank.local_frame_indices)])
    offset, waypoint, available, actions = geometry_rows(policy,image)
    output = local/run_id
    output.mkdir(exist_ok=True)
    frozen = dict(design=DESIGN, bank=base["bank_sha256"], encoder=base["checkpoint_sha256"],
                  labels_sha256=sha256(labels_path), config=cfg, source_sha256=sha256(__file__),
                  simulator_sha256=sha256(HERE/"waypoint_simulator.py"), smoke=smoke)
    config = output/"frozen.json"
    if config.exists() and read_json(config) != frozen:
        raise ValueError("Existing run differs from the requested protocol")
    if not config.exists():
        write_json(config,frozen)
    training_path = output/"training.json"
    if not training_path.exists():
        write_json(training_path,fit(policy,phase,offset,waypoint,available,actions,output))
    training = read_json(training_path)
    print("Waypoint training:",training,flush=True)
    if sha256(output/"head.pt") != training["checkpoint_sha256"]:
        raise ValueError("Residual checkpoint changed")
    head = head_model().eval()
    head.load_state_dict(torch.load(output/"head.pt",weights_only=True))
    settings = read_json(LIFT/split["evaluation_config"]["path"])
    seeds = settings["test_seeds"][:1] if smoke else settings["test_seeds"]
    horizon = 140 if smoke else settings["horizon"]
    fixture_hash = split["evaluation_initial_state_contract"]["sha256"]
    results = {}
    simulator = partial(Simulator,entrypoint=HERE/"waypoint_simulator.py")
    for variant in GROUPS:
        path = output/variant
        protocol = dict(frozen,seeds=seeds,horizon=horizon,fixture_hash=fixture_hash,variant=variant,
                        checkpoint=training["checkpoint_sha256"])
        if path.exists() and read_json(path/"protocol.json") != protocol:
            raise ValueError("Existing rollout protocol differs")
        if path.exists() and not (path/"interventions.json").exists():
            path.rename(path.with_name(f"{variant}_incomplete_{time.time_ns()}"))
        if not path.exists():
            path.mkdir()
            write_json(path/"protocol.json",protocol)
            counters = dict(eligible_steps=0,modified_steps=0)
            act = policy_for(policy,phase,offset,waypoint,available,actions,head,variant,counters)
            print(variant,flush=True)
            run_rollouts(act,cfg,LIFT,seeds,horizon,min(len(seeds),DESIGN["workers"]),path,simulator_factory=simulator)
            write_json(path/"interventions.json",counters)
        rows = check_trace(path,protocol)
        if any(int(row["steps"])!=horizon for row in rows):
            raise ValueError("Unexpected rollout horizon")
        results[variant] = dict(metrics=read_json(path/"metrics.json"),interventions=read_json(path/"interventions.json"))
    results["geometry_vs_phase"] = paired_counts(output/"phase_replay",output/"geometry",protocol)
    results["mlp_vs_geometry"] = paired_counts(output/"geometry",output/"geometry_mlp",protocol)
    if not smoke:
        dest = HERE/"results"/run_id
        dest.mkdir(parents=True,exist_ok=False)
        for variant in GROUPS:
            for name in ("protocol.json","rollouts.csv"):
                shutil.copyfile(output/variant/name,dest/f"{variant}_{name}")
        for name in ("training.json","frozen.json"):
            shutil.copyfile(output/name,dest/name)
        artifacts = [output/"head.pt",output/"training_pairs.npy",*[output/v/"actions.csv" for v in GROUPS]]
        write_json(dest/"artifact_manifest.json",[dict(path=p.relative_to(LIFT).as_posix(),bytes=p.stat().st_size,
                                                       sha256=sha256(p)) for p in artifacts])
        write_json(dest/"summary.json",results)
    print(results,flush=True)


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
