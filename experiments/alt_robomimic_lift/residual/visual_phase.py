"""Replace the live simulator phase with a classifier while keeping visual localization fixed."""
import argparse
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from visual_geometry import Locator, dataset
from waypoint_study import DESIGN as GEOMETRY, geometry_rows, policy_for
from study import HERE, LIFT, check_trace, load_base, paired_counts, read_json, run_rollouts, sha256, write_json
from phase_study import PhaseSimulator
from alt_lift.data import IMAGE_KEYS, PROPRIO_KEYS

DESIGN = dict(epochs=35, checkpoint_interval=5, batch_size=96, learning_rate=0.0005,
              model_seed=42, shuffle_seed=43, class_weight="inverse sqrt of train frequency",
              checkpoint_selection="best held-out macro F1, then earliest epoch",
              current_phase="learned from camera images and proprioception",
              memory_phase="fixed labels on the 100 training demonstrations")


def classified_dataset(image, episode_ids, labels):
    examples, order = dataset(image, episode_ids)
    phases = torch.tensor([labels[key][step] for key,step in order],dtype=torch.long)
    return TensorDataset(*examples.tensors[:2],phases)


@torch.inference_mode()
def confusion(model, examples, device):
    result = torch.zeros((3,3),dtype=torch.long)
    for rgb, prop, phase in DataLoader(examples,batch_size=DESIGN["batch_size"]):
        predicted = model(rgb.to(device),prop.to(device)).argmax(1).cpu()
        result += torch.bincount(phase*3+predicted,minlength=9).reshape(3,3)
    diag = result.diag()
    denominator = result.sum(1)+result.sum(0)
    return dict(confusion_true_rows=result.tolist(),accuracy=float(diag.sum()/result.sum()),
                macro_f1=float((2*diag/denominator.clamp_min(1)).mean()),
                recall=(diag/result.sum(1).clamp_min(1)).tolist())


def fit(model, train, held, device, checkpoint):
    torch.manual_seed(DESIGN["model_seed"])
    model.head[-1].reset_parameters()
    rng = torch.Generator().manual_seed(DESIGN["shuffle_seed"])
    batches = DataLoader(train,batch_size=DESIGN["batch_size"],shuffle=True,generator=rng)
    counts = train.tensors[2].bincount(minlength=3)
    if not torch.all(counts > 0):
        raise ValueError("The training set must contain all three stages")
    weight = counts.to(device).float().rsqrt()
    criterion = nn.CrossEntropyLoss(weight=weight/weight.mean())
    optimizer = torch.optim.Adam(model.parameters(),lr=DESIGN["learning_rate"])
    history, best = [], -1.
    for epoch in range(1,DESIGN["epochs"]+1):
        model.train()
        for rgb, prop, label in batches:
            logits = model(rgb.to(device),prop.to(device))*10
            loss = criterion(logits,label.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % DESIGN["checkpoint_interval"] == 0:
            model.eval()
            row = dict(epoch=epoch,**confusion(model,held,device))
            print("Held-out phase",row,flush=True)
            history.append(row)
            if row["macro_f1"]>best:
                selected, best = epoch, row["macro_f1"]
                torch.save(model.state_dict(),checkpoint)
    return dict(selected_epoch=selected,history=history,train_counts=counts.tolist(),
                checkpoint_sha256=sha256(checkpoint))


def main(source, run_id, smoke):
    policy, base, split, image, cfg = load_base(source)
    root = LIFT/"artifacts/residual"
    output = root/run_id
    output.mkdir(exist_ok=True)
    reference = root/"lift100_visual_geometry_v1"
    archived = HERE/"results/lift100_visual_geometry_v1"
    control = read_json(archived/"protocol.json")
    locator_file = reference/"locator.pt"
    offsets_file = reference/"visual_bank_offsets.npy"
    label_file = root/"lift100_phase_v1/bank_phases.json"
    if (sha256(locator_file)!=control["checkpoint"] or sha256(offsets_file)!=control["predicted_bank"]
            or sha256(label_file)!=control["phase_sha256"] or base["bank_sha256"]!=control["bank"]
            or base["checkpoint_sha256"]!=control["encoder"]):
        raise ValueError("Existing position-only control does not match the experiment")
    stage_labels = read_json(label_file)["episodes"]
    bank = policy.bank
    phase = np.array([stage_labels[k][i] for k,i in zip(bank.episode_ids,bank.local_frame_indices)])
    _, waypoint, available, actions = geometry_rows(policy,image)
    offsets = np.load(offsets_file,allow_pickle=False)
    training_ids = list(policy.actions_by_episode)
    held_ids = split["heldout_validation_episode_ids"]
    if set(training_ids)&set(held_ids):
        raise ValueError("Episode overlap")
    held_file = output/"holdout_stages.json"
    if not held_file.exists():
        sim = PhaseSimulator(LIFT,output/"labeling.log")
        try:
            labels = sim.send("label_demonstrations",held_ids)
        finally:
            sim.close()
        write_json(held_file,labels)
    resolved = dict(design=DESIGN, geometry=GEOMETRY, bank=control["bank"], encoder=control["encoder"],
                    locator=control["checkpoint"], predicted_bank=control["predicted_bank"],
                    train_labels=sha256(label_file),holdout_labels=sha256(held_file),
                    train_episodes=training_ids,holdout_episodes=held_ids,
                    source_sha256=sha256(__file__),geometry_sha256=sha256(HERE/"waypoint_study.py"))
    frozen = output/"frozen.json"
    if frozen.exists() and read_json(frozen)!=resolved:
        raise ValueError("Existing training protocol differs")
    if not frozen.exists():
        write_json(frozen,resolved)
    state = torch.load(locator_file,map_location="cpu",weights_only=True)
    locator = Locator().to(policy.device).eval()
    locator.load_state_dict(state)
    classifier = Locator().to(policy.device)
    classifier.load_state_dict(state)
    checkpoint = output/"classifier.pt"
    training_file = output/"training.json"
    if not training_file.exists():
        examples = classified_dataset(image,training_ids,stage_labels)
        held = classified_dataset(image,held_ids,read_json(held_file)["episodes"])
        write_json(training_file,fit(classifier,examples,held,policy.device,checkpoint))
    training = read_json(training_file)
    if sha256(checkpoint)!=training["checkpoint_sha256"]:
        raise ValueError("Classifier checkpoint changed")
    classifier.load_state_dict(torch.load(checkpoint,weights_only=True))
    classifier.eval()
    stage_counts = [0,0,0]

    def parse(obs, proprio):
        if set(obs)!=set(IMAGE_KEYS+PROPRIO_KEYS):
            raise ValueError(f"Unexpected online policy fields: {set(obs)}")
        rgb = torch.from_numpy(np.stack([obs[key] for key in IMAGE_KEYS])[None]).to(policy.device)
        prop = torch.from_numpy(proprio[None]).to(policy.device)
        return rgb, prop

    def locate(obs, proprio):
        return locator(*parse(obs,proprio))[0].cpu().numpy()

    def stage(obs, proprio):
        idx = classifier(*parse(obs,proprio)).argmax(1).item()
        stage_counts[idx] += 1
        return idx

    settings = read_json(LIFT/split["evaluation_config"]["path"])
    seeds = settings["test_seeds"][:1] if smoke else settings["test_seeds"]
    horizon = 140 if smoke else settings["horizon"]
    protocol = dict(resolved,seeds=seeds,horizon=horizon,config=cfg,
                    fixture_hash=split["evaluation_initial_state_contract"]["sha256"],
                    classifier=training["checkpoint_sha256"], online_keys=list(IMAGE_KEYS+PROPRIO_KEYS))
    path = output/("smoke" if smoke else "formal")
    if path.exists() and read_json(path/"protocol.json")!=protocol:
        raise ValueError("Existing evaluation differs")
    if path.exists() and not (path/"interventions.json").exists():
        path.rename(path.with_name(f"{path.name}_incomplete_{time.time_ns()}"))
    if not path.exists():
        path.mkdir()
        write_json(path/"protocol.json",protocol)
        counters = dict(eligible_steps=0,modified_steps=0)
        act = policy_for(policy,phase,offsets,waypoint,available,actions,None,"geometry",counters,
                         estimate_offset=locate,estimate_phase=stage)
        run_rollouts(act,cfg,LIFT,seeds,horizon,min(len(seeds),GEOMETRY["workers"]),path)
        write_json(path/"interventions.json",dict(counters, predicted_phase_counts=stage_counts))
    if any(int(row["steps"])!=horizon for row in check_trace(path,protocol)):
        raise ValueError("Incomplete rollout")
    if not smoke:
        previous = reference/"formal"
        comparisons = {}
        for name, ref, old in (("oracle_phase_visual_position",previous,archived/"rollouts.csv"),
                               ("oracle_phase_direct_replay",root/"lift100_waypoint_v1/phase_replay",
                                HERE/"results/lift100_waypoint_v1/phase_replay_rollouts.csv")):
            original = read_json(ref/"protocol.json")
            keys = ("seeds","horizon","fixture_hash","bank","encoder","config")
            if any(original[k]!=protocol[k] for k in keys) or sha256(ref/"rollouts.csv")!=sha256(old):
                raise ValueError("Archived comparison differs")
            comparisons[name] = paired_counts(ref,path,protocol)
        result = HERE/"results"/run_id
        result.mkdir(parents=True,exist_ok=False)
        for filename in ("protocol.json","rollouts.csv","metrics.json","interventions.json"):
            shutil.copyfile(path/filename,result/filename)
        shutil.copyfile(training_file,result/"training.json")
        write_json(result/"paired.json",comparisons)
        write_json(result/"artifact_manifest.json",
                   [dict(path=p.relative_to(LIFT).as_posix(),bytes=p.stat().st_size,sha256=sha256(p))
                    for p in (checkpoint,held_file,path/"actions.csv")])
        print(comparisons,flush=True)


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
