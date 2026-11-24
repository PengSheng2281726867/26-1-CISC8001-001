"""Learn cube-relative position from the frozen Lift demonstration split and test visual geometry."""
import argparse
from functools import partial
from pathlib import Path
import shutil
import time

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from waypoint_study import DESIGN as GEOMETRY, geometry_rows, policy_for
from study import HERE, LIFT, check_trace, load_base, paired_counts, read_json, run_rollouts, sha256, write_json
from alt_lift.data import IMAGE_KEYS, PROPRIO_KEYS, canonicalize_proprio
from evaluate import Simulator

DESIGN = dict(epochs=40, checkpoint_interval=5, batch_size=96, learning_rate=0.001,
              model_seed=42, shuffle_seed=43, selection="minimum held-out 3D vector RMSE",
              target="cube center minus EEF xyz, world-axis meters", stage_source="simulation contact and height",
              query_position="vision and proprioception", historical_position="vision and proprioception")


class Locator(nn.Module):
    """Shared spatial encoder, per-camera soft keypoints, and a joint XYZ regressor."""

    def __init__(self):
        super().__init__()
        self.image = nn.Sequential(nn.Conv2d(3,16,5,stride=2,padding=2),nn.GroupNorm(4,16),nn.ReLU(),
                                   nn.Conv2d(16,32,3,stride=2,padding=1),nn.GroupNorm(4,32),nn.ReLU(),
                                   nn.Conv2d(32,32,3,padding=1),nn.GroupNorm(4,32),nn.ReLU())
        self.maps = nn.Conv2d(32,8,1)
        self.head = nn.Sequential(nn.Linear(2*(8*2+32)+9,64),nn.ReLU(),nn.Linear(64,32),
                                  nn.ReLU(),nn.Linear(32,3))
        grid = torch.linspace(-1,1,21)
        yy, xx = torch.meshgrid(grid,grid,indexing="ij")
        self.register_buffer("grid",torch.stack((xx.flatten(),yy.flatten()),dim=-1))
        self.register_buffer("proprio_mean",torch.zeros(9))
        self.register_buffer("proprio_scale",torch.ones(9))

    def forward(self, images, proprio):
        count = images.shape[0]
        images = images.reshape(-1,84,84,3).permute(0,3,1,2).float().div(127.5).sub(1)
        features = self.image(images)
        keypoints = self.maps(features).flatten(2).softmax(-1) @ self.grid
        spatial = torch.cat((keypoints.flatten(1),features.mean(dim=(2,3))),dim=1).reshape(count,-1)
        prop = (proprio-self.proprio_mean)/self.proprio_scale
        return self.head(torch.cat((spatial,prop),dim=1))/10


def dataset(image, episodes):
    images, prop, labels, ids, frames = [],[],[],[],[]
    with h5py.File(image,"r") as source:
        for key in episodes:
            obs = source["data"][key]["obs"]
            proprio = canonicalize_proprio(np.column_stack([obs[k][:] for k in PROPRIO_KEYS]))
            images.append(np.stack([obs[k][:] for k in IMAGE_KEYS],axis=1))
            prop.append(proprio)
            labels.append((obs["object"][:,:3]-proprio[:,:3]).astype(np.float32))
            ids.extend([key]*len(proprio))
            frames.extend(range(len(proprio)))
    arrays = [torch.from_numpy(np.concatenate(values)) for values in (images,prop,labels)]
    return TensorDataset(*arrays), dict(zip(zip(ids,frames),range(len(ids))))


@torch.inference_mode()
def predictions(model, examples, device):
    return torch.cat([model(rgb.to(device),p.to(device)).cpu() for rgb,p,_ in
                      DataLoader(examples,batch_size=DESIGN["batch_size"],shuffle=False)])


def errors(pred, target):
    difference = (pred-target)*1000
    distance = torch.linalg.vector_norm(difference,dim=1)
    return dict(frames=len(pred), per_axis_rmse_mm=difference.square().mean(0).sqrt().tolist(),
                vector_rmse_mm=distance.square().mean().sqrt().item(),
                median_euclidean_mm=distance.median().item(),p90_euclidean_mm=distance.quantile(.9).item())


def fit(train, holdout, model, device, output):
    p = train.tensors[1]
    model.proprio_mean.copy_(p.mean(0))
    model.proprio_scale.copy_(p.std(0,unbiased=False).clamp_min(1e-4))
    optimizer = torch.optim.Adam(model.parameters(),lr=DESIGN["learning_rate"])
    rng = torch.Generator().manual_seed(DESIGN["shuffle_seed"])
    batches = DataLoader(train,batch_size=DESIGN["batch_size"],shuffle=True,generator=rng)
    history = []
    best = float("inf")
    for epoch in range(1,DESIGN["epochs"]+1):
        model.train()
        for rgb,proprio,target in batches:
            command = model(rgb.to(device),proprio.to(device))
            loss = ((command-target.to(device))*10).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % DESIGN["checkpoint_interval"] == 0:
            model.eval()
            metrics = errors(predictions(model,holdout,device),holdout.tensors[2])
            history.append(dict(epoch=epoch,**metrics))
            print("Held-out localization",history[-1],flush=True)
            if metrics["vector_rmse_mm"] < best:
                best = metrics["vector_rmse_mm"]
                selected = epoch
                torch.save(model.state_dict(),output/"locator.pt")
    mean = train.tensors[2].mean(0).expand_as(holdout.tensors[2])
    return dict(selected_epoch=selected,history=history,constant_baseline=errors(mean,holdout.tensors[2]),
                parameters=sum(p.numel() for p in model.parameters()),train_frames=len(train),
                checkpoint_sha256=sha256(output/"locator.pt"))


def main(source, run_id, smoke):
    policy, base, split, image, cfg = load_base(source)
    root = LIFT/"artifacts/residual"
    bank = policy.bank
    stage_file = root/"lift100_phase_v1/bank_phases.json"
    stages = read_json(stage_file)
    if stages["bank_sha256"] != base["bank_sha256"]:
        raise ValueError("Unexpected stage label bank")
    phase = np.asarray([stages["episodes"][key][i] for key,i in zip(bank.episode_ids,bank.local_frame_indices)])
    output = root/run_id
    output.mkdir(exist_ok=True)
    _, waypoint, available, actions = geometry_rows(policy,image)
    ids, held_ids = list(policy.actions_by_episode), split["heldout_validation_episode_ids"]
    if set(ids)&set(held_ids):
        raise ValueError("Demonstration split overlaps")
    resolved = dict(design=DESIGN,geometry=GEOMETRY,bank=base["bank_sha256"],encoder=base["checkpoint_sha256"],
                    config=cfg,phase_sha256=sha256(stage_file),dataset_fingerprint=base["dataset_fingerprint"],
                    train_ids=ids,holdout_ids=held_ids,source_sha256=sha256(__file__),
                    geometry_source_sha256=sha256(HERE/"waypoint_study.py"))
    if (output/"frozen.json").exists() and read_json(output/"frozen.json")!=resolved:
        raise ValueError("Existing run differs")
    if not (output/"frozen.json").exists():
        write_json(output/"frozen.json",resolved)
    train, order = dataset(image,ids)
    torch.manual_seed(DESIGN["model_seed"])
    model = Locator().to(policy.device)
    record = output/"training.json"
    if not record.exists():
        held, _ = dataset(image,held_ids)
        write_json(record,fit(train,held,model,policy.device,output))
    training = read_json(record)
    checkpoint = output/"locator.pt"
    if sha256(checkpoint) != training["checkpoint_sha256"]:
        raise ValueError("Changed localization checkpoint")
    model.load_state_dict(torch.load(checkpoint,weights_only=True))
    model.eval()
    train_pred = predictions(model,train,policy.device).numpy()
    predicted = np.stack([train_pred[order[key,frame]] for key,frame in zip(bank.episode_ids,bank.local_frame_indices)])
    np.save(output/"visual_bank_offsets.npy",predicted)

    def locate(obs, proprio):
        images = torch.from_numpy(np.stack([obs[k] for k in IMAGE_KEYS])[None]).to(policy.device)
        p = torch.from_numpy(proprio[None]).to(policy.device)
        return model(images,p)[0].cpu().numpy()

    settings = read_json(LIFT/split["evaluation_config"]["path"])
    seeds = settings["test_seeds"][:1] if smoke else settings["test_seeds"]
    horizon = 140 if smoke else settings["horizon"]
    protocol = dict(resolved,seeds=seeds,horizon=horizon,fixture_hash=split["evaluation_initial_state_contract"]["sha256"],
                    checkpoint=sha256(checkpoint),predicted_bank=sha256(output/"visual_bank_offsets.npy"))
    path = output/("smoke" if smoke else "formal")
    if path.exists() and read_json(path/"protocol.json")!=protocol:
        raise ValueError("Existing evaluation differs")
    if path.exists() and not (path/"interventions.json").exists():
        path.rename(path.with_name(f"{path.name}_incomplete_{time.time_ns()}"))
    if not path.exists():
        path.mkdir()
        write_json(path/"protocol.json",protocol)
        counters = dict(eligible_steps=0,modified_steps=0)
        act = policy_for(policy,phase,predicted,waypoint,available,actions,None,"geometry",counters,estimate_offset=locate)
        simulator = partial(Simulator,entrypoint=HERE/"phase_simulator.py")
        run_rollouts(act,cfg,LIFT,seeds,horizon,min(len(seeds),GEOMETRY["workers"]),path,simulator_factory=simulator)
        write_json(path/"interventions.json",counters)
    rows = check_trace(path,protocol)
    if any(int(row["steps"])!=horizon for row in rows):
        raise ValueError("Incomplete formal rollout")
    if not smoke:
        controls = root/"lift100_waypoint_v1"
        archive = HERE/"results/lift100_waypoint_v1"
        comparisons = {}
        for name in ("phase_replay","geometry"):
            ref = controls/name
            previous = read_json(ref/"protocol.json")
            keys = ("seeds","horizon","fixture_hash","bank","encoder","config")
            if (any(protocol[key]!=previous[key] for key in keys)
                    or sha256(ref/"rollouts.csv")!=sha256(archive/f"{name}_rollouts.csv")):
                raise ValueError("Archived geometry control differs")
            comparisons[name] = paired_counts(ref,path,protocol)
        result = HERE/"results"/run_id
        result.mkdir(parents=True,exist_ok=False)
        for name in ("protocol.json","rollouts.csv","metrics.json","interventions.json"):
            shutil.copyfile(path/name,result/name)
        shutil.copyfile(record,result/"training.json")
        write_json(result/"paired.json",comparisons)
        artifacts = [checkpoint,output/"visual_bank_offsets.npy",path/"actions.csv"]
        write_json(result/"artifact_manifest.json",[dict(path=p.relative_to(LIFT).as_posix(),bytes=p.stat().st_size,
                                                         sha256=sha256(p)) for p in artifacts])
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
