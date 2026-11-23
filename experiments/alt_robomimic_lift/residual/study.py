"""Fit a six-command residual on a frozen visual Lift retriever; evaluate in RoboSuite."""
import argparse
import csv
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import nn

HERE = Path(__file__).resolve().parent
LIFT = HERE.parent
sys.path.insert(0, str(LIFT.parent / "lift_box_dp/src"))
from common import add_references, contract, read_json, sha256, write_json
from evaluate import run_rollouts
from evaluate_epochs import check_trace
from matched_budget import fixtures

add_references(LIFT)
from alt_lift.data import LiftFrameStore, ProprioNormalizer, IMAGE_KEYS, PROPRIO_KEYS
from alt_lift.encoder import VisualProprioEncoder
from alt_lift.memory import MemoryBank, build_memory_bank
from alt_lift.policy import ActionRetrievalPolicy

DESIGN = read_json(HERE / "protocol.json")
BASE = DESIGN["base_run"]


def features(z, prop, bank, rows, prior):
    other = bank.proprio[rows]
    relative = Rotation.from_quat(prop[:, 3:7]) * Rotation.from_quat(other[:, 3:7]).inv()
    return np.column_stack((z - bank.embeddings[rows], prop[:, :3] - other[:, :3],
                            relative.as_rotvec(), prop[:, 7:] - other[:, 7:], prior)).astype("float32")


def load_base():
    cfg, alt, _, split, image = contract()
    assert alt == LIFT
    record = read_json(LIFT / "experiments/runs" / BASE / "artifact_manifest.json")
    source = LIFT / "artifacts/runs" / BASE
    bank_path, encoder_path = source / "memory_bank.npz", source / "checkpoint_final.pt"
    if (sha256(bank_path) != record["bank_sha256"] or sha256(encoder_path) != record["checkpoint_sha256"]
            or record["split_manifest_sha256"] != cfg["split_sha256"]):
        raise ValueError("Frozen Lift artifacts do not match")
    bank = MemoryBank.load(bank_path)
    train = split["train_subsets"]["30"]
    if set(bank.episode_ids) != set(train) or set(train) & set(split["heldout_validation_episode_ids"]):
        raise ValueError("Invalid demonstration split")
    with h5py.File(image, "r") as data:
        commands = {key: data["data"][key]["actions"][:] for key in train}
    state = torch.load(encoder_path, map_location="cpu", weights_only=True)
    encoder = VisualProprioEncoder(state["variant"], pretrained_path=None)
    encoder.load_state_dict(state["model_state_dict"])
    norm = ProprioNormalizer(state["proprio_mean"].numpy(), state["proprio_std"].numpy())
    policy = ActionRetrievalPolicy(encoder, bank, norm, commands, torch.device("cuda"))
    return policy, record, split, image, cfg


def pairs(query, policy, expert):
    bank = policy.bank
    recalled = []
    for start in range(0, len(query.embeddings), 128):
        part = slice(start, start + 128)
        scores = query.embeddings[part] @ bank.embeddings.T
        scores[query.episode_ids[part, None] == bank.episode_ids[None, :]] = -np.inf
        recalled.extend(scores.argmax(axis=1))
    rows = np.asarray(recalled)
    if np.any(query.episode_ids == bank.episode_ids[rows]):
        raise ValueError("Query episode leaked into retrieval")
    prior = np.stack([policy.actions_by_episode[key][i] for key, i in
                      zip(bank.episode_ids[rows], bank.local_frame_indices[rows])])
    targets = np.stack([expert[key][i] for key, i in zip(query.episode_ids, query.local_frame_indices)])
    x = features(query.embeddings, query.proprio, bank, rows, prior)
    return torch.from_numpy(x), torch.tensor(targets[:, :6] - prior[:, :6], dtype=torch.float32), rows


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(143, 64), nn.ReLU(), nn.Linear(64, 64),
                                    nn.ReLU(), nn.Linear(64, 6))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)
        self.register_buffer("mean", torch.zeros(143))
        self.register_buffer("scale", torch.ones(143))

    def forward(self, x):
        return self.layers((x - self.mean) / self.scale)


def correct(prior, delta):
    result = prior.copy()
    result[..., :6] = np.clip(prior[..., :6] + delta, -1, 1)
    return result


def train(run):
    policy, base, split, image, cfg = load_base()
    run.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    x, y, rows = pairs(policy.bank, policy, policy.actions_by_episode)
    held = LiftFrameStore(image, split["heldout_validation_episode_ids"])
    val_bank = build_memory_bank(policy.model, held, policy.normalizer, run / "heldout_queries.npz", policy.device)
    vx, vy, _ = pairs(val_bank, policy, {k: v.actions for k, v in held.episodes.items()})
    np.save(run / "training_neighbor_rows.npy", rows)
    torch.manual_seed(DESIGN["residual_seed"])
    rng = torch.Generator().manual_seed(DESIGN["shuffle_seed"])
    head = Head()
    head.mean.copy_(x.mean(0))
    head.scale.copy_(x.std(0, unbiased=False).clamp_min(0.001))
    opt = torch.optim.Adam(head.parameters(), lr=DESIGN["learning_rate"])
    # Test the actual inference path against original ALT on held-out RGB observations.
    with h5py.File(image, "r") as data:
        for key in held.episode_ids:
            obs = {k: data["data"][key]["obs"][k][0] for k in IMAGE_KEYS + PROPRIO_KEYS}
            original, _ = policy.act(obs)
            actual = online(policy, head)([obs])[0]
            np.testing.assert_array_equal(actual, original)
    logs, checkpoints = [], {}
    for epoch in range(1, max(DESIGN["epochs"]) + 1):
        for batch in torch.randperm(len(x), generator=rng).split(DESIGN["batch_size"]):
            loss = nn.functional.mse_loss(head(x[batch]), y[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            logs.append(dict(epoch=epoch, train_rmse=(head(x)-y).square().mean().sqrt().item(),
                             heldout_rmse=(head(vx)-vy).square().mean().sqrt().item()))
        if epoch in DESIGN["epochs"]:
            path = run / f"epoch{epoch}.pt"
            torch.save(head.state_dict(), path)
            checkpoints[str(epoch)] = sha256(path)
    result = dict(design=DESIGN, source_sha256=sha256(__file__), protocol_sha256=sha256(HERE/"protocol.json"),
                  base_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=LIFT,text=True).strip(),
                  base_checkpoint_sha256=base["checkpoint_sha256"], bank_sha256=base["bank_sha256"],
                  split_sha256=cfg["split_sha256"], dataset_fingerprint=base["dataset_fingerprint"],
                  train_ids=split["train_subsets"]["30"], holdout_ids=list(held.episode_ids),
                  train_frames=len(x), heldout_frames=len(vx), zero_identity_observations=len(held.episode_ids),
                  policy_python=platform.python_version(), torch=str(torch.__version__), gpu=torch.cuda.get_device_name(),
                  parameters=sum(p.numel() for p in head.parameters()), train_seconds=time.perf_counter()-started,
                  total_updates=epoch*((len(x)+DESIGN["batch_size"]-1)//DESIGN["batch_size"]),
                  heldout_baseline_rmse=vy.square().mean().sqrt().item(), history=logs, checkpoints=checkpoints,
                  training_neighbor_rows_sha256=sha256(run/"training_neighbor_rows.npy"))
    write_json(run/"training.json", result)
    print("Training complete", len(x), len(vx), logs[-1], flush=True)


def online(policy, head):
    @torch.inference_mode()
    def act(observations):
        priors, infos = zip(*(policy.act(obs, include_embedding=True) for obs in observations))
        priors = np.stack(priors)
        x = features(np.stack([r["query_embedding"] for r in infos]), np.stack([r["query_proprio"] for r in infos]),
                     policy.bank, [r["bank_row"] for r in infos], priors)
        return correct(priors, head(torch.from_numpy(x)).numpy())
    return act


def evaluate(run):
    policy, base, split, _, cfg = load_base()
    trained = read_json(run/"training.json")
    if trained["design"] != DESIGN or trained["bank_sha256"] != base["bank_sha256"]:
        raise ValueError("Incompatible training run")
    _, seeds, records, hashes = fixtures(split, LIFT)
    seeds = seeds[:DESIGN["validation_count"]]
    test_cfg = read_json(LIFT / split["evaluation_config"]["path"])

    def score(epoch, stage, starts, horizon, validation, fixture_hash):
        path = run/f"epoch{epoch}.pt"
        if sha256(path) != trained["checkpoints"][str(epoch)]:
            raise ValueError("Checkpoint checksum mismatch")
        out = run/f"{stage}_e{epoch}"
        protocol = dict(epoch=epoch, seeds=starts, horizon=horizon, config=cfg, fixture_hash=fixture_hash,
                        checkpoint=sha256(path), bank=base["bank_sha256"], execution_horizon=1)
        if out.exists():
            if read_json(out/"protocol.json") != protocol:
                raise ValueError("Existing rollout protocol differs")
            if not (out/"metrics.json").exists():
                out.rename(out.with_name(out.name+f"_incomplete_{time.time_ns()}"))
        if not out.exists():
            out.mkdir()
            write_json(out/"protocol.json",protocol)
            head = Head().eval()
            head.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
            run_rollouts(online(policy,head), cfg, LIFT, starts, horizon, min(len(starts),DESIGN["workers"]),
                         out, validation_fixtures=validation)
        check_trace(out,protocol)
        return read_json(out/"metrics.json")

    metrics = {str(e):score(e,"validation",seeds,DESIGN["validation_horizon"],records,hashes)
               for e in DESIGN["epochs"]}
    selected = max(DESIGN["epochs"],key=lambda e:tuple(metrics[str(e)][k]["count"] for k in DESIGN["selection"][:-1])+(-e,))
    write_json(run/"selection.json", dict(epoch=selected,metrics=metrics))
    print("Validation-selected epoch",selected,flush=True)
    score(selected,"test",test_cfg["test_seeds"],test_cfg["horizon"],None,split["evaluation_initial_state_contract"]["sha256"])


def summarize(run):
    tr = read_json(run/"training.json")
    sel = read_json(run/"selection.json")
    result = run/f"test_e{sel['epoch']}"
    protocol = read_json(result/"protocol.json")
    ref = LIFT.parent / "lift_box_dp/evidence/alt30_seed42_v1/eval01"
    base = read_json(ref/"protocol.json")
    if (base["test_seeds"] != protocol["seeds"] or base["horizon"] != protocol["horizon"]
            or base["initial_states_sha256"] != protocol["fixture_hash"] or base["config"] != protocol["config"]
            or base["hashes"]["checkpoint_final.pt"] != tr["base_checkpoint_sha256"]
            or base["hashes"]["memory_bank.npz"] != tr["bank_sha256"]):
        raise ValueError("Archived comparator differs")
    verified = read_json(LIFT.parent/"lift_box_dp/experiments/exp_001_lift30/comparison_dp30_seed42_v1_eval01.json")
    for filename in ("protocol.json", "rollouts.csv", "metrics.json"):
        if sha256(ref/filename) != verified["alt"]["file_sha256"][filename]:
            raise ValueError("Archived baseline changed")
    with (ref/"rollouts.csv").open() as f:
        before = list(csv.DictReader(f))
    after = check_trace(result,protocol)
    if [r["seed"] for r in before] != [r["seed"] for r in after]:
        raise ValueError("Seeds are unpaired")
    rng = np.random.default_rng(42)
    draws = rng.integers(len(after),size=(10000,len(after)))
    paired = {}
    for k in ("hold_success","ever_success","final_success"):
        a = np.array([int(r[k]=="True") for r in before])
        b = np.array([int(r[k]=="True") for r in after])
        d = b-a
        paired[k] = dict(alt=int(a.sum()), residual=int(b.sum()), gained=int((d>0).sum()), lost=int((d<0).sum()),
                         difference=float(d.mean()), paired_bootstrap95=np.quantile(d[draws].mean(1),[.025,.975]).tolist())
    out = HERE/"results"/run.name
    out.mkdir(parents=True,exist_ok=False)
    for path in (run/"selection.json",run/"training.json",result/"rollouts.csv",result/"metrics.json",result/"protocol.json"):
        shutil.copyfile(path,out/path.name)
    write_json(out/"paired.json",paired)
    print(paired)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=["train","evaluate","summarize"])
    parser.add_argument("run_id")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in (".",".."):
        parser.error("Use a single directory name for the run")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    {"train":train,"evaluate":evaluate,"summarize":summarize}[args.phase](LIFT/"artifacts/residual"/args.run_id)
