"""Train and evaluate local residual corrections for Push-T retrieval."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/study.json"
CFG = json.loads(CONFIG.read_text(encoding="utf-8"))
REF = (ROOT / CFG["reference_project"]).resolve()
ORACLE = REF / "experiments/geometry_oracle_alt"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, str(ORACLE))
from common import load, now, save, sha
from geometry_bank import GeometryBank, components, wrap
from geometry_policy import GeometryPolicy
from pusht.repro import tree_fingerprint

for key in ("TORCH_HOME", "WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR",
            "MPLCONFIGDIR", "NUMBA_CACHE_DIR"):
    os.environ[key] = str(ROOT / "artifacts/cache" / key.lower())

H = CFG["horizon"]


def rotate(vectors, theta):
    c, s = np.cos(theta), np.sin(theta)
    x, y = np.moveaxis(vectors, -1, 0)
    return np.stack((c*x - s*y, s*x + c*y), axis=-1)


def local(pos, state):
    return rotate(pos - state[2:4], -state[4])


def context(query, matched, command):
    q = local(query[:2], query)
    p = local(matched[:2], matched)
    block = local(matched[2:4], query)
    angle = float(wrap(matched[4] - query[4]))
    prior = local(command, matched)
    features = np.r_[q/128, p/128, block/128, np.sin(angle), np.cos(angle), prior.ravel()/128]
    return features.astype(np.float32), prior.astype(np.float32)


def world(command, state):
    return np.clip(rotate(command, state[4]) + state[2:4], 0, 512).astype(np.float32)


class Retrieval:
    def __init__(self, bank, mode):
        self.bank, self.mode = bank, mode
        self.states = bank.states.astype(float)
        self.local = np.stack([local(s[:2], s) for s in self.states])

    def indices(self, query, exclude=None, count=1):
        terms = components(query, self.states)
        if self.mode == "global":
            scores = terms.sum(axis=1)
        else:
            rel = local(query[:2], query)
            scores = (terms[:, 1] * (512/40)**2
                      + terms[:, 2] * (180/15)**2
                      + np.sum((self.local-rel)**2, axis=1) / 30**2)
        if exclude is not None:
            scores[self.bank.episode_id == exclude] = np.inf
        if count == 1:
            return [int(np.argmin(scores))]
        return np.argsort(scores, kind="stable")[:count].tolist()

    def example(self, state, index):
        chunk, source = self.bank.action_chunk(index, H)
        features, prior = context(state, self.states[index], chunk)
        return features, prior, chunk, source


class Residual(nn.Module):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit
        self.net = nn.Sequential(nn.Linear(24, 64), nn.ReLU(), nn.Linear(64, 64),
                                 nn.ReLU(), nn.Linear(64, H*2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        output = self.net(features).reshape(-1, H, 2)
        return output * 128 if self.limit is None else self.limit * output.tanh()


def build_examples(retrieval, episodes, count, training):
    data = retrieval.bank.data
    features, priors, targets, raw, states, ids = [], [], [], [], [], []
    for ep in episodes:
        start, stop = data.starts[ep], data.ends[ep]
        for step, state in enumerate(data.root["data/state"][start:stop-H+1].astype(float)):
            goal, _ = data.action_chunk(ep, step, H)
            selected = retrieval.indices(state, exclude=ep if training else None, count=count)
            if training and any(retrieval.bank.episode_id[i] == ep for i in selected):
                raise ValueError("Same-episode training lookup leaked")
            examples = [retrieval.example(state, i) for i in selected]
            features.append([x[0] for x in examples])
            priors.append([x[1] for x in examples])
            raw.append([x[2] for x in examples])
            targets.append(local(goal, state))
            states.append(state)
            ids.append(ep)
    return tuple(np.asarray(x) for x in (features, priors, targets, raw, states, ids))


def metrics(actual, expected):
    diff = actual - expected
    squared = (diff**2).mean(axis=(1, 2))
    return {"rmse": float(np.sqrt(squared.mean())),
            "first_action_rmse": float(np.sqrt((diff[:, 0]**2).mean())),
            "p90_chunk_rmse": float(np.quantile(np.sqrt(squared), .9))}


def train(round_id, plan, bank, manifest, output):
    torch.manual_seed(CFG["train_seed"])
    retrieval = Retrieval(bank, plan["retrieval"])
    count = plan["train_candidates"]
    train_set = build_examples(retrieval, manifest["train_subsets"]["30"], count, True)
    val_set = build_examples(retrieval, manifest["heldout_validation_episode_ids"], 1, False)
    x, prior, target, *_ = train_set
    x = torch.from_numpy(x.reshape(-1, 24))
    y = torch.from_numpy((target[:, None] - prior).reshape(-1, H, 2).astype(np.float32))
    model = Residual(plan["limit"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["learning_rate"])
    loss_fn = nn.MSELoss() if plan["loss"] == "mse" else nn.SmoothL1Loss(beta=.1)
    generator = torch.Generator().manual_seed(CFG["train_seed"])
    curve = []
    for epoch in range(1, CFG["epochs"]+1):
        losses = []
        for selection in torch.randperm(len(x), generator=generator).split(CFG["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x[selection])/128, y[selection]/128)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        curve.append({"epoch": epoch, "loss": float(np.mean(losses))})
    checkpoint = output / "residual.pt"
    torch.save(model.state_dict(), checkpoint)
    vx, vp, target, raw, states, _ = val_set
    with torch.inference_mode():
        corrections = model(torch.from_numpy(vx[:, 0])).numpy()
    predictions = {"raw": raw[:, 0],
                   "aligned": np.stack([world(p[0], s) for p, s in zip(vp, states)]),
                   "residual": np.stack([world(p[0]+d, s) for p, d, s in zip(vp, corrections, states)])}
    expert = np.stack([world(p, s) for p, s in zip(target, states)])
    result = {name: metrics(prediction, expert) for name, prediction in predictions.items()}
    result.update({"train_queries": len(train_set[0]), "train_pairs": len(x),
                   "validation_queries": len(vx), "checkpoint_sha256": sha(checkpoint),
                   "round": round_id})
    save(output / "curve.json", curve)
    save(output / "validation.json", result)
    print(json.dumps(result), flush=True)


class CorrectedPolicy(GeometryPolicy):
    def __init__(self, retrieval, mode, model, sink):
        super().__init__(retrieval.bank, sink=sink)
        self.retrieval, self.mode, self.model = retrieval, mode, model

    @torch.inference_mode()
    def predict_action(self, obs):
        snapshot = self.state_provider(obs["agent_pos"][0, -1].cpu().numpy())
        state = snapshot["state"]
        index = self.retrieval.indices(state)[0]
        x, prior, raw, source = self.retrieval.example(state, index)
        correction = np.zeros_like(prior)
        if self.mode == "residual":
            correction = self.model(torch.from_numpy(x[None]))[0].numpy()
        command = raw if self.mode == "raw" else world(prior + correction, state)
        if command.shape != (H, 2) or not np.isfinite(command).all():
            raise ValueError("Invalid command")
        self.query_records.append({
            "evaluation_seed": snapshot["seed"], "current_step": snapshot["step"],
            "episode": int(self.bank.episode_id[index]), "frame": int(self.bank.frame_index[index]),
            "source_indices": source.tolist(), "state": state.tolist(),
            "matched_state": self.retrieval.states[index].tolist(),
            "commands": command.tolist(), "residual_l2": float(np.linalg.norm(correction)),
        })
        return {"action": torch.from_numpy(command[None])}


def evaluate(round_id, plan, bank, manifest, cfg, output):
    from oracle_runner import run_official

    trained = ROOT / "artifacts" / f"round{round_id}_train"
    training = json.loads((trained / "run.json").read_text(encoding="utf-8"))
    checkpoint = trained / "residual.pt"
    validation = json.loads((trained / "validation.json").read_text(encoding="utf-8"))
    if training["plan"] != plan or sha(checkpoint) != validation["checkpoint_sha256"]:
        raise ValueError("Training artifacts differ from the frozen evaluation plan")
    model = Residual(plan["limit"])
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    model.eval()
    seeds = manifest["test_env_seeds"]
    retrieval = Retrieval(bank, plan["retrieval"])
    scores = {}
    report = {}
    for mode in ("raw", "aligned", "residual"):
        folder = output / mode
        folder.mkdir()
        with (folder / "queries.jsonl").open("x", encoding="utf-8") as stream:
            policy = CorrectedPolicy(retrieval, mode, model, stream)
            proxy, elapsed = run_official(policy, cfg, seeds, folder)
        if [r["evaluation_seed"] for r in proxy.records] != seeds:
            raise ValueError("Incomplete seed set")
        scores[mode] = np.array([r["official_episode_score"] for r in proxy.records])
        with (output / f"{mode}.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(proxy.records[0]))
            writer.writeheader()
            writer.writerows(proxy.records)
        report[mode] = {"n": len(seeds), "mean": float(scores[mode].mean()),
                        "sd": float(scores[mode].std(ddof=1)),
                        "mean_final": float(np.mean([r["final_score"] for r in proxy.records])),
                        "seconds": elapsed, "queries": len(policy.query_records),
                        "query_sha256": sha(folder / "queries.jsonl")}
    rng = np.random.default_rng(CFG["bootstrap_seed"])
    sample = rng.integers(0, len(seeds), (CFG["bootstrap_draws"], len(seeds)))
    report["paired"] = {}
    for left, right in (("aligned", "raw"), ("residual", "aligned"), ("residual", "raw")):
        diff = scores[left] - scores[right]
        report["paired"][f"{left}_minus_{right}"] = {
            "delta": float(diff.mean()), "ci95": np.quantile(diff[sample].mean(axis=1), [.025, .975]).tolist(),
            "higher": int((diff > 1e-10).sum()), "lower": int((diff < -1e-10).sum())}
    report["checkpoint_sha256"] = sha(checkpoint)
    save(output / "summary.json", report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["train", "evaluate"])
    parser.add_argument("round", type=int, choices=[1, 2, 3])
    args = parser.parse_args()
    torch.set_num_threads(2)
    cfg, base, manifest = load()
    if tree_fingerprint(base["paths"]["dataset"])["sha256"] != manifest["dataset_fingerprint"]:
        raise ValueError("Dataset checksum mismatch")
    bank = GeometryBank(base["paths"]["dataset"], manifest["train_subsets"]["30"],
                        cfg["distance"]["weights"], cfg["distance"]["position_scale"])
    if len(bank.states) != 3806:
        raise ValueError("Unexpected training bank size")
    q = np.array([3., 4., 2., 4., np.pi/2])
    np.testing.assert_allclose(local(q[:2], q), [0, -1], atol=1e-12)
    np.testing.assert_allclose(world(local(q[:2], q), q), q[:2])
    path = ROOT / "configs" / f"round{args.round}.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    output = ROOT / "artifacts" / f"round{args.round}_{args.phase}"
    output.mkdir(parents=True, exist_ok=False)
    status = {"phase": args.phase, "round": args.round, "plan": plan, "config": CFG,
              "split_sha256": cfg["split_sha256"], "dataset_fingerprint": manifest["dataset_fingerprint"],
              "code_sha256": sha(__file__), "torch": torch.__version__, "started": now(),
              "train_ids": manifest["train_subsets"]["30"],
              "validation_ids": manifest["heldout_validation_episode_ids"],
              "test_seeds": manifest["test_env_seeds"], "status": "running"}
    save(output / "run.json", status)
    start = time.perf_counter()
    try:
        if args.phase == "train":
            train(args.round, plan, bank, manifest, output)
        else:
            evaluate(args.round, plan, bank, manifest, cfg, output)
        status.update(status="completed", seconds=time.perf_counter()-start, finished=now())
    except Exception:
        import traceback
        status.update(status="failed", traceback=traceback.format_exc(), finished=now())
        raise
    finally:
        save(output / "run.json", status)


if __name__ == "__main__":
    main()
