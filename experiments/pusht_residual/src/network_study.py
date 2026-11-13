"""Network-only Push-T residual ablations with fixed retrieval."""
import argparse
import csv
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from study import (CFG, H, ROOT, GeometryBank, Retrieval, Residual, CorrectedPolicy,
                   build_examples, load, metrics, save, sha, tree_fingerprint, world)

EXP = ROOT / "experiments/network_study"
OUT = ROOT / "artifacts/network_study"
VALIDATION = list(range(91000, 91016))
MODES = ("full", "taper")


class TaperedResidual(Residual):
    def forward(self, features):
        correction = super().forward(features)
        return correction * torch.linspace(1, 0, H, device=features.device)[None, :, None]


def model_for(plan):
    return (TaperedResidual if plan["architecture"] == "taper" else Residual)(plan["limit"])


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fit(plan, bank, manifest, output):
    torch.manual_seed(CFG["train_seed"])
    retrieval = Retrieval(bank, "relative")
    episodes = manifest["train_subsets"][str(plan["query_count"])]
    heldout = manifest["heldout_validation_episode_ids"]
    assert not set(episodes) & set(heldout)
    assert not set(bank.episodes.tolist()) & set(heldout)
    features, priors, target, *_ = build_examples(retrieval, episodes, 1, True)
    vx, vp, vt, _, states, _ = build_examples(retrieval, heldout, 1, False)
    x = torch.from_numpy(features[:, 0])
    y = torch.from_numpy((target-priors[:, 0]).astype(np.float32))
    model = model_for(plan)
    assert torch.count_nonzero(model(x[:1])).item() == 0
    opt = torch.optim.Adam(model.parameters(), lr=CFG["learning_rate"])
    generator = torch.Generator().manual_seed(CFG["train_seed"])
    batch = CFG["batch_size"]
    per_epoch = (len(x)+batch-1)//batch
    checkpoints = {int(epoch)*per_epoch: f"e{epoch:03d}" for epoch in plan.get("epochs", [])}
    if "updates" in plan:
        checkpoints[plan["updates"]] = "matched"
    target_world = np.stack([world(a, s) for a,s in zip(vt,states)])
    aligned = np.stack([world(a[0],s) for a,s in zip(vp,states)])
    details = {"query_episodes": episodes, "train_queries":len(x), "validation_queries":len(vx),
               "updates_per_pass":per_epoch, "aligned":metrics(aligned,target_world), "checkpoints":{}}
    steps = 0
    while steps < max(checkpoints):
        for sel in torch.randperm(len(x), generator=generator).split(batch):
            opt.zero_grad(set_to_none=True)
            loss = ((model(x[sel])-y[sel])/128).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            if steps == 0:
                assert all(torch.isfinite(p.grad).all() for p in model.parameters())
            opt.step()
            steps += 1
            if steps in checkpoints:
                label = checkpoints[steps]
                path = output / (label+".pt")
                torch.save(model.state_dict(),path)
                with torch.inference_mode():
                    correction = model(torch.from_numpy(vx[:, 0])).numpy()
                pred = np.stack([world(a[0]+d,s) for a,d,s in zip(vp,correction,states)])
                details["checkpoints"][label] = {"updates":steps, "effective_epochs":steps/per_epoch,
                    "last_batch_loss":float(loss.detach()), "sha256":sha(path), "offline":metrics(pred,target_world)}
            if steps == max(checkpoints):
                break
    save(output/"training.json", details)


def evaluate(model, bank, cfg, seeds, output):
    from oracle_runner import OracleImageEnv, OracleVectorTrace, runner_module

    assert seeds == list(range(seeds[0], seeds[0]+len(seeds)))
    output.mkdir()
    with (output/"queries.jsonl").open("x", encoding="utf-8") as sink:
        policy = CorrectedPolicy(Retrieval(bank,"relative"), "aligned" if model is None else "residual", model, sink)
        with patch.object(runner_module, "PushTImageEnv", OracleImageEnv):
            runner = runner_module.PushTImageRunner(
                output_dir=str(output), n_train=0, n_train_vis=0, n_test=len(seeds),
                n_test_vis=0, test_start_seed=seeds[0], legacy_test=cfg["evaluation"]["legacy_test"],
                max_steps=300, n_obs_steps=1, n_action_steps=H, n_envs=1, fps=10,
                past_action=False, tqdm_interval_sec=60)
        proxy = OracleVectorTrace(runner.env,seeds,policy,300,cfg["diagnostics"]["position_sync_atol"])
        policy.state_provider=proxy.read_current_state
        runner.env=proxy
        try:
            official = runner.run(policy)
        finally:
            proxy.env.close()
    assert [r["evaluation_seed"] for r in proxy.records] == seeds
    steps = dict.fromkeys(seeds,0)
    for row in policy.query_records:
        seed = row["evaluation_seed"]
        assert row["current_step"] == steps[seed] and 1 <= row["executed_actions"] <= H
        steps[seed] += row["executed_actions"]
        _, indices = bank.data.action_chunk(row["episode"],row["frame"],H)
        assert row["episode"] in bank.episodes
        np.testing.assert_array_equal(row["source_indices"],indices)
        assert np.all((np.asarray(row["commands"]) >= 0) & (np.asarray(row["commands"]) <= 512))
    scores = np.array([r["official_episode_score"] for r in proxy.records])
    for row in proxy.records:
        seed = row["evaluation_seed"]
        assert steps[seed] == row["steps"]
        assert row["official_episode_score"] == official[f"test/sim_max_reward_{seed}"]
    np.testing.assert_allclose(scores.mean(),official["test/mean_score"],atol=1e-12)
    with (output/"rollouts.csv").open("x",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(proxy.records[0]))
        writer.writeheader()
        writer.writerows(proxy.records)
    return {"n":len(seeds), "mean":float(scores.mean()),
            "final":float(np.mean([r["final_score"] for r in proxy.records])),
            "success":sum(r["terminated"] for r in proxy.records),
            "queries":len(policy.query_records), "query_sha256":sha(output/"queries.jsonl")}


def load_model(plan, directory, key):
    checkpoint = directory/(key+".pt")
    assert sha(checkpoint) == read(directory/"training.json")["checkpoints"][key]["sha256"]
    model=model_for(plan)
    model.load_state_dict(torch.load(checkpoint,weights_only=True))
    return model.eval()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=("iterate","test"))
    parser.add_argument("round",type=int,nargs="?",choices=(1,2,3))
    args=parser.parse_args()
    torch.set_num_threads(2)
    cfg,base,manifest=load()
    assert tree_fingerprint(base["paths"]["dataset"])["sha256"] == manifest["dataset_fingerprint"]
    seeds=manifest["test_env_seeds"]
    assert not set(seeds)&set(VALIDATION)
    bank=GeometryBank(base["paths"]["dataset"],manifest["train_subsets"]["30"])
    assert len(bank.states) == 3806
    if args.phase=="iterate":
        if args.round is None:
            parser.error("iterate requires a round")
        path=EXP/f"round{args.round}.json"
        plan=read(path)
        output=OUT/f"round{args.round}"
        output.mkdir(parents=True,exist_ok=False)
        run={"status":"running","plan":plan,"config_sha256":sha(path),"code_sha256":sha(__file__),
             "split_sha256":cfg["split_sha256"],"dataset_fingerprint":manifest["dataset_fingerprint"],
             "bank_ids":bank.episodes.tolist(),"validation_seeds":VALIDATION,
             "common_config":CFG,"torch":torch.__version__}
        save(output/"run.json",run)
        try:
            fit(plan,bank,manifest,output)
            keys=read(output/"training.json")["checkpoints"]
            results={key:evaluate(load_model(plan,output,key),bank,cfg,VALIDATION,output/key)
                     for key in keys}
            best=max(results,key=lambda key: (results[key]["mean"],-keys[key]["updates"]))
            save(output/"validation.json",{"selected":best,"checkpoints":results})
            run["status"]="completed"
            print(json.dumps({"selected":best,"checkpoints":results}),flush=True)
        except Exception:
            import traceback
            run.update(status="failed",traceback=traceback.format_exc())
            raise
        finally:
            save(output/"run.json",run)
    else:
        if args.round is not None:
            parser.error("test evaluates all three rounds together")
        sources=[]
        for n in (1,2,3):
            directory=OUT/f"round{n}"
            path=EXP/f"round{n}.json"
            run=read(directory/"run.json")
            assert run["status"]=="completed" and run["config_sha256"]==sha(path)
            key=read(directory/"validation.json")["selected"]
            sources.append((n,directory,key,read(path)))
        output=OUT/"test"
        output.mkdir(exist_ok=False)
        report={"aligned":evaluate(None,bank,cfg,seeds,output/"aligned")}
        for n,directory,key,plan in sources:
            report[f"round{n}"]=evaluate(load_model(plan,directory,key),bank,cfg,seeds,output/f"round{n}")
            report[f"round{n}"]["checkpoint"]=key
        def scores(label):
            with (output/label/"rollouts.csv").open(newline="",encoding="utf-8") as f:
                return np.array([float(r["official_episode_score"]) for r in csv.DictReader(f)])
        old=ROOT/"experiments/residual_study/evidence/round3_evaluate_aligned.csv"
        with old.open(newline="",encoding="utf-8") as f:
            np.testing.assert_allclose(scores("aligned"),[float(r["official_episode_score"]) for r in csv.DictReader(f)],
                                       atol=1e-12,rtol=0)
        rng=np.random.default_rng(CFG["bootstrap_seed"])
        draws=rng.integers(0,len(seeds),(CFG["bootstrap_draws"],len(seeds)))
        for n in (1,2,3):
            label=f"round{n}"
            delta=scores(label)-scores("aligned")
            report[label].update(delta=float(delta.mean()),ci95=np.quantile(delta[draws].mean(axis=1),[.025,.975]).tolist())
        save(output/"summary.json",report)
        print(json.dumps(report),flush=True)


if __name__=="__main__":
    main()
