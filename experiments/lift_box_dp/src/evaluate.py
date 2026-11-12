"""Evaluate frozen policies on the same Lift initial conditions and success checks."""
import argparse
import contextlib
import csv
import math
import pickle
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from common import ROOT, add_references, contract, read_json, sha256, write_json, wsl_path
from model import batch_observations, make_model


class Simulator:
    def __init__(self, alt, stderr_path):
        self.log = stderr_path.open("wb")
        self.process = subprocess.Popen(
            ["wsl.exe", "-d", "Ubuntu", "--", "env", "LP_NUM_THREADS=2", "OMP_NUM_THREADS=1",
             "OPENBLAS_NUM_THREADS=1", "bash", wsl_path(alt / "scripts/with_legacy_env.sh"),
             wsl_path(ROOT / "src/simulator.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
        )

    def send(self, command, value):
        try:
            pickle.dump((command, value), self.process.stdin, protocol=4)
            self.process.stdin.flush()
            return pickle.load(self.process.stdout)
        except (EOFError, BrokenPipeError, pickle.UnpicklingError) as exc:
            raise RuntimeError(f"WSL simulator pipe failed. See {self.log.name}") from exc

    def close(self):
        if self.process.poll() is None:
            try:
                pickle.dump(("close", None), self.process.stdin, protocol=4)
                self.process.stdin.flush()
                self.process.wait(timeout=15)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait()
        with contextlib.suppress(OSError):
            self.process.stdin.close()
        self.process.stdout.close()
        self.log.close()


def load_alt(alt, image, train_ids, device):
    from alt_lift.data import ProprioNormalizer
    from alt_lift.encoder import VisualProprioEncoder
    from alt_lift.memory import MemoryBank
    from alt_lift.policy import ActionRetrievalPolicy
    import h5py

    run_id = "alt_lift_ph30_image_proprio_seed42_v1"
    run = alt / "artifacts/runs" / run_id
    expected = read_json(alt / "experiments/runs" / run_id / "artifact_manifest.json")
    checkpoint = run / "checkpoint_final.pt"
    bank_path = run / "memory_bank.npz"
    hashes = {p.name: sha256(p) for p in (checkpoint, bank_path)}
    if any(digest not in str(expected) for digest in hashes.values()):
        raise ValueError("ALT checkpoint or bank differs from original run")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = VisualProprioEncoder(state["variant"], pretrained_path=None)
    model.load_state_dict(state["model_state_dict"])
    normalizer = ProprioNormalizer(state["proprio_mean"].numpy(), state["proprio_std"].numpy())
    bank = MemoryBank.load(bank_path)
    if set(bank.episode_ids.tolist()) != set(train_ids):
        raise ValueError("ALT bank does not match the frozen training IDs")
    with h5py.File(str(image), "r") as h5:
        actions = {key: np.asarray(h5["data"][key]["actions"]) for key in train_ids}
    policy = ActionRetrievalPolicy(model.to(device).eval(), bank, normalizer, actions, device)
    return lambda obs: np.stack([policy.act(item, chunk_length=1)[0] for item in obs]), hashes


def load_dp(run, cfg, device, smoke):
    checkpoint = run / "final_ema.pt"
    metadata = read_json(run / "training.json")
    digest = sha256(checkpoint)
    if digest != metadata["checkpoint_sha256"]:
        raise ValueError("DP checkpoint checksum mismatch")
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if weights["config"] != cfg or weights["smoke"] != smoke:
        raise ValueError("DP checkpoint protocol does not match evaluation")
    model = make_model(cfg)
    model.load_state_dict(weights["weights"])
    model.to(device).eval()

    def act(obs):
        action = model.predict_action(batch_observations(obs, device))["action"]
        return action[:, 0].cpu().numpy()

    return act, {"final_ema.pt": digest}


def wilson(count, total):
    rate = count / total
    z2 = 1.959963984540054 ** 2
    center = (count + z2 / 2) / (total + z2)
    margin = math.sqrt(z2 * (rate * (1 - rate) * total + z2 / 4)) / (total + z2)
    return [max(0, center - margin), min(1, center + margin)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", choices=["dp", "alt"])
    parser.add_argument("run_id")
    parser.add_argument("evaluation_id")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg, alt, dp, split, image = contract()
    add_references(alt, dp)
    settings = read_json(alt / split["evaluation_config"]["path"])
    seeds = settings["test_seeds"][:1] if args.smoke else settings["test_seeds"]
    horizon = 5 if args.smoke else settings["horizon"]
    workers = min(cfg["eval_workers"], len(seeds))
    run = ROOT / "artifacts" / args.run_id
    output = run / args.evaluation_id
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    if args.policy == "dp":
        act, hashes = load_dp(run, cfg, device, args.smoke)
    else:
        act, hashes = load_alt(alt, image, split["train_subsets"]["30"], device)
    write_json(output / "protocol.json", {
        "policy": args.policy, "run_id": args.run_id, "evaluation_id": args.evaluation_id,
        "config": cfg, "hashes": hashes, "test_seeds": seeds, "horizon": horizon,
        "smoke": args.smoke, "workers": workers, "execution_horizon": 1,
        "initial_states_sha256": split["evaluation_initial_state_contract"]["sha256"],
        "evaluation_config_sha256": split["evaluation_config"]["sha256"],
    })
    run_rollouts(act, cfg, alt, seeds, horizon, workers, output)


def run_rollouts(act, cfg, alt, seeds, horizon, workers, output, validation_fixtures=None):
    torch.manual_seed(cfg["sampling_seed"])
    start = time.perf_counter()
    results = []
    with contextlib.ExitStack() as stack:
        pool = stack.enter_context(ThreadPoolExecutor(max_workers=workers))
        simulators = []
        for i in range(workers):
            simulator = Simulator(alt, output / f"simulator{i}.log")
            stack.callback(simulator.close)
            simulators.append(simulator)
        episodes = stack.enter_context((output / "rollouts.csv").open("x", newline="", encoding="utf-8"))
        traces = stack.enter_context((output / "actions.csv").open("x", newline="", encoding="utf-8"))
        trace = csv.writer(traces)
        trace.writerow(["seed", "step", "above_threshold", "reward", "action"])
        writer = csv.DictWriter(episodes, fieldnames=[
            "seed", "steps", "ever_success", "first_success_step", "final_success",
            "longest_hold_steps", "hold_success", "terminal_hold_steps", "final_reward",
            "policy_ms_per_step", "terminated",
        ])
        writer.writeheader()
        for offset in range(0, len(seeds), workers):
            batch = seeds[offset:offset + workers]
            states = [{"seed": seed, "steps": 0, "first": None, "hold": 0, "longest": 0,
                       "reward": 0, "policy_ms": 0, "done": False} for seed in batch]
            if validation_fixtures is None:
                reset_command = "reset"
                reset_values = batch
            else:
                reset_command = "reset_validation"
                reset_values = [(seed, validation_fixtures[seed]) for seed in batch]
            obs = list(pool.map(lambda pair: pair[0].send(reset_command, pair[1]),
                                zip(simulators, reset_values)))
            for step in range(1, horizon + 1):
                active = [i for i, state in enumerate(states) if not state["done"]]
                if not active:
                    break
                tick = time.perf_counter()
                with torch.inference_mode():
                    actions = act([obs[i] for i in active])
                millis = (time.perf_counter() - tick) * 1000
                if actions.shape != (len(active), 7) or not np.isfinite(actions).all():
                    raise ValueError("Policy returned invalid action shape or value")
                if np.max(np.abs(actions)) > 1.00001:
                    raise ValueError("Policy command is outside original controller bounds")
                transitions = pool.map(lambda pair: pair[0].send("step", pair[1]),
                                       zip((simulators[i] for i in active), actions))
                for i, action, transition in zip(active, actions, transitions):
                    obs[i], above, reward, done = transition
                    state = states[i]
                    state["steps"] = step
                    state["reward"] = reward
                    state["done"] = done
                    state["policy_ms"] += millis
                    if above and state["first"] is None:
                        state["first"] = step
                    state["hold"] = state["hold"] + 1 if above else 0
                    state["longest"] = max(state["longest"], state["hold"])
                    trace.writerow([state["seed"], step, int(above), reward, action.tolist()])
                if step % 100 == 0:
                    print(f"seeds={batch} step={step}/{horizon} elapsed={time.perf_counter() - start:.0f}s", flush=True)
            for state in states:
                row = dict(seed=state["seed"], steps=state["steps"], ever_success=state["first"] is not None,
                           first_success_step=state["first"], final_success=state["hold"] > 0,
                           longest_hold_steps=state["longest"], hold_success=state["longest"] >= cfg["hold_steps"],
                           terminal_hold_steps=state["hold"], final_reward=state["reward"],
                           policy_ms_per_step=state["policy_ms"] / state["steps"], terminated=state["done"])
                writer.writerow(row)
                results.append(row)
            episodes.flush()
            traces.flush()
            print(f"completed seeds={batch}: {sum(row['ever_success'] for row in results)}/{len(results)} ever",
                  flush=True)
    metrics = {"n_rollouts": len(results), "wall_seconds": time.perf_counter() - start,
               "mean_batch_policy_ms": float(np.mean([r["policy_ms_per_step"] for r in results]))}
    for key in ("ever_success", "final_success", "hold_success"):
        count = sum(r[key] for r in results)
        metrics[key] = {"count": count, "rate": count / len(results), "wilson95": wilson(count, len(results))}
    write_json(output / "metrics.json", metrics)
    print(metrics, flush=True)
    return metrics


if __name__ == "__main__":
    main()
