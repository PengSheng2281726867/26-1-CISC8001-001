"""Immutable Lift-10 epoch-study protocol, derived from the original Lift split."""
from common import ROOT, CONFIG_PATH, contract, read_json, sha256

STUDY_DIR = ROOT / "experiments/exp_002_lift10_epochs"
STUDY_PATH = STUDY_DIR / "config.json"


SCALE50_PATH = ROOT / "experiments/exp_003_lift50_epochs/config.json"


def study(supplement=False, scale=10, method="alt"):
    if scale not in (10, 50) or (scale == 50 and supplement):
        raise ValueError("Invalid ALT study scale or supplement combination")
    cfg, alt, dp, split, image = contract()
    protocol = read_json(STUDY_PATH)
    ids = split["train_subsets"]["30"][:protocol["train_count"]]
    val = protocol["validation_seeds"]
    if len(ids) != 10 or len(val) != len(set(val)) or set(val) & set(split["test_env_seeds"]):
        raise ValueError("Invalid Lift-10 training or validation selection")
    cfg.update(protocol)
    cfg["study_config_sha256"] = sha256(STUDY_PATH)
    cfg["parent_config_sha256"] = sha256(CONFIG_PATH)
    if supplement:
        supplement_path = STUDY_DIR / "supplement.json"
        extra = read_json(supplement_path)
        if set(extra) != {"alt_epochs", "dp_epochs"}:
            raise ValueError("Supplement may only change checkpoint epochs")
        for epochs in extra.values():
            if not epochs or epochs != sorted(set(epochs)) or epochs[0] < 1:
                raise ValueError("Supplement epochs must be positive and strictly increasing")
        cfg.update(extra)
        cfg["supplement_config_sha256"] = sha256(supplement_path)
    if scale == 50:
        scale_protocol = read_json(SCALE50_PATH)
        scale_ids = split["train_subsets"]["50"]
        if (set(scale_protocol) != {"train_count", "alt_epochs"} or scale_protocol["train_count"] != 50
                or len(set(scale_ids)) != 50 or scale_ids[:30] != split["train_subsets"]["30"]
                or set(scale_ids) & set(split["heldout_validation_episode_ids"])):
            raise ValueError("Invalid nested 50-episode training subset")
        epochs = scale_protocol["alt_epochs"]
        if not epochs or epochs != sorted(set(epochs)) or epochs[0] < 1:
            raise ValueError("Invalid ALT-50 checkpoint epochs")
        cfg.update(scale_protocol)
        cfg["scale_config_sha256"] = sha256(SCALE50_PATH)
        ids = scale_ids
        if method == "dp":
            dp50_path = SCALE50_PATH.with_name("dp_config.json")
            dp50 = read_json(dp50_path)
            epochs = dp50["dp_epochs"]
            if (set(dp50) != {"dp_epochs", "dp_warmup_steps"} or not epochs
                    or epochs != sorted(set(epochs)) or epochs[0] < 1 or dp50["dp_warmup_steps"] < 0):
                raise ValueError("Invalid DP-50 training protocol")
            cfg.update(dp50)
            cfg["dp50_config_sha256"] = sha256(dp50_path)
    cfg["train_ids"] = ids
    cfg["epochs"] = cfg["dp_epochs"][-1]
    return cfg, alt, dp, split, image
