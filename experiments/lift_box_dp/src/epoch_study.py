"""Immutable Lift-10 epoch-study protocol, derived from the original Lift split."""
from common import ROOT, CONFIG_PATH, contract, read_json, sha256

STUDY_DIR = ROOT / "experiments/exp_002_lift10_epochs"
STUDY_PATH = STUDY_DIR / "config.json"


def study(supplement=False):
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
    cfg["train_ids"] = ids
    cfg["epochs"] = cfg["dp_epochs"][-1]
    return cfg, alt, dp, split, image
