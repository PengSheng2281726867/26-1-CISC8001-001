"""Resolve the full demonstration pool for the Lift saturation study."""
from common import CONFIG_PATH, ROOT, contract, read_json, sha256

PROTOCOL = ROOT.parent / "full_pool/lift.json"


def full_pool(smoke=False):
    cfg, alt, dp, split, image = contract()
    design = read_json(PROTOCOL)
    pool = split["ordered_train_pool_episode_ids"]
    heldout = split["heldout_validation_episode_ids"]
    if (len(set(pool)) != design["train_count"] or len(pool) != design["train_count"]
            or len(heldout) != len(set(heldout)) or set(pool) & set(heldout)
            or len(pool) + len(heldout) != split["episode_count_total"]
            or pool[:100] != split["train_subsets"]["100"]):
        raise ValueError("Invalid frozen Lift full train pool")
    cfg.update(design, train_ids=pool, epochs=design["dp_epochs"][-1],
               full_pool_protocol_sha256=sha256(PROTOCOL), parent_config_sha256=sha256(CONFIG_PATH),
               full_pool_smoke=smoke)
    if smoke:
        cfg.update(max_updates=2, alt_epochs=[1], dp_epochs=[1], epochs=1)
    return cfg, alt, dp, split, image
