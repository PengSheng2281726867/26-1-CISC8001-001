"""Resolve the fixed-budget study using the existing Lift split and fixtures."""
from common import CONFIG_PATH, ROOT, contract, read_json, sha256

PROTOCOL_PATH = ROOT.parent / "matched_budget/protocol.json"
FIXTURE_DIR = ROOT / "experiments/exp_002_lift10_epochs"


def matched(scale, seed, smoke=False):
    cfg, alt, dp, split, image = contract()
    protocol = read_json(PROTOCOL_PATH)
    design = protocol["lift"]
    if scale not in design["scales"] or seed not in protocol["model_seeds"]:
        raise ValueError("Unknown fixed-budget condition")
    ids = split["train_subsets"][str(max(scale, 30))][:scale]
    if len(set(ids)) != scale or set(ids) & set(split["heldout_validation_episode_ids"]):
        raise ValueError("Invalid nested Lift training subset")
    cfg.update(train_ids=ids, train_count=scale, batch_size=protocol["batch_size"],
               model_seed=seed, shuffle_seed=seed + 1, augmentation_seed=seed + 2,
               sampling_seed=design["sampling_seed"], max_updates=2 if smoke else design["updates"],
               dp_warmup_steps=175, checkpoint_selection=protocol["checkpoint_selection"],
               fixed_budget_smoke=smoke, test_workers=design["workers"],
               matched_protocol_sha256=sha256(PROTOCOL_PATH), parent_config_sha256=sha256(CONFIG_PATH))
    return cfg, alt, dp, split, image


def fixtures(split, alt):
    design = read_json(PROTOCOL_PATH)["lift"]
    original_path = FIXTURE_DIR / "validation_initial_states.json"
    extra_path = FIXTURE_DIR / "validation_extra16_initial_states.json"
    original, extra = read_json(original_path), read_json(extra_path)
    if extra["original_sha256"] != sha256(original_path):
        raise ValueError("Original validation conditions changed")
    rows = original["initial_conditions"] + extra["initial_conditions"]
    seeds = list(range(design["validation_seed_start"], design["validation_seed_start"] + design["validation_count"]))
    hashes = {row["sim_state_sha256"] for row in rows}
    test_path = alt / split["evaluation_initial_state_contract"]["path"]
    test_hashes = {row["sim_state_sha256"] for row in read_json(test_path)["initial_conditions"]}
    if [row["seed"] for row in rows] != seeds or len(hashes) != len(seeds) or hashes & test_hashes:
        raise ValueError("Unexpected Lift validation conditions")
    return design, seeds, dict(zip(seeds, rows)), {path.name: sha256(path) for path in (original_path, extra_path)}
