"""Check protocol snapshots against the design relevant to a task."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "protocol.json"


def compatible_hashes(task, protocol):
    snapshots = [PROTOCOL_PATH]
    if task == "pusht":
        snapshots.append(HERE / "protocol_before_lift_fixtures.json")
    fields = (task, "model_seeds", "batch_size", "checkpoint_selection")
    hashes = set()
    for path in snapshots:
        content = path.read_bytes()
        snapshot = json.loads(content)
        if all(snapshot[key] == protocol[key] for key in fields):
            hashes.add(hashlib.sha256(content).hexdigest())
    return hashes
