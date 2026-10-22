import hashlib
import json
import unittest
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parents[1]


class RunConfigTests(unittest.TestCase):
    def test_resolved_config_matches_frozen_split(self):
        config_path = HERE / "configs" / "alt_lift30.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        contract = json.loads((HERE / "configs" / "lift_ph_split42.json").read_text(encoding="utf-8"))
        self.assertEqual(config["experiment_id"], "alt_lift_ph30_image_proprio_seed42_v1")
        self.assertEqual(config["training"]["train_scale"], 30)
        self.assertEqual(config["training"]["model_seed"], 42)
        self.assertEqual(config["objective"]["temperature"], 0.4)
        self.assertEqual(config["execution"]["action_chunk_length"], 1)
        self.assertEqual(config["memory"]["retrieval"], "cosine_top_1")
        self.assertTrue(config["model"]["backbone_fine_tuned"])
        self.assertEqual(
            contract["run_config"]["sha256"],
            hashlib.sha256(config_path.read_bytes()).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
