import json
import random
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]


class SplitContractTests(unittest.TestCase):
    def test_exact_counts_disjointness_and_nesting(self):
        contract = json.loads((HERE / "configs" / "lift_ph_split42.json").read_text(encoding="utf-8"))
        valid = contract["heldout_validation_episode_ids"]
        pool = contract["ordered_train_pool_episode_ids"]
        subsets = contract["train_subsets"]
        self.assertEqual(contract["split_seed"], 42)
        self.assertEqual(contract["episode_count_total"], 200)
        self.assertEqual(len(valid), 20)
        self.assertEqual(len(pool), 180)
        self.assertEqual(len(set(valid)), 20)
        self.assertEqual(len(set(pool)), 180)
        self.assertFalse(set(valid) & set(pool))
        self.assertEqual(len(subsets["30"]), 30)
        self.assertEqual(len(subsets["50"]), 50)
        self.assertEqual(len(subsets["100"]), 100)
        self.assertEqual(subsets["30"], pool[:30])
        self.assertEqual(subsets["50"], pool[:50])
        self.assertEqual(subsets["100"], pool[:100])
        self.assertTrue(set(subsets["30"]) < set(subsets["50"]) < set(subsets["100"]))
        self.assertFalse(set(valid) & set(subsets["100"]))
        self.assertEqual(len(set(valid) | set(pool)), 200)

    def test_seeded_order_matches_frozen_algorithm(self):
        contract = json.loads((HERE / "configs" / "lift_ph_split42.json").read_text(encoding="utf-8"))
        ids = ["demo_%d" % index for index in range(200)]
        random.Random(42).shuffle(ids)
        self.assertEqual(contract["heldout_validation_episode_ids"], ids[:20])
        self.assertEqual(contract["ordered_train_pool_episode_ids"], ids[20:])

    def test_eval_conditions_match_frozen_seed_list(self):
        contract = json.loads((HERE / "configs" / "lift_ph_split42.json").read_text(encoding="utf-8"))
        fixtures = json.loads((HERE / "results" / "evaluation_initial_states.json").read_text(encoding="utf-8"))
        self.assertEqual(contract["test_env_seeds"], list(range(100000, 100050)))
        self.assertEqual(contract["test_env_seeds"], fixtures["seed_contract"]["seeds"])
        self.assertTrue(contract["evaluation_initial_state_contract"]["repeat_checks_passed"])


if __name__ == "__main__":
    unittest.main()
