import json
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]


class DatasetContractTests(unittest.TestCase):
    def test_audited_schema_and_alignment(self):
        audit = json.loads((HERE / "results" / "dataset_audit.json").read_text(encoding="utf-8"))
        image = audit["image_dataset"]
        self.assertEqual(image["episode_count"], 200)
        self.assertEqual(image["total_frames"], 9666)
        self.assertEqual(image["action_shape_per_episode"][1], 7)
        self.assertEqual(image["episode_length_min_median_max"], [36, 48, 64])
        self.assertEqual(image["observation_fields"]["agentview_image"]["shape_per_episode"][1:], [84, 84, 3])
        self.assertEqual(image["observation_fields"]["agentview_image"]["dtype"], "uint8")
        self.assertEqual(image["observation_fields"]["robot0_eef_pos"]["shape_per_episode"][1:], [3])
        self.assertEqual(image["observation_fields"]["robot0_eef_quat"]["shape_per_episode"][1:], [4])
        self.assertEqual(image["observation_fields"]["robot0_gripper_qpos"]["shape_per_episode"][1:], [2])
        self.assertFalse(image["actions_abs_present"])
        self.assertTrue(image["next_obs_present"])
        self.assertTrue(audit["cross_file_checks"]["same_episode_ids"])
        self.assertTrue(audit["cross_file_checks"]["all_shared_action_arrays_identical"])
        self.assertTrue(audit["cross_file_checks"]["all_image_episode_lengths_aligned"])
        self.assertEqual(
            audit["alignment_contract"]["expert_action_at_t"], "actions[t]"
        )
        self.assertEqual(len(audit["next_obs_alignment_checks"]), 15)
        self.assertTrue(all(row["exact_match"] for row in audit["next_obs_alignment_checks"]))


if __name__ == "__main__":
    unittest.main()
