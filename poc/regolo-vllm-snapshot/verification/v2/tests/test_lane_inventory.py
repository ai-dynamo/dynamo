import json
import pathlib
import unittest


V2_ROOT = pathlib.Path(__file__).resolve().parents[1]
LANES = V2_ROOT / "lanes"


class LaneInventoryTests(unittest.TestCase):
    def test_exactly_ten_pinned_lanes_cover_v2a_and_each_feature_seed(self):
        manifests = sorted(LANES.glob("*.json"))
        self.assertEqual(len(manifests), 10)
        lanes = [json.loads(path.read_text()) for path in manifests]
        self.assertEqual({lane["seed"] for lane in lanes}, set(range(20260814, 20260824)))
        self.assertEqual(
            {lane["name"] for lane in lanes},
            {
                "v2-a", "kv-sleep-l1", "right-sized-kv", "kv-accordion",
                "cuda-graph-diet", "criu-direct-io", "ram-hot-tier",
                "frontend-engine-split", "ghost-kv", "gms",
            },
        )
        v2a = [lane for lane in lanes if lane["seed"] == 20260814]
        self.assertEqual(len(v2a), 1)
        self.assertEqual(v2a[0]["stage"], "V2-A")
        self.assertEqual(v2a[0]["single_mutation"], "observer_only_phase_collection")
        self.assertTrue(all(lane["stage"] != "V2-A" for lane in lanes if lane["seed"] != 20260814))

    def test_every_lane_is_single_mutation_and_has_unique_isolated_identities(self):
        lanes = [json.loads(path.read_text()) for path in sorted(LANES.glob("*.json"))]
        required = {
            "name", "stage", "seed", "baseline_group", "single_mutation", "worktree",
            "branch", "artifact_directory", "image_digest", "checkpoint_identity", "workload", "gates",
        }
        for lane in lanes:
            with self.subTest(lane=lane.get("name")):
                self.assertEqual(set(lane), required | {"gms"} if lane.get("name") == "gms" else required)
                self.assertIsInstance(lane["single_mutation"], str)
                self.assertTrue(lane["single_mutation"].strip())
                self.assertNotIsInstance(lane["single_mutation"], (list, dict))
                self.assertIsInstance(lane["workload"], dict)
                self.assertIsInstance(lane["gates"], dict)
        for field in ("name", "worktree", "branch", "artifact_directory", "image_digest", "checkpoint_identity"):
            with self.subTest(field=field):
                values = [lane[field] for lane in lanes]
                self.assertEqual(len(values), len(set(values)))

    def test_gms_is_pinned_to_131_r610_without_raw_cross_driver_comparison_and_nested_modelexpress_is_ineligible(self):
        lanes = [json.loads(path.read_text()) for path in sorted(LANES.glob("*.json"))]
        by_name = {lane["name"]: lane for lane in lanes}
        gms = by_name["gms"]
        self.assertEqual(gms["gms"]["dynamo_version"], "v1.3.1")
        self.assertEqual(gms["gms"]["driver_family"], "R610")
        self.assertFalse(gms["gms"]["raw_cross_driver_comparison"])
        self.assertEqual(gms["gms"]["modelexpress"]["status"], "INELIGIBLE")
        self.assertTrue(gms["gms"]["modelexpress"]["reason"].strip())


if __name__ == "__main__":
    unittest.main()
