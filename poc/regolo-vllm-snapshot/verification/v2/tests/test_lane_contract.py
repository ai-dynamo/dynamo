import hashlib
import json
import pathlib
import stat
import tempfile
import unittest

from _support import load_harness


PINNED_IDENTITY = {
    "protocol_version": "V2.0-draft",
    "baseline_protocol_version": "V0.1",
    "baseline_protocol_sha256": "1a380518c87764574e08940b24fac63882f6e87108ab23f4a68ada22075b9511",
    "baseline_results_sha256": "b55e89421e759c06a3be0a57a6030b7e12445662d01988efc566e2ed848b91f6",
    "baseline_phase_analysis_sha256": "e6dec04dcf1b0cf484889587126dec50f11582f39527d9c2c81fcf716b91a36d",
    "dynamo_version": "v1.3.0",
    "dynamo_commit": "8ce9e22f11576402102ea9d8b8e46233f5430a0d",
    "model": "openai/gpt-oss-20b",
    "model_revision": "6cee5e81ee83917806bbde320786a8fb61efebee",
    "gpu_product": "NVIDIA L40S",
    "driver_version": "580.178.04",
    "node": "ec213103",
    "source_image": "docker.io/vllm/vllm-openai@sha256:c2f3b1b964e47809b722b5e75b61b1e7b39a50f70388cf2bf2418f16a9f31da2",
    "candidate_image": "docker.io/library/regolo-vllm-snapshot@sha256:84e626a76456827946ada12120fd6842ae7eefc4b2a4005663bab137385f030a",
    "compatibility_hash": "a42c07d50e863d43838bcf0ec3c07c544324579f3df80cc08047191838e1e805",
    "command": ["vllm"],
    "args": [
        "serve", "openai/gpt-oss-20b", "--revision",
        "6cee5e81ee83917806bbde320786a8fb61efebee", "--host", "0.0.0.0",
        "--port", "8000", "--max-model-len", "4096", "--gpu-memory-utilization", "0.85",
    ],
}


def lane(seed=20260814):
    value = {
        "baseline_group": "v1-driver580-v3",
        "single_mutation": "observer_only_phase_collection",
        "seed": seed,
        "identity": PINNED_IDENTITY,
        "workload": {"kind": "synthetic", "prompt": "The answer to 1+1 is"},
    }
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    value["digest"] = hashlib.sha256(canonical).hexdigest()
    return value


class LaneContractTests(unittest.TestCase):
    def setUp(self):
        self.harness = load_harness()

    def test_lane_requires_real_frozen_identity_and_digest_independent_of_authorization(self):
        value = lane()
        self.assertEqual(
            self.harness.validate_lane(value, expected_identity=PINNED_IDENTITY), value
        )
        self.assertNotIn("authorization", value)
        for field, invalid in (
            ("dynamo_commit", "0" * 40),
            ("model", "other/model"),
            ("compatibility_hash", "0" * 64),
        ):
            candidate = json.loads(json.dumps(value))
            candidate["identity"][field] = invalid
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.harness.validate_lane(
                    candidate,
                    expected_identity=PINNED_IDENTITY,
                    authorization={"execution_authorized": True},
                )
        self.assertEqual(
            self.harness.validate_lane(
                value,
                expected_identity=PINNED_IDENTITY,
                authorization={"execution_authorized": False},
            ),
            value,
        )

    def test_plan_is_deterministic_paired_blinded_and_key_is_seed_randomized(self):
        first = self.harness.make_paired_blinded_plan(lane())
        self.assertEqual(first, self.harness.make_paired_blinded_plan(lane()))
        schedule = first["schedule"]
        key = first["unblinding_key"]
        self.assertEqual(len(schedule), 40)
        self.assertEqual(set(key), {"A", "B"})
        self.assertEqual(set(key.values()), {"cold", "restore"})
        self.assertNotIn("cold", json.dumps(schedule).lower())
        self.assertNotIn("restore", json.dumps(schedule).lower())
        self.assertEqual(
            {tuple(sorted(self.harness.make_paired_blinded_plan(lane(seed))["unblinding_key"].items()))
             for seed in range(1, 33)},
            {(("A", "cold"), ("B", "restore")), (("A", "restore"), ("B", "cold"))},
        )
        for block in range(1, 21):
            runs = [run for run in schedule if run["block"] == block]
            self.assertEqual(len(runs), 2)
            self.assertEqual({run["opaque_arm"] for run in runs}, {"A", "B"})
            self.assertEqual({run["sequence_in_block"] for run in runs}, {1, 2})
            self.assertTrue(all(run["run_id"].startswith("v2-") for run in runs))

    def test_seal_plan_writes_schedule_and_private_key_separately(self):
        plan = self.harness.make_paired_blinded_plan(lane())
        with tempfile.TemporaryDirectory() as directory:
            schedule_path, key_path = self.harness.seal_plan(plan, pathlib.Path(directory))
            self.assertNotEqual(schedule_path, key_path)
            self.assertEqual(stat.S_IMODE(key_path.stat().st_mode), 0o600)
            self.assertEqual(json.loads(schedule_path.read_text()), plan["schedule"])
            self.assertEqual(json.loads(key_path.read_text()), plan["unblinding_key"])
            self.assertNotIn("cold", schedule_path.read_text().lower())
            self.assertNotIn("restore", schedule_path.read_text().lower())


if __name__ == "__main__":
    unittest.main()
