import hashlib
import importlib.util
import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
LIB = ROOT / "implementation" / "lib" / "snapshot_poc.py"


def load_library():
    spec = importlib.util.spec_from_file_location("snapshot_poc", LIB)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {LIB}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CompatibilityHashTests(unittest.TestCase):
    def setUp(self):
        self.lib = load_library()
        self.identity = {
            "image_digest": "registry.example/vllm@sha256:" + "a" * 64,
            "model_revision": "0123456789abcdef",
            "gpu_product": "NVIDIA L40S",
            "driver_version": "580.65.06",
            "command": ["snapshot-entrypoint", "--"],
            "args": ["vllm", "serve", "org/model", "--port", "8000"],
            "pod_spec": {"nodeName": "gpu-a", "containers": [{"name": "server"}]},
        }

    def test_hash_is_sha256_of_canonical_identity(self):
        got = self.lib.compatibility_hash(self.identity)
        canonical = json.dumps(
            self.identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        self.assertEqual(got, hashlib.sha256(canonical).hexdigest())

    def test_hash_rejects_mutable_image_tag(self):
        self.identity["image_digest"] = "registry.example/vllm:latest"
        with self.assertRaises(ValueError):
            self.lib.compatibility_hash(self.identity)

    def test_hash_rejects_missing_or_extra_fields(self):
        for identity in (
            {k: v for k, v in self.identity.items() if k != "driver_version"},
            {**self.identity, "cuda": "13"},
        ):
            with self.assertRaises(ValueError):
                self.lib.compatibility_hash(identity)


class RandomizationTests(unittest.TestCase):
    def setUp(self):
        self.lib = load_library()

    def test_plan_is_reproducible_paired_and_blinded(self):
        first = self.lib.make_run_plan(seed=20260811, paired_blocks=10)
        second = self.lib.make_run_plan(seed=20260811, paired_blocks=10)
        self.assertEqual(first, second)
        schedule, key = first
        self.assertEqual(len(schedule), 20)
        self.assertEqual(set(key), {"A", "B"})
        self.assertEqual(set(key.values()), {"cold", "restore"})
        self.assertNotIn("cold", json.dumps(schedule).lower())
        self.assertNotIn("restore", json.dumps(schedule).lower())
        for block in range(1, 11):
            runs = [r for r in schedule if r["block"] == block]
            self.assertEqual(len(runs), 2)
            self.assertEqual({r["opaque_arm"] for r in runs}, {"A", "B"})


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.lib = load_library()

    @staticmethod
    def records(restore_latency=20.0, restore_ok=True, memory_ratio=1.0):
        rows = []
        for block in range(1, 11):
            for arm, latency, memory, success in (
                ("A", 80.0, 40000.0, None),
                ("B", restore_latency, 40000.0 * memory_ratio, restore_ok),
            ):
                rows.append(
                    {
                        "run_id": f"v1-{block:02d}-{arm}",
                        "block": block,
                        "opaque_arm": arm,
                        "ready_s": latency - 2,
                        "http_200_s": latency - 1,
                        "first_token_s": latency,
                        "gpu_memory_mib": memory,
                        "valid_response": True,
                        "restore_success": success,
                        "checkpoint_duration_s": 120.0 if block == 1 and arm == "A" else None,
                        "checkpoint_size_bytes": 1000000000 if block == 1 and arm == "A" else None,
                        "excluded": False,
                        "exclusion_reason": None,
                        "cluster_incident_evidence": None,
                    }
                )
        return rows

    def test_go_requires_all_frozen_criteria(self):
        report = self.lib.summarize(self.records(), {"A": "cold", "B": "restore"})
        self.assertEqual(report["decision"], "Go")
        self.assertEqual(report["median_speedup"], 4.0)
        self.assertEqual(report["restore_success_rate"], 1.0)
        self.assertEqual(report["break_even_restores"], 2.0)

    def test_correct_but_slow_is_optimize(self):
        report = self.lib.summarize(
            self.records(restore_latency=40.0), {"A": "cold", "B": "restore"}
        )
        self.assertEqual(report["decision"], "Optimize")

    def test_failure_or_memory_drift_is_no_go(self):
        failed = self.lib.summarize(
            self.records(restore_ok=False), {"A": "cold", "B": "restore"}
        )
        drifted = self.lib.summarize(
            self.records(memory_ratio=1.051), {"A": "cold", "B": "restore"}
        )
        self.assertEqual(failed["decision"], "No-Go")
        self.assertEqual(drifted["decision"], "No-Go")

    def test_exclusion_without_event_evidence_invalidates_protocol(self):
        rows = self.records()
        rows[0]["excluded"] = True
        rows[0]["exclusion_reason"] = "node issue"
        with self.assertRaises(ValueError):
            self.lib.summarize(rows, {"A": "cold", "B": "restore"})

    def test_single_run_exclusion_invalidates_pair(self):
        rows = self.records()
        rows[0].update(
            excluded=True,
            exclusion_reason="cluster node became NotReady",
            cluster_incident_evidence="events/v1-01-A.json",
        )
        with self.assertRaises(ValueError):
            self.lib.summarize(rows, {"A": "cold", "B": "restore"})


if __name__ == "__main__":
    unittest.main()
