import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
V0 = ROOT / "verification/v0/protocol.json"
V2 = ROOT / "verification/v2/protocol.draft.json"


class V2ProtocolContractTests(unittest.TestCase):
    def setUp(self):
        self.v0 = json.loads(V0.read_text())
        self.v2 = json.loads(V2.read_text())

    def test_draft_cannot_be_executed_without_explicit_approval(self):
        self.assertEqual(self.v2["status"], "awaiting_explicit_approval")
        self.assertIs(self.v2["execution_authorized"], False)
        self.assertIs(self.v2["retention_cleanup_authorized"], False)

    def test_v2_is_separate_and_preserves_pinned_v1_identity(self):
        self.assertNotEqual(self.v2["seed"], self.v0["seed"])
        self.assertEqual(self.v2["pinned"], self.v0["pinned"])
        self.assertEqual(self.v2["artifact_prefix"], "v2-")
        self.assertEqual(self.v2["baseline"]["protocol_version"], "V0.1")
        self.assertEqual(
            self.v2["baseline"]["raw_results_sha256"],
            "b55e89421e759c06a3be0a57a6030b7e12445662d01988efc566e2ed848b91f6",
        )

    def test_design_retains_pairing_blinding_and_every_observation(self):
        design = self.v2["campaign"]
        self.assertGreaterEqual(design["paired_blocks"], 20)
        self.assertEqual(design["runs_per_block"], 2)
        self.assertIs(design["blinded"], True)
        self.assertEqual(design["outlier_policy"], "none")
        self.assertEqual(
            design["exclusion_policy"],
            "documented_cluster_incident_only_repeat_complete_pair",
        )

    def test_phase_metrics_and_absolute_tail_targets_are_explicit(self):
        required = {
            "pod_to_scheduled_s",
            "pod_to_restore_start_s",
            "criu_restore_s",
            "cuda_restore_s",
            "ready_s",
            "http_200_s",
            "first_token_s",
            "token_after_restore_summary_s",
            "checkpoint_storage_read_bytes",
            "checkpoint_storage_read_throughput_bytes_s",
            "node_memory_available_bytes",
            "node_page_cache_bytes",
        }
        self.assertTrue(required.issubset(self.v2["required_metrics"]))
        targets = self.v2["decision_gates"]["optimized_go"]
        self.assertLessEqual(targets["first_token_median_s_max"], 15)
        self.assertLessEqual(targets["first_token_p95_s_max"], 25)
        self.assertLessEqual(targets["first_token_max_s_max"], 40)
        self.assertEqual(targets["restore_successes_required"], 20)
        self.assertEqual(targets["valid_responses_required"], 40)

    def test_supported_diagnosis_and_experimental_optimization_are_separate(self):
        stages = self.v2["stages"]
        self.assertEqual(stages[0]["name"], "V2-A-instrumented-supported-path")
        self.assertNotIn("GMS", stages[0]["allowed_changes"])
        self.assertEqual(stages[1]["name"], "V2-B-experimental-optimization")
        self.assertIs(stages[1]["separate_approval_required"], True)
        self.assertIn("GMS", stages[1]["candidate_changes"])
        self.assertIn("optimized CRIU", stages[1]["candidate_changes"])
        self.assertIn("never override a disabled safety feature gate", stages[1]["constraints"])

    def test_kv_cache_release_is_the_first_planned_optimization(self):
        kv = self.v2["kv_cache_release"]
        self.assertEqual(kv["stage"], "V2-B1")
        self.assertEqual(kv["status"], "planned_not_implemented")
        self.assertEqual(kv["baseline_kv_cache_gib"], 21.37)
        self.assertGreater(kv["baseline_checkpoint_share_percent"], 50)
        self.assertEqual(
            kv["mechanism"],
            "vLLM sleep level 1 after a fail-closed drain, followed by wake-up after restore",
        )
        self.assertIn("zero in-flight requests", kv["preconditions"])
        self.assertIn("checkpoint aborts if drain cannot be proven", kv["failure_behavior"])
        self.assertIn(
            "readiness remains false until KV cache reallocation and a valid inference probe complete",
            kv["restore_invariants"],
        )
        self.assertLessEqual(kv["acceptance"]["checkpoint_size_ratio_to_v1_max"], 0.60)
        self.assertGreaterEqual(kv["acceptance"]["gpu_memory_freed_gib_min"], 18)
        self.assertFalse(kv["control_plane"]["public_vllm_dev_api_allowed"])

    def test_kv_release_has_complete_measurement_and_safety_gates(self):
        required = {
            "checkpoint_size_bytes",
            "pages_12_size_bytes",
            "kv_prepare_s",
            "kv_wake_s",
            "gpu_memory_before_kv_release_mib",
            "gpu_memory_after_kv_release_mib",
            "gpu_memory_after_kv_wake_mib",
            "inflight_requests_before_checkpoint",
        }
        self.assertTrue(required.issubset(self.v2["required_metrics"]))
        tests = set(self.v2["kv_cache_release"]["required_tests"])
        self.assertIn("checkpoint is rejected with any in-flight request", tests)
        self.assertIn("two valid post-restore responses with an empty logical KV cache", tests)
        self.assertIn("control endpoint is unreachable outside the Pod", tests)

    def test_security_and_rollback_keep_v1_and_secrets_intact(self):
        security = set(self.v2["security_invariants"])
        self.assertIn("automountServiceAccountToken=false", security)
        self.assertIn("encrypted checkpoint storage", security)
        self.assertIn("no customer prompts or traffic", security)
        rollback = set(self.v2["rollback"])
        self.assertIn("do not modify or delete V0/I1/V1 artifacts", rollback)
        self.assertIn(
            "retain namespace PVC checkpoint image and keys until approved cleanup", rollback
        )


if __name__ == "__main__":
    unittest.main()
