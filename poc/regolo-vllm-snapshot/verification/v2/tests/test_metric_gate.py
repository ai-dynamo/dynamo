import hashlib
import pathlib
import tempfile
import unittest
from copy import deepcopy

from _support import load_harness
from test_lane_contract import lane


REQUIRED_METRICS = {
    "pod_to_scheduled_s", "pod_to_restore_start_s", "criu_restore_s", "cuda_restore_s",
    "ready_s", "http_200_s", "first_token_s", "cgroup_io_stat", "diskstats",
    "node_page_cache_bytes", "node_memory_available_bytes", "psi_cpu", "psi_io",
    "psi_memory", "node_cpu_utilization", "gpu_memory_mib", "checkpoint_size_bytes",
    "pages_12_size_bytes", "rootfs_size_bytes", "metadata_size_bytes", "prepare_s",
    "sleep_s", "wake_s", "admission_closed", "harness_inflight", "vllm_running",
    "vllm_waiting", "tokens_per_second", "token_after_restore_summary_s",
    "checkpoint_storage_read_bytes", "checkpoint_storage_read_throughput_bytes_s",
    "node_page_cache_delta_bytes", "node_memory_available_delta_bytes",
}

RESTORE_ONLY = {
    "pod_to_restore_start_s", "criu_restore_s", "cuda_restore_s", "prepare_s",
    "sleep_s", "wake_s", "token_after_restore_summary_s", "checkpoint_storage_read_bytes",
    "checkpoint_storage_read_throughput_bytes_s",
}

V2_B1_ONLY = {"prepare_s", "sleep_s", "wake_s"}


def measurements(restore=True):
    value = {field: 1 for field in REQUIRED_METRICS}
    if not restore:
        value.update({field: None for field in RESTORE_ONLY})
    value.update(
        cgroup_io_stat={"253:0": {"rbytes": 1, "wbytes": 1}},
        diskstats={"dm-0": {"sectors_read": 1}, "loop6": {}, "sda": {}},
        psi_cpu={"some": {"avg10": 0.0, "total": 0}},
        psi_io={"some": {"avg10": 0.0, "total": 0}},
        psi_memory={"some": {"avg10": 0.0, "total": 0}},
        admission_closed=True, harness_inflight=0, vllm_running=0, vllm_waiting=0,
        tokens_per_second=100.0 if not restore else 95.0,
    )
    return value


class MetricRecordAndGateTests(unittest.TestCase):
    def setUp(self):
        self.harness = load_harness()
        self.lane = lane()
        self.plan = self.harness.make_paired_blinded_plan(self.lane)
        self.schedule = self.plan["schedule"]
        self.key = self.plan["unblinding_key"]

    def complete_records(self, artifact_dir=None):
        records = []
        for run in self.schedule:
            restore = self.key[run["opaque_arm"]] == "restore"
            record = self.harness.complete_metric_record(
                run_id=run["run_id"], metrics=measurements(restore), failure_reason=None
            )
            record.update(
                block=run["block"], sequence_in_block=run["sequence_in_block"],
                opaque_arm=run["opaque_arm"], restore_success=restore, valid_response=True,
                first_token_s=10.0 if restore else 100.0, gpu_memory_mib=100.0,
                excluded=False, error=None, lane_digest=self.lane["digest"],
                schedule_digest=self.harness.schedule_digest(self.schedule),
                execution_digest="d" * 64,
            )
            if artifact_dir is not None:
                artifact_dir = pathlib.Path(artifact_dir)
                files = {
                    "raw_events": (f"events/{run['run_id']}.json", b'{"items":[]}\n'),
                    "raw_logs": (f"logs/{run['run_id']}.jsonl", b"" if not restore else b"agent log\n"),
                    "raw_telemetry": (f"telemetry/{run['run_id']}.json", b'{"cpu":{}}\n'),
                    "raw_response": (f"responses/{run['run_id']}.json", b'{"response":" 2"}\n'),
                }
                for stem, (ref, content) in files.items():
                    path = artifact_dir / ref
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(content)
                    record[stem + "_ref"] = ref
                    record[stem + "_sha256"] = hashlib.sha256(content).hexdigest()
            records.append(record)
        return records

    def evaluate(self, records, optimized=False):
        function = self.harness.evaluate_optimized_gate if optimized else self.harness.evaluate_diagnosis_gate
        return function(
            records, schedule=self.schedule, lane_digest=self.lane["digest"],
            unblinding_key=self.key,
        )

    def test_cold_records_allow_only_restore_specific_metrics_to_be_null(self):
        success = self.harness.complete_metric_record(
            run_id="v2-01-01", metrics=measurements(False), failure_reason=None
        )
        self.assertTrue(all(success[field] is None for field in RESTORE_ONLY))
        candidate = measurements(True)
        candidate["criu_restore_s"] = None
        with self.assertRaises(ValueError):
            self.harness.complete_metric_record(
                run_id="v2-01-02", metrics=candidate, failure_reason=None
            )

    def test_v2_a_restore_keeps_v2_b1_kv_release_metrics_inapplicable(self):
        candidate = measurements(True)
        candidate.update({field: None for field in V2_B1_ONLY})
        record = self.harness.complete_metric_record(
            run_id="v2-01-02", metrics=candidate, failure_reason=None
        )
        self.assertTrue(all(record[field] is None for field in V2_B1_ONLY))

    def test_restore_storage_and_memory_delta_metrics_are_required_and_gate_rejects_missing_storage_evidence(self):
        new_fields = {
            "token_after_restore_summary_s": 1.0,
            "checkpoint_storage_read_bytes": 4096,
            "checkpoint_storage_read_throughput_bytes_s": 1024.0,
            "node_page_cache_delta_bytes": -4096,
            "node_memory_available_delta_bytes": 8192,
        }
        restore = measurements(True) | new_fields
        record = self.harness.complete_metric_record("v2-01-01", restore, None)
        self.assertTrue(all(record[field] is not None for field in new_fields))
        cold = measurements(False) | {
            **{field: None for field in (
                "token_after_restore_summary_s", "checkpoint_storage_read_bytes",
                "checkpoint_storage_read_throughput_bytes_s",
            )},
            "node_page_cache_delta_bytes": -1,
            "node_memory_available_delta_bytes": 1,
        }
        self.assertTrue(all(self.harness.complete_metric_record("v2-01-02", cold, None)[field] is None for field in (
            "token_after_restore_summary_s", "checkpoint_storage_read_bytes",
            "checkpoint_storage_read_throughput_bytes_s",
        )))
        records = self.complete_records()
        self.assertTrue(self.evaluate(records)["passed"])
        missing = deepcopy(records)
        next(row for row in missing if self.key[row["opaque_arm"]] == "restore")["checkpoint_storage_read_bytes"] = None
        self.assertFalse(self.evaluate(missing)["passed"])

    def test_restore_storage_read_bytes_and_throughput_are_strictly_positive(self):
        for field in ("checkpoint_storage_read_bytes", "checkpoint_storage_read_throughput_bytes_s"):
            with self.subTest(field=field):
                candidate = measurements(True)
                candidate[field] = 0
                with self.assertRaises(ValueError):
                    self.harness.complete_metric_record("v2-01-01", candidate, None)

    def test_diagnosis_gate_binds_every_record_to_frozen_lane_and_schedule(self):
        records = self.complete_records()
        self.assertTrue(self.evaluate(records)["passed"])
        cases = {
            "lane": lambda rows: rows[0].__setitem__("lane_digest", "0" * 64),
            "schedule": lambda rows: rows[0].__setitem__("schedule_digest", "0" * 64),
            "arm": lambda rows: rows[0].__setitem__("opaque_arm", rows[1]["opaque_arm"]),
            "block": lambda rows: rows[0].__setitem__("block", 20),
            "sequence": lambda rows: rows[0].__setitem__("sequence_in_block", 2),
        }
        for name, mutate in cases.items():
            candidate = deepcopy(records)
            mutate(candidate)
            with self.subTest(name=name):
                self.assertFalse(self.evaluate(candidate)["passed"])

    def test_diagnosis_and_optimized_go_are_separate(self):
        records = self.complete_records()
        restore = [row for row in records if self.key[row["opaque_arm"]] == "restore"]
        for row in restore:
            row["first_token_s"] = 30.0
            row["criu_restore_s"] = 15.0
        self.assertTrue(self.evaluate(records)["passed"])
        self.assertFalse(self.evaluate(records, optimized=True)["passed"])

    def test_optimized_gate_uses_restore_only_and_derived_paired_throughput(self):
        records = self.complete_records()
        result = self.evaluate(records, optimized=True)
        self.assertTrue(result["passed"])
        self.assertEqual(result["candidate"]["first_token_s"]["median"], 10.0)
        restore = [row for row in records if self.key[row["opaque_arm"]] == "restore"]
        restore[0]["tokens_per_second"] = 89.0
        self.assertFalse(self.evaluate(records, optimized=True)["passed"])

    def test_diagnosis_rejects_failures_exclusions_and_invalid_drain_evidence(self):
        records = self.complete_records()
        cases = {
            "restore": lambda rows: next(row for row in rows if self.key[row["opaque_arm"]] == "restore").__setitem__("restore_success", False),
            "response": lambda rows: rows[0].__setitem__("valid_response", False),
            "admission": lambda rows: rows[0].__setitem__("admission_closed", False),
            "inflight": lambda rows: rows[0].__setitem__("harness_inflight", 1),
            "running": lambda rows: rows[0].__setitem__("vllm_running", 1),
            "waiting": lambda rows: rows[0].__setitem__("vllm_waiting", 1),
            "error": lambda rows: rows[0].__setitem__("error", "collector failed"),
            "exclusion": lambda rows: rows[0].__setitem__("excluded", True),
        }
        for name, mutate in cases.items():
            candidate = deepcopy(records)
            mutate(candidate)
            with self.subTest(name=name):
                self.assertFalse(self.evaluate(candidate)["passed"])

    def test_gate_binds_success_raw_evidence_to_artifact_dir_but_exempts_failure_rows(self):
        for mutation in ("tamper", "missing", "escape"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                artifact_dir = pathlib.Path(directory) / "artifacts"
                records = self.complete_records(artifact_dir)
                target = artifact_dir / records[0]["raw_events_ref"]
                if mutation == "tamper":
                    target.write_bytes(b"tampered")
                elif mutation == "missing":
                    target.unlink()
                else:
                    records[0]["raw_events_ref"] = "../outside.json"
                    outside = pathlib.Path(directory) / "outside.json"
                    outside.write_bytes(b"outside")
                    records[0]["raw_events_sha256"] = hashlib.sha256(outside.read_bytes()).hexdigest()
                with self.assertRaises(ValueError):
                    self.harness.evaluate_diagnosis_gate(
                        records, schedule=self.schedule, lane_digest=self.lane["digest"],
                        unblinding_key=self.key, artifact_dir=artifact_dir,
                    )

        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = pathlib.Path(directory) / "artifacts"
            records = self.complete_records(artifact_dir)
            failed = records[0]
            failed["failure_reason"] = "collector failed"
            for stem in ("raw_events", "raw_logs", "raw_telemetry", "raw_response"):
                (artifact_dir / failed.pop(stem + "_ref")).unlink()
                failed.pop(stem + "_sha256")
            result = self.harness.evaluate_diagnosis_gate(
                records, schedule=self.schedule, lane_digest=self.lane["digest"],
                unblinding_key=self.key, artifact_dir=artifact_dir,
            )
            self.assertFalse(result["passed"])

    def test_gate_requires_one_execution_digest_for_every_paired_record(self):
        records = self.complete_records()
        self.assertTrue(self.evaluate(records)["passed"])
        for index in (0, len(records) - 1):
            with self.subTest(missing=index):
                candidate = deepcopy(records)
                candidate[index].pop("execution_digest")
                self.assertFalse(self.evaluate(candidate)["passed"])
        records[0]["execution_digest"] = "e" * 64
        self.assertFalse(self.evaluate(records)["passed"])


if __name__ == "__main__":
    unittest.main()
