# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("normalize_results.py")
SPEC = importlib.util.spec_from_file_location("normalize_results", SCRIPT)
assert SPEC and SPEC.loader
normalize_results = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(normalize_results)


class NormalizeResultsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.run_id = "run-20260706-abcd"
        self.aiperf_run = self.root / "aiperf" / self.run_id
        self.repetition = self.aiperf_run / "profile_runs" / "run_0001"
        self.repetition.mkdir(parents=True)
        (self.root / "validation").mkdir()

        parameters = {
            "workload": {
                "repetitions": 1,
                "num_requests": 2,
                "input_sequence_length": 4096,
                "output_sequence_length": 256,
            }
        }
        self.spec = {
            "status": "locked",
            "parameters": parameters,
            "fingerprint": normalize_results.canonical_fingerprint(parameters),
            "aiperf_spec": {
                "spec_id": "spec-20260706-19f1",
                "fingerprint": "aiperf-fingerprint",
            },
        }
        self._write_json(self.root / "spec.lock.json", self.spec)

        aggregate = {}
        for name, (unit, statistics) in normalize_results.AGGREGATE_METRICS.items():
            aggregate[name] = {"unit": unit}
            aggregate[name].update(
                {
                    statistic: float(index + 1)
                    for index, statistic in enumerate(statistics)
                }
            )
        self._write_json(self.repetition / "profile_export_aiperf.json", aggregate)

        records = [self._record("request-1"), self._record("request-2")]
        (self.repetition / "profile_export.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

        self.report_path = self.root / "validation" / f"{self.run_id}.json"
        self.report = {
            "valid": True,
            "run_root": str(self.root),
            "aiperf_run_dir": str(self.aiperf_run),
            "checks": [{"name": "all gates", "passed": True}],
            "repetitions": [
                {
                    "name": "run_0001",
                    "raw_record_count": 2,
                    "successful_transfer_count": 2,
                    "worker_pairs": [["prefill", "decode"]],
                }
            ],
        }
        self._write_json(self.report_path, self.report)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_json(self, path: Path, value: object) -> None:
        path.write_text(json.dumps(value), encoding="utf-8")

    def _record(
        self, request_id: str, isl: int = 4096, osl: int = 256
    ) -> dict[str, object]:
        return {
            "metadata": {"x_request_id": request_id},
            "metrics": {
                "time_to_first_token": {"value": 10.0},
                "request_latency": {"value": 20.0},
                "inter_token_latency": {"value": 1.0},
                "input_sequence_length": {"value": isl},
                "output_sequence_length": {"value": osl},
            },
        }

    def test_normalizes_valid_report_and_records_source_hashes(self) -> None:
        normalized, requests = normalize_results.normalize(
            self.root, self.aiperf_run, self.report_path
        )

        self.assertEqual(normalized["run_id"], self.run_id)
        self.assertEqual(normalized["spec_fingerprint"], self.spec["fingerprint"])
        self.assertEqual(len(normalized["repetitions"]), 1)
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]["ttft_ms"], 10.0)
        self.assertEqual(requests[0]["server_input_tokens"], 4096)
        self.assertEqual(requests[0]["server_output_tokens"], 256)
        self.assertEqual(len(normalized["source_artifacts"]), 2)

    def test_preserves_server_input_count_with_chat_template_overhead(self) -> None:
        records = [
            self._record("request-1", isl=4108),
            self._record("request-2", isl=4108),
        ]
        (self.repetition / "profile_export.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

        normalized, requests = normalize_results.normalize(
            self.root, self.aiperf_run, self.report_path
        )

        self.assertEqual(
            normalized["locked_workload"]["configured_input_sequence_length"], 4096
        )
        self.assertEqual(
            {request["server_input_tokens"] for request in requests}, {4108}
        )

    def test_csv_output_uses_lf_line_endings(self) -> None:
        normalized, requests = normalize_results.normalize(
            self.root, self.aiperf_run, self.report_path
        )
        json_output = self.root / "normalized.json"
        csv_output = self.root / "requests.csv"

        normalize_results.write_outputs(normalized, requests, json_output, csv_output)

        self.assertNotIn(b"\r\n", csv_output.read_bytes())

    def test_rejects_invalid_validation_report(self) -> None:
        self.report["valid"] = False
        self._write_json(self.report_path, self.report)

        with self.assertRaisesRegex(ValueError, "does not mark the run valid"):
            normalize_results.normalize(self.root, self.aiperf_run, self.report_path)

    def test_rejects_valid_flag_without_completed_checks(self) -> None:
        self.report["checks"] = []
        self._write_json(self.report_path, self.report)

        with self.assertRaisesRegex(ValueError, "failed or incomplete check"):
            normalize_results.normalize(self.root, self.aiperf_run, self.report_path)

    def test_rejects_validation_report_for_another_run(self) -> None:
        self.report["aiperf_run_dir"] = str(self.root / "aiperf" / "another-run")
        self._write_json(self.report_path, self.report)

        with self.assertRaisesRegex(ValueError, "AIPerf path does not match"):
            normalize_results.normalize(self.root, self.aiperf_run, self.report_path)

    def test_rejects_modified_locked_parameters(self) -> None:
        self.spec["parameters"]["workload"]["num_requests"] = 3
        self._write_json(self.root / "spec.lock.json", self.spec)

        with self.assertRaisesRegex(ValueError, "fingerprint does not match"):
            normalize_results.normalize(self.root, self.aiperf_run, self.report_path)

    def test_rejects_request_sequence_length_drift(self) -> None:
        records = [self._record("request-1"), self._record("request-2", osl=255)]
        (self.repetition / "profile_export.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "unexpected output length"):
            normalize_results.normalize(self.root, self.aiperf_run, self.report_path)


if __name__ == "__main__":
    unittest.main()
