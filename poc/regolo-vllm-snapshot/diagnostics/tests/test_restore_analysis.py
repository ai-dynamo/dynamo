import importlib.util
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
LIB = ROOT / "diagnostics/restore_analysis.py"


def load_library():
    spec = importlib.util.spec_from_file_location("restore_analysis", LIB)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def log_line(timestamp, message, payload):
    return f"{timestamp} INFO executor {message}\t{json.dumps(payload)}\n"


class DurationTests(unittest.TestCase):
    def test_parse_go_duration_supports_compound_and_subsecond_units(self):
        parse = load_library().parse_go_duration
        self.assertAlmostEqual(parse("1m44.917689427s"), 104.917689427)
        self.assertAlmostEqual(parse("93.042368ms"), 0.093042368)
        self.assertAlmostEqual(parse("2h3m4.5s"), 7384.5)

    def test_parse_go_duration_rejects_noncanonical_or_negative_values(self):
        parse = load_library().parse_go_duration
        for value in ("", "4", "-1s", "1s trailing", "nan"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse(value)


class RestoreAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.module = load_library()
        self.results = [
            {
                "run_id": "v2-01-01",
                "block": 1,
                "opaque_arm": "A",
                "pod_created_at": "2026-08-13T16:00:00Z",
                "ready_s": 100.0,
                "http_200_s": 101.0,
                "first_token_s": 102.0,
                "checkpoint_size_bytes": 40_000_000_000,
            },
            {
                "run_id": "v2-01-02",
                "block": 1,
                "opaque_arm": "B",
                "pod_created_at": "2026-08-13T16:03:00Z",
                "ready_s": 20.0,
                "http_200_s": 21.0,
                "first_token_s": 30.0,
                "checkpoint_size_bytes": None,
            },
            {
                "run_id": "v2-02-01",
                "block": 2,
                "opaque_arm": "B",
                "pod_created_at": "2026-08-13T16:05:00Z",
                "ready_s": 50.0,
                "http_200_s": 51.0,
                "first_token_s": 60.0,
                "checkpoint_size_bytes": None,
            },
        ]
        self.key = {"A": "cold", "B": "restore"}
        self.log = "".join(
            [
                log_line(
                    "2026-08-13T16:03:02.000000000Z",
                    "=== Starting external restore ===",
                    {"pod": "v2-01-02"},
                ),
                log_line(
                    "2026-08-13T16:03:22.000000000Z",
                    "Restore timing summary",
                    {
                        "pod": "dynamo-snapshot-poc/v2-01-02",
                        "restore": {
                            "duration": "20s",
                            "phases": {
                                "criu_restore_duration": "12s",
                                "cuda_duration": "7s",
                                "host_inspect_duration": "10ms",
                                "nsrestore_setup_duration": "990ms",
                            },
                        },
                    },
                ),
                log_line(
                    "2026-08-13T16:05:03.000000000Z",
                    "=== Starting external restore ===",
                    {"pod": "dynamo-snapshot-poc/v2-02-01"},
                ),
                log_line(
                    "2026-08-13T16:05:53.000000000Z",
                    "Restore timing summary",
                    {
                        "pod": "v2-02-01",
                        "restore": {
                            "duration": "50s",
                            "phases": {
                                "criu_restore_duration": "42s",
                                "cuda_duration": "7s",
                                "host_inspect_duration": "20ms",
                                "nsrestore_setup_duration": "980ms",
                            },
                        },
                    },
                ),
            ]
        )
        self.events = {
            "v2-01-02": {
                "items": [
                    {
                        "reason": "Scheduled",
                        "eventTime": "2026-08-13T16:03:00.250000Z",
                    }
                ]
            },
            "v2-02-01": {
                "items": [
                    {
                        "reason": "Scheduled",
                        "eventTime": "2026-08-13T16:05:00.500000Z",
                    }
                ]
            },
        }

    def test_analysis_attributes_tail_excess_and_preserves_every_restore(self):
        report = self.module.analyze(
            self.results, self.key, self.log, self.events, expected_restores=2
        )

        self.assertEqual([row["run_id"] for row in report["runs"]], ["v2-01-02", "v2-02-01"])
        self.assertEqual(report["summary"]["first_token_s"]["median"], 45.0)
        self.assertEqual(report["summary"]["criu_restore_s"]["median"], 27.0)
        self.assertEqual(report["tail"]["run_id"], "v2-02-01")
        self.assertEqual(report["tail"]["first_token_excess_over_median_s"], 15.0)
        self.assertEqual(report["tail"]["criu_excess_over_median_s"], 15.0)
        self.assertEqual(report["tail"]["criu_share_of_tail_excess"], 1.0)
        self.assertEqual(report["checkpoint_size_bytes"], 40_000_000_000)
        self.assertAlmostEqual(report["runs"][0]["effective_checkpoint_gb_per_s"], 10 / 3)
        self.assertEqual(report["runs"][0]["pod_created_at"], "2026-08-13T16:03:00Z")
        self.assertEqual(
            report["runs"][0]["restore_summary_at"], "2026-08-13T16:03:22.000000000Z"
        )
        self.assertEqual(report["runs"][0]["pod_to_scheduled_s"], 0.25)
        self.assertEqual(report["runs"][1]["pod_to_restore_start_s"], 3.0)
        self.assertEqual(report["runs"][1]["token_after_restore_summary_s"], 7.0)

    def test_analysis_fails_closed_for_missing_or_duplicate_timing_summaries(self):
        with self.assertRaisesRegex(ValueError, "missing timing summary"):
            self.module.analyze(
                self.results, self.key, self.log.splitlines()[0] + "\n", self.events, 2
            )

        duplicate = self.log + self.log.splitlines()[1] + "\n"
        with self.assertRaisesRegex(ValueError, "duplicate timing summary"):
            self.module.analyze(self.results, self.key, duplicate, self.events, 2)

        with self.assertRaisesRegex(ValueError, "missing Scheduled event"):
            self.module.analyze(self.results, self.key, self.log, {}, 2)

    def test_analysis_fails_closed_for_incomplete_metrics_or_phase_totals(self):
        incomplete = [dict(row) for row in self.results]
        incomplete[1]["ready_s"] = None
        with self.assertRaisesRegex(ValueError, "missing ready_s"):
            self.module.analyze(incomplete, self.key, self.log, self.events, 2)

        inconsistent_log = self.log.replace('"duration": "20s"', '"duration": "21s"')
        with self.assertRaisesRegex(ValueError, "phase durations do not sum"):
            self.module.analyze(self.results, self.key, inconsistent_log, self.events, 2)

    def test_cli_writes_deterministic_json_without_modifying_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            results = root / "results.jsonl"
            key = root / "key.json"
            agent_log = root / "agent.log"
            events = root / "events"
            output = root / "analysis.json"
            results.write_text("".join(json.dumps(row) + "\n" for row in self.results))
            key.write_text(json.dumps(self.key))
            agent_log.write_text(self.log)
            events.mkdir()
            for run_id, payload in self.events.items():
                (events / f"{run_id}.json").write_text(json.dumps(payload))
            before = {path: path.read_bytes() for path in (results, key, agent_log)}

            self.module.main(
                [
                    "--results",
                    str(results),
                    "--key",
                    str(key),
                    "--agent-log",
                    str(agent_log),
                    "--events-dir",
                    str(events),
                    "--expected-restores",
                    "2",
                    "--output",
                    str(output),
                ]
            )

            self.assertTrue(output.exists())
            self.assertEqual(json.loads(output.read_text())["tail"]["run_id"], "v2-02-01")
            self.assertEqual(before, {path: path.read_bytes() for path in before})


if __name__ == "__main__":
    unittest.main()
