import hashlib
import json
import pathlib
import stat
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from test_lane_contract import lane
from test_metric_gate import measurements
from _support import load_harness


V2_ROOT = pathlib.Path(__file__).resolve().parents[1]
V2CTL = V2_ROOT / "harness" / "v2ctl.py"


class V2CtlTests(unittest.TestCase):
    def run_ctl(self, *args):
        return subprocess.run(
            [sys.executable, str(V2CTL), *map(str, args)], text=True, capture_output=True
        )

    def write_json(self, directory, name, value):
        path = pathlib.Path(directory) / name
        path.write_text(json.dumps(value))
        return path

    def successful_gated_run(self, root):
        """Build a sealed, otherwise gate-passing ledger plus four raw files/run."""
        harness = load_harness()
        lane_path = self.write_json(root, "lane.json", lane())
        authorization = self.write_json(root, "auth.json", {"execution_authorized": True})
        output = root / "run"
        initialized = self.run_ctl(
            "init-run", "--lane", lane_path, "--authorization", authorization, "--output", output
        )
        self.assertEqual(initialized.returncode, 0, initialized.stderr)
        schedule = json.loads((output / "schedule.json").read_text())
        key = json.loads((output / "unblinding-key.json").read_text())
        artifact_dir = root / "artifacts"
        ledger = harness.ResultsLedger(output / "results.jsonl")
        for run in schedule:
            restore = key[run["opaque_arm"]] == "restore"
            record = harness.complete_metric_record(run["run_id"], measurements(restore), None)
            record.update(
                block=run["block"], sequence_in_block=run["sequence_in_block"], opaque_arm=run["opaque_arm"],
                restore_success=restore, valid_response=True,
                first_token_s=10.0 if restore else 100.0, gpu_memory_mib=100.0,
                excluded=False, error=None, lane_digest=lane()["digest"],
                schedule_digest=harness.schedule_digest(schedule),
            )
            raw = {
                "raw_events": (f"events/{run['run_id']}.json", b'{"items":[]}\n'),
                "raw_logs": (f"logs/{run['run_id']}.jsonl", b"agent log\n" if restore else b""),
                "raw_telemetry": (f"telemetry/{run['run_id']}.json", b'{"cpu":{}}\n'),
                "raw_response": (f"responses/{run['run_id']}.json", b'{"response":" 2"}\n'),
            }
            for stem, (ref, content) in raw.items():
                path = artifact_dir / ref
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                record[stem + "_ref"] = ref
                record[stem + "_sha256"] = hashlib.sha256(content).hexdigest()
            ledger.append(record)
        return lane_path, output / "schedule.json", output / "unblinding-key.json", output / "results.jsonl", artifact_dir

    def test_init_run_requires_separate_true_authorization_and_new_output(self):
        with tempfile.TemporaryDirectory() as directory:
            lane_path = self.write_json(directory, "lane.json", lane())
            false_auth = self.write_json(directory, "false.json", {"execution_authorized": False})
            missing_auth = self.write_json(directory, "missing.json", {})
            for authorization in (false_auth, missing_auth):
                with self.subTest(authorization=authorization.name):
                    result = self.run_ctl(
                        "init-run", "--lane", lane_path, "--authorization", authorization,
                        "--output", pathlib.Path(directory) / authorization.stem,
                    )
                    self.assertNotEqual(result.returncode, 0)
            existing = pathlib.Path(directory) / "existing"
            existing.mkdir()
            true_auth = self.write_json(directory, "true.json", {"execution_authorized": True})
            result = self.run_ctl(
                "init-run", "--lane", lane_path, "--authorization", true_auth, "--output", existing
            )
            self.assertNotEqual(result.returncode, 0)

    def test_init_run_creates_private_plan_key_and_empty_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            lane_path = self.write_json(directory, "lane.json", lane())
            authorization = self.write_json(directory, "auth.json", {"execution_authorized": True})
            output = pathlib.Path(directory) / "new-run"
            result = self.run_ctl(
                "init-run", "--lane", lane_path, "--authorization", authorization, "--output", output
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            schedule = output / "schedule.json"
            key = output / "unblinding-key.json"
            ledger = output / "results.jsonl"
            self.assertTrue(schedule.is_file())
            self.assertTrue(key.is_file())
            self.assertEqual(stat.S_IMODE(key.stat().st_mode), 0o600)
            self.assertEqual(ledger.read_bytes(), b"")
            self.assertNotIn("cold", schedule.read_text().lower())
            self.assertNotIn("restore", schedule.read_text().lower())

    def test_verify_ledger_returns_success_for_initialized_ledger_and_failure_when_tampered(self):
        with tempfile.TemporaryDirectory() as directory:
            lane_path = self.write_json(directory, "lane.json", lane())
            authorization = self.write_json(directory, "auth.json", {"execution_authorized": True})
            output = pathlib.Path(directory) / "run"
            initialized = self.run_ctl(
                "init-run", "--lane", lane_path, "--authorization", authorization, "--output", output
            )
            self.assertEqual(initialized.returncode, 0, initialized.stderr)
            ledger = output / "results.jsonl"
            self.assertEqual(self.run_ctl("verify-ledger", ledger).returncode, 0)
            ledger.write_text('{"forged":true}\n')
            self.assertNotEqual(self.run_ctl("verify-ledger", ledger).returncode, 0)

    def test_gate_rejects_lane_schedule_or_key_not_anchored_to_frozen_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            lane_path = self.write_json(directory, "lane.json", lane())
            authorization = self.write_json(directory, "auth.json", {"execution_authorized": True})
            output = root / "run"
            initialized = self.run_ctl(
                "init-run", "--lane", lane_path, "--authorization", authorization, "--output", output
            )
            self.assertEqual(initialized.returncode, 0, initialized.stderr)
            schedule = output / "schedule.json"
            key = output / "unblinding-key.json"
            ledger = output / "results.jsonl"
            forged = json.loads(schedule.read_text())
            forged[0]["opaque_arm"] = forged[1]["opaque_arm"]
            schedule.write_text(json.dumps(forged))
            rejected = self.run_ctl(
                "gate", "--lane", lane_path, "--schedule", schedule,
                "--key", key, "--ledger", ledger,
            )
            self.assertNotEqual(rejected.returncode, 0)

    def test_gate_requires_artifact_dir_and_accepts_hash_bound_success_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            lane_path, schedule, key, ledger, artifact_dir = self.successful_gated_run(root)
            base = ("gate", "--lane", lane_path, "--schedule", schedule, "--key", key, "--ledger", ledger)
            self.assertNotEqual(self.run_ctl(*base).returncode, 0)
            accepted = self.run_ctl(*base, "--artifact-dir", artifact_dir)
            self.assertEqual(accepted.returncode, 0, accepted.stderr)

    def test_gate_rejects_missing_digest_mismatched_symlink_or_escape_success_evidence(self):
        for mutation in ("missing", "digest", "symlink", "escape"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = pathlib.Path(directory)
                lane_path, schedule, key, ledger_path, artifact_dir = self.successful_gated_run(root)
                rows = load_harness().ResultsLedger(ledger_path).read()
                target = artifact_dir / rows[0]["raw_events_ref"]
                if mutation == "missing":
                    target.unlink()
                elif mutation == "digest":
                    target.write_bytes(b"tampered")
                elif mutation == "symlink":
                    outside = root / "outside.json"
                    outside.write_bytes(b"outside")
                    target.unlink()
                    target.symlink_to(outside)
                else:
                    rows[0]["raw_events_ref"] = "../outside.json"
                    outside = root / "outside.json"
                    outside.write_bytes(b"outside")
                    rows[0]["raw_events_sha256"] = hashlib.sha256(outside.read_bytes()).hexdigest()
                    ledger_path.unlink()
                    rebuilt = load_harness().ResultsLedger(ledger_path)
                    for row in rows:
                        rebuilt.append({key: value for key, value in row.items()
                                        if key not in {"sequence", "previous_record_digest", "record_digest"}})
                rejected = self.run_ctl(
                    "gate", "--lane", lane_path, "--schedule", schedule, "--key", key, "--ledger", ledger_path,
                    "--artifact-dir", artifact_dir,
                )
                self.assertNotEqual(rejected.returncode, 0)

    def test_collect_host_uses_only_injected_proc_paths_and_explicit_sizes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            inputs = {
                "meminfo": "MemAvailable: 400 kB\nCached: 300 kB\nSReclaimable: 50 kB\nShmem: 20 kB\n",
                "psi-cpu": "some avg10=0.1 avg60=0.2 avg300=0.3 total=4\n",
                "psi-io": "some avg10=0.1 avg60=0.2 avg300=0.3 total=4\n",
                "psi-memory": "some avg10=0.1 avg60=0.2 avg300=0.3 total=4\n",
                "io.stat": "253:0 rbytes=10 wbytes=20\n",
                "diskstats": (
                    "8 0 sda 1 2 3 4 5 6 7 8 9 10 11\n"
                    "7 6 loop6 1 2 3 4 5 6 7 8 9 10 11\n"
                    "253 0 dm-0 1 2 3 4 5 6 7 8 9 10 11\n"
                ),
                "proc-stat-before": "cpu 100 0 100 800 0 0 0 0 0 0\n",
                "proc-stat-after": "cpu 150 0 150 900 0 0 0 0 0 0\n",
                "gpu-memory": "100\n200.5\n",
            }
            paths = {}
            for name, body in inputs.items():
                paths[name] = root / name
                paths[name].write_text(body)
            checkpoint = root / "checkpoint"
            pages = root / "pages"
            checkpoint.mkdir()
            pages.mkdir()
            (checkpoint / "image").write_bytes(b"1234")
            (pages / "page").write_bytes(b"12")
            result = self.run_ctl(
                "collect-host", "--meminfo", paths["meminfo"], "--psi-cpu", paths["psi-cpu"],
                "--psi-io", paths["psi-io"], "--psi-memory", paths["psi-memory"],
                "--io-stat", paths["io.stat"], "--diskstats", paths["diskstats"],
                "--proc-stat-before", paths["proc-stat-before"],
                "--proc-stat-after", paths["proc-stat-after"],
                "--gpu-memory", paths["gpu-memory"],
                "--size", f"checkpoint={checkpoint}", "--size", f"pages={pages}",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            collected = json.loads(result.stdout)
            self.assertEqual(collected["meminfo"]["mem_available_bytes"], 409600)
            self.assertEqual(set(collected["diskstats"]), {"dm-0", "loop6", "sda"})
            self.assertEqual(collected["sizes"], {"checkpoint": 4, "pages": 2})
            self.assertEqual(collected["node_cpu_utilization"], 0.5)
            self.assertEqual(collected["gpu_memory_mib"], 300.5)

    def test_append_record_and_cold_advise_are_explicit_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            ledger = root / "results.jsonl"
            ledger.touch(mode=0o600)
            record = self.write_json(directory, "record.json", {"run_id": "v2-01-01", "status": "failure"})
            appended = self.run_ctl("append-record", "--ledger", ledger, "--record", record)
            self.assertEqual(appended.returncode, 0, appended.stderr)
            self.assertEqual(json.loads(appended.stdout)["sequence"], 1)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            candidate = checkpoint / "pages.img"
            candidate.write_bytes(b"pages")
            advised = self.run_ctl(
                "cold-advise", "--allow-root", checkpoint, "--file", candidate
            )
            self.assertEqual(advised.returncode, 0, advised.stderr)
            outside = root / "outside.img"
            outside.write_bytes(b"outside")
            rejected = self.run_ctl(
                "cold-advise", "--allow-root", checkpoint, "--file", outside
            )
            self.assertNotEqual(rejected.returncode, 0)


if __name__ == "__main__":
    unittest.main()
