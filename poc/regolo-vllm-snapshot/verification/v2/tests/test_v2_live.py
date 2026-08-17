"""Black-box contract for the dependency-injected V2 live runner.

This deliberately exercises no Kubernetes client, network, credential, or real
cache operation.  The runner receives a callable command transport and an
observer callable; production may wire those to kubectl/snapshotctl and the
phase collector, while these tests retain the exact commands and records.
"""

import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from _support import load_harness


V2_ROOT = pathlib.Path(__file__).resolve().parents[1]
LIVE = V2_ROOT / "harness" / "v2_live.py"


def load_live():
    if not LIVE.is_file():
        raise FileNotFoundError(f"missing required V2 live runner: {LIVE}")
    spec = importlib.util.spec_from_file_location("v2_live_under_test", LIVE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {LIVE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Commands:
    """A stdlib-only command transport which never executes a command."""

    def __init__(self, timeline=None):
        self.calls = []
        self.timeline = timeline

    def __call__(self, command):
        call = tuple(map(str, command))
        self.calls.append(call)
        if self.timeline is not None:
            self.timeline.append(("command", call))
        return subprocess.CompletedProcess(command, 0, "ok", "")


class PodLifecycleCommands(Commands):
    """Frozen Pod-only cleanup transport: present once, then absent after delete."""

    def __init__(self, *, replacement_uid=False):
        super().__init__()
        self.get_count = 0
        self.replacement_uid = replacement_uid

    def __call__(self, command):
        call = tuple(map(str, command))
        self.calls.append(call)
        if call[:5] == ("kubectl", "-n", "v2-live-test", "get", "pod"):
            self.get_count += 1
            if self.get_count in {2, 3}:
                return subprocess.CompletedProcess(command, 0, '{"metadata":{"uid":"uid-current-v2-pod"}}', "")
            if self.get_count in {4, 5}:
                uid = "uid-replaced" if self.replacement_uid else "uid-current-v2-pod"
                return subprocess.CompletedProcess(
                    command, 0, '{"metadata":{"uid":"%s","deletionTimestamp":"2026-08-14T00:00:10Z"}}' % uid, ""
                )
            return subprocess.CompletedProcess(command, 1, "", "NotFound")
        return subprocess.CompletedProcess(command, 0, "ok", "")


class PreparedFailureCollector:
    """Records lifecycle ordering without needing a Pod or network."""

    def __init__(self, timeline, pod_uid=None):
        self.timeline = timeline
        self.pod_uid = pod_uid

    def prepare(self, run, mode, pod_name):
        self.timeline.append(("prepare", run["run_id"], mode, pod_name))

    def __call__(self, run, mode, pod_name):
        self.timeline.append(("collect", run["run_id"], mode, pod_name))
        return {
            "metrics": {}, "failure_reason": "stop after lifecycle", "failure_stage": "collector",
            "pod_uid": self.pod_uid or "offline-" + pod_name,
        }


class V2LiveRunnerTests(unittest.TestCase):
    def setUp(self):
        self.harness = load_harness()
        self.live = load_live()
        self.lane = json.loads((V2_ROOT / "lane.json").read_text())
        self.plan = self.harness.make_paired_blinded_plan(self.lane)

    def metrics(self, restore):
        values = {field: 1 for field in self.harness.REQUIRED_METRICS}
        if not restore:
            for field in (
                "pod_to_restore_start_s", "criu_restore_s", "cuda_restore_s",
                "prepare_s", "sleep_s", "wake_s",
                "token_after_restore_summary_s", "checkpoint_storage_read_bytes",
                "checkpoint_storage_read_throughput_bytes_s",
            ):
                values[field] = None
        # V2-A does not perform the V2-B1 KV-cache release sequence.  These
        # phase measurements are deliberately inapplicable even on restore.
        values.update(prepare_s=None, sleep_s=None, wake_s=None)
        values.update(
            cgroup_io_stat={"253:0": {"rbytes": 1, "wbytes": 1}},
            diskstats={"dm-0": {}, "loop6": {}, "sda": {}},
            psi_cpu={"some": {"avg10": 0.0, "total": 0}},
            psi_io={"some": {"avg10": 0.0, "total": 0}},
            psi_memory={"some": {"avg10": 0.0, "total": 0}},
            admission_closed=True,
            harness_inflight=0,
            vllm_running=0,
            vllm_waiting=0,
            tokens_per_second=100.0,
        )
        return values

    def campaign(self):
        identity = self.lane["identity"]
        return {
            "namespace": "v2-live-test",
            "node": identity["node"],
            "snapshotctl": "/opt/pinned/snapshotctl",
            "snapshotctl_sha256": "a" * 64,
            "checkpoint": {
                "checkpoint_id": "h-" + identity["compatibility_hash"][:61],
                "compatibility_hash": identity["compatibility_hash"],
            },
        }

    def observer(self, run, mode, pod_name):
        evidence_root = getattr(self, "_evidence_root", None)
        if evidence_root is not None:
            payloads = {
                "raw_events_ref": ("events", ".json", b'{"items":[]}\n'),
                "raw_logs_ref": ("logs", ".jsonl", b"agent log\n"),
                "raw_telemetry_ref": ("telemetry", ".json", b'{"cpu":{}}\n'),
                "raw_response_ref": ("responses", ".json", b'{"response":" 2"}\n'),
            }
            for field, (directory, suffix, content) in payloads.items():
                path = evidence_root / directory / (run["run_id"] + suffix)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
        return {
            "metrics": self.metrics(mode == "restore"),
            "pod_uid": "offline-" + pod_name,
            "valid_response": True,
            "restore_success": mode == "restore",
            "raw_events_ref": f"events/{run['run_id']}.json",
            "raw_logs_ref": f"logs/{run['run_id']}.jsonl",
            "raw_telemetry_ref": f"telemetry/{run['run_id']}.json",
            "raw_response_ref": f"responses/{run['run_id']}.json",
        }

    @staticmethod
    def raw_evidence(artifact_dir, run_id, *, cold=False):
        artifact_dir = pathlib.Path(artifact_dir)
        files = {
            "raw_events": (f"events/{run_id}.json", b'{"items":[]}\n'),
            "raw_logs": (f"logs/{run_id}.jsonl", b"" if cold else b"agent log\n"),
            "raw_telemetry": (f"telemetry/{run_id}.json", b'{"cpu":{}}\n'),
            "raw_response": (f"responses/{run_id}.json", b'{"response":" 2"}\n'),
        }
        output = {}
        for stem, (ref, content) in files.items():
            path = artifact_dir / ref
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            output[stem + "_ref"] = ref
            output[stem + "_sha256"] = hashlib.sha256(content).hexdigest()
        return output

    def make_runner(self, root, **overrides):
        root = pathlib.Path(root)
        root.mkdir(parents=True, exist_ok=True)
        schedule_path, key_path = self.harness.seal_plan(self.plan, pathlib.Path(root) / "plan")
        lane_path = pathlib.Path(root) / "lane.json"
        lane_path.write_text(json.dumps(self.lane, sort_keys=True, separators=(",", ":")))
        snapshotctl = root / "snapshotctl"
        snapshotctl.write_bytes(b"pinned snapshotctl fixture\n")
        authorization_path = root / "authorization.json"
        authorization_path.write_text('{"execution_authorized":true}')
        campaign = self.campaign()
        campaign["snapshotctl"] = str(snapshotctl)
        campaign["snapshotctl_sha256"] = hashlib.sha256(snapshotctl.read_bytes()).hexdigest()
        attestation_path = root / "checkpoint-attestation.json"
        attestation_path.write_bytes(b'{"checkpoint":"frozen-v2"}\n')
        campaign["checkpoint"].update(
            attestation_path=str(attestation_path),
            attestation_sha256=hashlib.sha256(attestation_path.read_bytes()).hexdigest(),
        )
        def checkpoint_validator(path, expected_sha256):
            return (
                pathlib.Path(path).is_file()
                and hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest() == expected_sha256
            )

        def cluster_preflight():
            return True

        arguments = {
            "lane_path": lane_path,
            "authorization_path": authorization_path,
            "schedule_path": schedule_path,
            "key_path": key_path,
            "ledger_path": pathlib.Path(root) / "results.jsonl",
            "campaign": campaign,
            "command_runner": Commands(),
            "collector": self.observer,
            "artifact_dir": pathlib.Path(root) / "artifacts",
            "checkpoint_files": (pathlib.Path(root) / "checkpoint" / "pages.img",),
            "checkpoint_validator": checkpoint_validator,
            "cluster_preflight": cluster_preflight,
            "execution_digest": "d" * 64,
        }
        arguments.update(overrides)
        self._evidence_root = pathlib.Path(arguments["artifact_dir"])
        return self.live.LiveRunner(**arguments), arguments

    def test_authorization_must_be_the_exact_true_object_before_any_command(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, arguments = self.make_runner(directory)
            arguments["authorization_path"].write_text('{"execution_authorized":false}')
            with self.assertRaises(ValueError):
                runner.run(limit=1)
            self.assertEqual(arguments["command_runner"].calls, [])

    def test_execution_digest_is_mandatory_persisted_and_rechecked_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, arguments = self.make_runner(directory)
            missing = dict(arguments)
            missing.pop("execution_digest")
            with self.assertRaises(ValueError):
                self.live.LiveRunner(**missing).run(limit=0)
            digest = arguments["execution_digest"]
            runner.run(limit=1)
            self.assertEqual(self.harness.ResultsLedger(arguments["ledger_path"]).read()[0]["execution_digest"], digest)
            resumed = dict(arguments, command_runner=Commands(), execution_digest="e" * 64)
            with self.assertRaises(ValueError):
                self.live.LiveRunner(**resumed).run(limit=1)
            self.assertEqual(resumed["command_runner"].calls, [])

    def test_snapshotctl_file_digest_is_verified_before_any_command(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, arguments = self.make_runner(directory)
            arguments["campaign"]["snapshotctl_sha256"] = "0" * 64
            runner, arguments = self.make_runner(
                pathlib.Path(directory) / "mismatch", campaign=arguments["campaign"]
            )
            with self.assertRaises(ValueError):
                runner.run(limit=1)
            self.assertEqual(arguments["command_runner"].calls, [])

    def test_manifest_preserves_frozen_runtime_and_mode_specific_start_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, _ = self.make_runner(directory)
            run = self.plan["schedule"][0]
            schedule_digest = self.harness.schedule_digest(self.plan["schedule"])
            frozen_command = self.lane["identity"]["command"]
            frozen_args = self.lane["identity"]["args"]
            for mode in ("cold", "restore"):
                with self.subTest(mode=mode):
                    _, manifest = runner._manifest(run, mode, self.lane, schedule_digest)
                    self.assertTrue(all(len(str(value)) <= 63 for value in manifest["metadata"]["labels"].values()))
                    self.assertEqual(manifest["metadata"]["annotations"]["poc.regolo.ai/lane-digest"], self.lane["digest"])
                    spec = manifest["spec"]
                    container = spec["containers"][0]
                    self.assertIs(spec["automountServiceAccountToken"], False)
                    self.assertEqual(manifest["metadata"]["labels"]["app.kubernetes.io/name"], "vllm-snapshot-poc")
                    self.assertEqual(spec["imagePullSecrets"], [])
                    self.assertEqual(spec["runtimeClassName"], "nvidia")
                    self.assertEqual(
                        container["readinessProbe"],
                        {"httpGet": {"path": "/health", "port": 8000}, "periodSeconds": 1, "failureThreshold": 1800},
                    )
                    self.assertEqual(container["ports"], [{"name": "http", "containerPort": 8000}])
                    volumes = {volume["name"]: volume for volume in spec["volumes"]}
                    self.assertEqual(volumes["snapshot-control"], {"name": "snapshot-control", "emptyDir": {}})
                    self.assertEqual(
                        volumes["model-cache"],
                        {"name": "model-cache", "hostPath": {"path": "/var/lib/regolo-vllm-poc/hf-cache", "type": "Directory"}},
                    )
                    self.assertEqual(
                        {mount["name"]: mount["mountPath"] for mount in container["volumeMounts"]},
                        {"snapshot-control": "/snapshot-control", "model-cache": "/root/.cache/huggingface"},
                    )
                    if mode == "cold":
                        self.assertEqual(container["command"], frozen_command)
                        self.assertEqual(container["args"], frozen_args)
                        self.assertEqual(container.get("env", []), [])
                    else:
                        self.assertEqual(container["command"], ["/usr/local/bin/snapshot-entrypoint", "--"])
                        self.assertEqual(container["args"], frozen_command + frozen_args)
                        self.assertEqual(
                            {entry["name"]: entry["value"] for entry in container["env"]},
                            {
                                "SNAPSHOT_READY_URL": "http://127.0.0.1:8000/health",
                                "DYN_SNAPSHOT_CONTROL_DIR": "/snapshot-control",
                                "DYN_SNAPSHOT_RESTORE_STANDBY": "0",
                            },
                        )

    def test_collector_prepare_precedes_restore_advice_and_start(self):
        with tempfile.TemporaryDirectory() as directory:
            timeline = []
            commands = Commands(timeline)
            collector = PreparedFailureCollector(timeline)

            def fadvise(paths):
                timeline.append(("fadvise", tuple(paths)))

            runner, arguments = self.make_runner(
                directory, command_runner=commands, collector=collector, fadvise=fadvise
            )
            with self.assertRaises(RuntimeError):
                runner.run(limit=1)
            prepare_index = next(index for index, entry in enumerate(timeline) if entry[0] == "prepare")
            fadvise_index = next(index for index, entry in enumerate(timeline) if entry[0] == "fadvise")
            restore_index = next(
                index for index, entry in enumerate(timeline)
                if entry[0] == "command" and entry[1][0] == arguments["campaign"]["snapshotctl"]
            )
            self.assertLess(prepare_index, fadvise_index)
            self.assertLess(fadvise_index, restore_index)

    def test_preflight_refuses_an_existing_non_owned_pod_before_create_or_restore(self):
        class ExistingPod(Commands):
            def __call__(self, command):
                call = tuple(map(str, command))
                self.calls.append(call)
                if call[:5] == ("kubectl", "-n", "v2-live-test", "get", "pod"):
                    return subprocess.CompletedProcess(command, 0, '{"metadata":{"uid":"foreign"},"labels":{}}', "")
                return subprocess.CompletedProcess(command, 0, "ok", "")

        with tempfile.TemporaryDirectory() as directory:
            commands = ExistingPod()
            runner, _ = self.make_runner(directory, command_runner=commands)
            with self.assertRaises(ValueError):
                runner.run(limit=1)
            self.assertTrue(any(call[3:5] == ("get", "pod") for call in commands.calls))
            self.assertFalse(any("create" in call or "restore" in call for call in commands.calls))

    def test_cleanup_is_bound_to_current_pod_uid_and_proves_final_absence(self):
        with tempfile.TemporaryDirectory() as directory:
            commands = PodLifecycleCommands()
            collector = PreparedFailureCollector([], pod_uid="uid-current-v2-pod")
            runner, arguments = self.make_runner(directory, command_runner=commands, collector=collector)
            with self.assertRaises(RuntimeError):
                runner.run(limit=1)
            self.assertEqual(commands.get_count, 6)
            row = self.harness.ResultsLedger(arguments["ledger_path"]).read()[0]
            delete_calls = [call for call in commands.calls if "--raw" in call]
            self.assertEqual(len(delete_calls), 1)
            member_uri = next(value for value in delete_calls[0] if value.startswith("/api/v1/namespaces/"))
            self.assertEqual(member_uri, f"/api/v1/namespaces/v2-live-test/pods/{row['pod_name']}")
            self.assertIn("-f", delete_calls[0])
            self.assertNotIn("selector", delete_calls[0])
            options_path = pathlib.Path(delete_calls[0][delete_calls[0].index("-f") + 1])
            self.assertEqual(json.loads(options_path.read_text()), {"preconditions": {"uid": "uid-current-v2-pod"}})
            self.assertFalse(any(call[:3] == ("kubectl", "delete", "namespace") for call in commands.calls))
            self.assertEqual(row["pod_uid"], "uid-current-v2-pod")

        with tempfile.TemporaryDirectory() as directory:
            commands = PodLifecycleCommands(replacement_uid=True)
            collector = PreparedFailureCollector([], pod_uid="uid-current-v2-pod")
            runner, _ = self.make_runner(directory, command_runner=commands, collector=collector)
            with self.assertRaises(RuntimeError):
                runner.run(limit=1)

    def test_observation_pod_uid_is_required_and_must_match_the_started_pod(self):
        with tempfile.TemporaryDirectory() as directory:
            def mismatched_observer(run, mode, pod_name):
                observation = self.observer(run, mode, pod_name)
                observation["pod_uid"] = "offline-other-pod"
                return observation

            runner, _ = self.make_runner(directory, collector=mismatched_observer)
            with self.assertRaises(RuntimeError):
                runner.run(limit=1)

    def test_restore_observation_without_success_is_terminal_failure_even_without_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            def unsuccessful_restore(run, mode, pod_name):
                observation = self.observer(run, mode, pod_name)
                observation["restore_success"] = False
                return observation

            runner, arguments = self.make_runner(directory, collector=unsuccessful_restore)
            self.assertEqual(self.plan["unblinding_key"][self.plan["schedule"][0]["opaque_arm"]], "restore")
            with self.assertRaises(RuntimeError):
                runner.run(limit=1)
            rows = self.harness.ResultsLedger(arguments["ledger_path"]).read()
            self.assertEqual(len(rows), 1)
            self.assertFalse(rows[0]["restore_success"])
            self.assertIsInstance(rows[0]["failure_reason"], str)
            self.assertTrue(rows[0]["failure_reason"])

    def test_validates_the_exact_frozen_lane_schedule_campaign_and_checkpoint_before_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, arguments = self.make_runner(directory)
            bad_lane = json.loads(arguments["lane_path"].read_text())
            bad_lane["identity"]["source_image"] = "docker.io/vllm/vllm-openai:latest"
            arguments["lane_path"].write_text(json.dumps(bad_lane))
            with self.assertRaises(ValueError):
                runner.run(limit=1)
            self.assertEqual(arguments["command_runner"].calls, [])

            arguments["lane_path"].write_text(json.dumps(self.lane))
            arguments["campaign"]["checkpoint"]["checkpoint_id"] = "h-" + "0" * 61
            runner, arguments = self.make_runner(
                pathlib.Path(directory) / "bad-checkpoint", campaign=arguments["campaign"]
            )
            with self.assertRaises(ValueError):
                runner.run(limit=1)
            self.assertEqual(arguments["command_runner"].calls, [])

    def test_cold_and_restore_use_pinned_paths_unique_v2_pods_and_no_public_control_plane(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, arguments = self.make_runner(directory)
            runner.run(limit=2)
            rows = self.harness.ResultsLedger(arguments["ledger_path"]).read()
            self.assertEqual([row["run_id"] for row in rows], ["v2-01-01", "v2-01-02"])
            self.assertEqual(len({row["pod_name"] for row in rows}), 2)
            self.assertTrue(all(row["pod_name"].startswith("v2-") for row in rows))
            self.assertTrue(all("v2-" in row["pod_name"] for row in rows))

            manifests = sorted((arguments["artifact_dir"] / "manifests").glob("*.json"))
            self.assertEqual(len(manifests), 2)
            for row, manifest_path in zip(rows, manifests):
                manifest = json.loads(manifest_path.read_text())
                self.assertEqual(manifest["kind"], "Pod")
                self.assertEqual(manifest["metadata"]["name"], row["pod_name"])
                self.assertEqual(manifest["metadata"]["labels"]["app.kubernetes.io/name"], "vllm-snapshot-poc")
                self.assertEqual(manifest["metadata"]["labels"]["poc.regolo.ai/run-id"], row["run_id"])
                self.assertEqual(manifest["spec"]["imagePullSecrets"], [])
                self.assertIs(manifest["spec"].get("automountServiceAccountToken"), False)
                self.assertNotIn("serviceAccountName", manifest["spec"])
                volumes = manifest["spec"].get("volumes", [])
                self.assertTrue(all("projected" not in volume for volume in volumes))
                self.assertTrue(all("serviceAccountToken" not in volume for volume in volumes))
                expected_image = (
                    self.lane["identity"]["source_image"]
                    if row["mode"] == "cold" else self.lane["identity"]["candidate_image"]
                )
                self.assertEqual(manifest["spec"]["containers"][0]["image"], expected_image)

            commands = arguments["command_runner"].calls
            self.assertTrue(any(call[:2] == ("kubectl", "create") for call in commands))
            restore = [call for call in commands if call and call[0] == arguments["campaign"]["snapshotctl"]]
            self.assertEqual(len(restore), 1)
            self.assertIn("restore", restore[0])
            self.assertIn("--checkpoint-id", restore[0])
            self.assertIn(self.campaign()["checkpoint"]["checkpoint_id"], restore[0])

    def test_collector_ledger_records_complete_metrics_bindings_and_raw_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, arguments = self.make_runner(directory)
            runner.run(limit=1)
            row = self.harness.ResultsLedger(arguments["ledger_path"]).read()[0]
            self.assertTrue(self.harness.REQUIRED_METRICS.issubset(row))
            self.assertEqual(row["lane_digest"], self.lane["digest"])
            self.assertEqual(row["schedule_digest"], self.harness.schedule_digest(self.plan["schedule"]))
            self.assertEqual(
                {key: row[key] for key in ("block", "sequence_in_block", "opaque_arm")},
                {key: self.plan["schedule"][0][key] for key in ("block", "sequence_in_block", "opaque_arm")},
            )
            self.assertEqual(row["raw_events_ref"], "events/v2-01-01.json")
            self.assertEqual(row["raw_logs_ref"], "logs/v2-01-01.jsonl")
            self.assertIsNone(row["failure_reason"])

    def test_success_ledger_hashes_confined_complete_evidence_and_allows_only_empty_cold_agent_log(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = pathlib.Path(directory) / "artifacts"

            def evidence_collector(run, mode, pod_name):
                contents = {
                    "events": b'{"items":[]}\n',
                    # An empty agent log is meaningful for cold start, but it
                    # still has to be retained and cryptographically bound.
                    "logs": b"" if mode == "cold" else b"restore agent log\n",
                    "telemetry": b'{"gpu_memory_mb":0}\n',
                    "responses": b'{"response":" 2"}\n',
                }
                suffixes = {"events": ".json", "logs": ".jsonl", "telemetry": ".json", "responses": ".json"}
                for category, content in contents.items():
                    path = artifact_dir / category / (run["run_id"] + suffixes[category])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(content)
                observation = self.observer(run, mode, pod_name)
                observation.update(
                    raw_events_ref=f"events/{run['run_id']}.json",
                    raw_logs_ref=f"logs/{run['run_id']}.jsonl",
                    raw_telemetry_ref=f"telemetry/{run['run_id']}.json",
                    raw_response_ref=f"responses/{run['run_id']}.json",
                )
                # Restore the deliberate empty-log fixture after the default
                # observer materializes its ordinary fixture.
                if mode == "cold":
                    (artifact_dir / "logs" / (run["run_id"] + ".jsonl")).write_bytes(b"")
                return observation

            runner, arguments = self.make_runner(
                directory, collector=evidence_collector, artifact_dir=artifact_dir
            )
            runner.run(limit=2)
            rows = self.harness.ResultsLedger(arguments["ledger_path"]).read()
            self.assertEqual(len(rows), 2)
            for row in rows:
                for category, suffix in (
                    ("events", ".json"), ("logs", ".jsonl"),
                    ("telemetry", ".json"), ("responses", ".json"),
                ):
                    evidence_stem = {
                        "events": "raw_events",
                        "logs": "raw_logs",
                        "telemetry": "raw_telemetry",
                        "responses": "raw_response",
                    }[category]
                    ref_key = evidence_stem + "_ref"
                    digest_key = evidence_stem + "_sha256"
                    self.assertEqual(row[ref_key], f"{category}/{row['run_id']}{suffix}")
                    path = artifact_dir / row[ref_key]
                    self.assertTrue(path.is_file())
                    self.assertEqual(row[digest_key], hashlib.sha256(path.read_bytes()).hexdigest())
                    if not (row["mode"] == "cold" and category == "logs"):
                        self.assertNotEqual(path.read_bytes(), b"")
            cold = next(row for row in rows if row["mode"] == "cold")
            self.assertEqual(cold["raw_logs_sha256"], hashlib.sha256(b"").hexdigest())

    def test_success_rejects_missing_unsafe_or_digest_mismatched_raw_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            outside = root / "outside.json"
            outside.write_bytes(b"outside")
            artifact_dir = root / "artifacts"
            link = artifact_dir / "events" / "link.json"
            link.parent.mkdir(parents=True)
            os.symlink(outside, link)

            cases = (
                ("missing", "events/missing.json", None),
                ("symlink", "events/link.json", None),
                ("outside", "../outside.json", None),
                ("digest", "events/v2-01-01.json", "0" * 64),
            )
            for label, bad_ref, bad_digest in cases:
                with self.subTest(label=label):
                    def unsafe_collector(run, mode, pod_name, ref=bad_ref, digest=bad_digest):
                        observation = self.observer(run, mode, pod_name)
                        observation["raw_events_ref"] = ref
                        if digest is not None:
                            observation["raw_events_sha256"] = digest
                        return observation

                    case_root = root / label
                    runner, _ = self.make_runner(
                        case_root, collector=unsafe_collector, artifact_dir=artifact_dir
                    )
                    with self.assertRaises(RuntimeError):
                        runner.run(limit=1)

    def test_checkpoint_attestation_is_digest_pinned_and_validator_runs_before_mutation_or_command(self):
        with tempfile.TemporaryDirectory() as directory:
            timeline = []
            commands = Commands(timeline)
            validator_calls = []

            def validator(path, expected_sha256):
                validator_calls.append((path, expected_sha256))
                timeline.append(("checkpoint-validator", path, expected_sha256))
                self.assertEqual(commands.calls, [])
                self.assertFalse((pathlib.Path(directory) / "artifacts").exists())
                return True

            runner, arguments = self.make_runner(
                directory,
                command_runner=commands,
                collector=PreparedFailureCollector(timeline),
                checkpoint_validator=validator,
            )
            with self.assertRaises(RuntimeError):
                runner.run(limit=1)
            self.assertEqual(
                validator_calls,
                [(arguments["campaign"]["checkpoint"]["attestation_path"],
                  arguments["campaign"]["checkpoint"]["attestation_sha256"])],
            )
            validator_index = next(index for index, item in enumerate(timeline) if item[0] == "checkpoint-validator")
            command_index = next(index for index, item in enumerate(timeline) if item[0] == "command")
            self.assertLess(validator_index, command_index)

            mismatched = json.loads(json.dumps(arguments["campaign"]))
            mismatched["checkpoint"]["attestation_sha256"] = "0" * 64
            bad_commands = Commands()
            bad_validator_calls = []
            runner, bad_arguments = self.make_runner(
                pathlib.Path(directory) / "mismatch",
                campaign=mismatched,
                command_runner=bad_commands,
                checkpoint_validator=lambda path, digest: bad_validator_calls.append((path, digest)) or True,
            )
            with self.assertRaises(ValueError):
                runner.run(limit=1)
            self.assertEqual(bad_commands.calls, [])
            self.assertEqual(bad_validator_calls, [])
            self.assertFalse(bad_arguments["artifact_dir"].exists())

    def test_cluster_preflight_is_required_per_run_before_prepare_artifact_or_command(self):
        with tempfile.TemporaryDirectory() as directory:
            timeline = []
            commands = Commands(timeline)
            collector = PreparedFailureCollector(timeline)

            def preflight():
                timeline.append(("cluster-preflight",))
                self.assertEqual(commands.calls, [])
                self.assertFalse((pathlib.Path(directory) / "artifacts").exists())
                return True

            runner, _ = self.make_runner(
                directory, command_runner=commands, collector=collector, cluster_preflight=preflight
            )
            with self.assertRaises(RuntimeError):
                runner.run(limit=1)
            preflight_index = timeline.index(("cluster-preflight",))
            prepare_index = next(index for index, item in enumerate(timeline) if item[0] == "prepare")
            command_index = next(index for index, item in enumerate(timeline) if item[0] == "command")
            self.assertLess(preflight_index, prepare_index)
            self.assertLess(preflight_index, command_index)

        for bad_preflight in (lambda: False, lambda: (_ for _ in ()).throw(ValueError("cluster unsafe"))):
            with self.subTest(preflight=bad_preflight), tempfile.TemporaryDirectory() as directory:
                commands = Commands()
                collector = PreparedFailureCollector([])
                runner, arguments = self.make_runner(
                    directory, command_runner=commands, collector=collector, cluster_preflight=bad_preflight
                )
                with self.assertRaises(ValueError):
                    runner.run(limit=1)
                self.assertEqual(commands.calls, [])
                self.assertFalse(arguments["artifact_dir"].exists())

    def test_post_start_uid_recovery_retries_only_for_exact_owned_pod_then_cleans_that_uid(self):
        class StartThenDelayedOwnedPod(Commands):
            def __init__(self, *, owned=True, first_uid_lookup="missing"):
                super().__init__()
                self.get_count = 0
                self.owned = owned
                self.first_uid_lookup = first_uid_lookup

            def __call__(self, command):
                call = tuple(map(str, command))
                self.calls.append(call)
                if call[:5] == ("kubectl", "-n", "v2-live-test", "get", "pod"):
                    self.get_count += 1
                    if self.get_count == 1 or self.get_count == 5:
                        return subprocess.CompletedProcess(command, 1, "", "NotFound")
                    if self.get_count == 2:
                        if self.first_uid_lookup == "malformed":
                            return subprocess.CompletedProcess(command, 0, "{malformed", "")
                        if self.first_uid_lookup == "persistent-failure":
                            return subprocess.CompletedProcess(command, 1, "", "transport unavailable")
                        return subprocess.CompletedProcess(command, 1, "", "NotFound")
                    if self.first_uid_lookup == "persistent-failure":
                        return subprocess.CompletedProcess(command, 1, "", "transport unavailable")
                    labels = {
                        "app.kubernetes.io/name": "vllm-snapshot-poc",
                        "poc.regolo.ai/lane": "v2-a",
                        "poc.regolo.ai/run-id": "v2-01-01" if self.owned else "other-run",
                    }
                    return subprocess.CompletedProcess(command, 0, json.dumps({
                        "metadata": {"uid": "uid-recovered", "labels": labels},
                        "spec": {"nodeName": self_node},
                    }), "")
                return subprocess.CompletedProcess(command, 0, "ok", "")

        self_node = self.lane["identity"]["node"]
        with tempfile.TemporaryDirectory() as directory:
            commands = StartThenDelayedOwnedPod()
            runner, arguments = self.make_runner(
                directory, command_runner=commands, collector=PreparedFailureCollector([], pod_uid="uid-recovered")
            )
            expected_pod, _ = runner._manifest(
                self.plan["schedule"][0], "restore", self.lane,
                self.harness.schedule_digest(self.plan["schedule"]),
            )
            with self.assertRaises(RuntimeError):
                runner.run(limit=1)
            self.assertEqual(commands.get_count, 5)
            deletes = [call for call in commands.calls if "--raw" in call]
            self.assertEqual(len(deletes), 1)
            member_uri = next(value for value in deletes[0] if value.startswith("/api/v1/namespaces/v2-live-test/pods/"))
            self.assertEqual(member_uri, "/api/v1/namespaces/v2-live-test/pods/" + expected_pod)
            options = pathlib.Path(deletes[0][deletes[0].index("-f") + 1])
            self.assertEqual(json.loads(options.read_text()), {"preconditions": {"uid": "uid-recovered"}})
            self.assertTrue(any(call and call[0] == arguments["campaign"]["snapshotctl"] for call in commands.calls))

        with tempfile.TemporaryDirectory() as directory:
            commands = StartThenDelayedOwnedPod(first_uid_lookup="malformed")
            runner, arguments = self.make_runner(
                directory, command_runner=commands, collector=PreparedFailureCollector([], pod_uid="uid-recovered")
            )
            with self.assertRaises(RuntimeError):
                runner.run(limit=1)
            self.assertGreaterEqual(commands.get_count, 5)
            self.assertTrue(any("--raw" in call for call in commands.calls))

        with tempfile.TemporaryDirectory() as directory:
            commands = StartThenDelayedOwnedPod(first_uid_lookup="persistent-failure")
            runner, arguments = self.make_runner(directory, command_runner=commands)
            with self.assertRaises(RuntimeError):
                runner.run(limit=1)
            self.assertTrue(any(call and call[0] == arguments["campaign"]["snapshotctl"] for call in commands.calls))
            self.assertFalse(any("--raw" in call for call in commands.calls))

        with tempfile.TemporaryDirectory() as directory:
            commands = StartThenDelayedOwnedPod(owned=False)
            runner, arguments = self.make_runner(directory, command_runner=commands)
            with self.assertRaises(RuntimeError):
                runner.run(limit=1)
            self.assertTrue(any(call and call[0] == arguments["campaign"]["snapshotctl"] for call in commands.calls))
            self.assertFalse(any("--raw" in call for call in commands.calls))

    def test_failure_is_terminally_ledgered_then_aborts_on_xid_oom_io_or_correctness(self):
        for reason in ("NVRM: Xid 79", "OOMKilled", "checkpoint I/O error", "correctness probe mismatch"):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                def failed_observer(run, mode, pod_name):
                    return {
                        "metrics": {}, "failure_reason": reason, "failure_stage": "collector",
                        "pod_uid": "offline-" + pod_name,
                    }

                runner, arguments = self.make_runner(directory, collector=failed_observer)
                with self.assertRaises(RuntimeError):
                    runner.run(limit=2)
                rows = self.harness.ResultsLedger(arguments["ledger_path"]).read()
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["failure_reason"], reason)
                self.assertEqual(rows[0]["failure_stage"], "collector")
                self.assertTrue(self.harness.REQUIRED_METRICS.issubset(rows[0]))
                self.assertTrue(all(row["failure_reason"] is None for row in rows[1:]))

                deletes = [call for call in arguments["command_runner"].calls if "--raw" in call]
                self.assertEqual(len(deletes), 1)
                member_uri = next(value for value in deletes[0] if value.startswith("/api/v1/namespaces/"))
                self.assertEqual(member_uri, f"/api/v1/namespaces/v2-live-test/pods/{rows[0]['pod_name']}")
                self.assertFalse(any(call[:3] == ("kubectl", "delete", "namespace") for call in arguments["command_runner"].calls))

    def test_resume_verifies_existing_ledger_and_skips_only_completed_exact_plan_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, arguments = self.make_runner(directory)
            first = self.plan["schedule"][0]
            mode = self.plan["unblinding_key"][first["opaque_arm"]]
            record = self.harness.complete_metric_record(first["run_id"], self.metrics(mode == "restore"), None)
            record.update(
                block=first["block"], sequence_in_block=first["sequence_in_block"],
                opaque_arm=first["opaque_arm"], mode=mode, pod_name="v2-resumed-v2-01-01",
                valid_response=True, restore_success=mode == "restore", excluded=False, error=None,
                lane_digest=self.lane["digest"], schedule_digest=self.harness.schedule_digest(self.plan["schedule"]),
                execution_digest=arguments["execution_digest"],
                **self.raw_evidence(arguments["artifact_dir"], first["run_id"], cold=mode == "cold"),
            )
            self.harness.ResultsLedger(arguments["ledger_path"]).append(record)
            runner.run(limit=1)
            rows = self.harness.ResultsLedger(arguments["ledger_path"]).read()
            self.assertEqual([row["run_id"] for row in rows], ["v2-01-01", "v2-01-02"])
            created = [call for call in arguments["command_runner"].calls if call[:2] == ("kubectl", "create")]
            self.assertEqual(len(created), 1)

            rows[0]["lane_digest"] = "0" * 64
            # Rebuild a syntactically valid but semantically wrong ledger record.
            pathlib.Path(arguments["ledger_path"]).unlink()
            bad = self.harness.ResultsLedger(arguments["ledger_path"])
            bad.append({key: value for key, value in rows[0].items() if key not in {"sequence", "previous_record_digest", "record_digest"}})
            runner, arguments = self.make_runner(
                pathlib.Path(directory) / "retry", ledger_path=arguments["ledger_path"]
            )
            with self.assertRaises(ValueError):
                runner.run(limit=1)

    def test_resume_rejects_a_ledger_row_whose_unblinded_mode_is_inverted(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, arguments = self.make_runner(directory)
            first = self.plan["schedule"][0]
            expected_mode = self.plan["unblinding_key"][first["opaque_arm"]]
            record = self.harness.complete_metric_record(
                first["run_id"], self.metrics(expected_mode == "restore"), None
            )
            record.update(
                block=first["block"], sequence_in_block=first["sequence_in_block"],
                opaque_arm=first["opaque_arm"],
                mode="cold" if expected_mode == "restore" else "restore",
                pod_name="v2-inverted-v2-01-01", pod_uid="uid-inverted",
                valid_response=True, restore_success=expected_mode == "restore",
                excluded=False, error=None, lane_digest=self.lane["digest"],
                schedule_digest=self.harness.schedule_digest(self.plan["schedule"]),
                execution_digest=arguments["execution_digest"],
                **self.raw_evidence(arguments["artifact_dir"], first["run_id"], cold=expected_mode == "cold"),
            )
            self.harness.ResultsLedger(arguments["ledger_path"]).append(record)
            with self.assertRaises(ValueError):
                runner.run(limit=1)

    def test_resume_revalidates_all_success_raw_evidence_before_any_command(self):
        mutations = ("tamper", "delete", "symlink")
        evidence_stems = ("raw_events", "raw_logs", "raw_telemetry", "raw_response")
        for stem in evidence_stems:
            for mutation in mutations:
                with self.subTest(evidence=stem, mutation=mutation), tempfile.TemporaryDirectory() as directory:
                    runner, arguments = self.make_runner(directory)
                    runner.run(limit=1)
                    row = self.harness.ResultsLedger(arguments["ledger_path"]).read()[0]
                    for required in evidence_stems:
                        self.assertIsInstance(row[required + "_ref"], str)
                        self.assertRegex(row[required + "_sha256"], r"^[0-9a-f]{64}$")
                    target = arguments["artifact_dir"] / row[stem + "_ref"]
                    if mutation == "tamper":
                        target.write_bytes(b"tampered evidence\n")
                    elif mutation == "delete":
                        target.unlink()
                    else:
                        outside = pathlib.Path(directory) / "outside-evidence"
                        outside.write_bytes(b"outside")
                        target.unlink()
                        os.symlink(outside, target)
                    fresh_commands = Commands()
                    fresh = dict(arguments, command_runner=fresh_commands)
                    with self.assertRaises(ValueError):
                        self.live.LiveRunner(**fresh).run(limit=1)
                    self.assertEqual(fresh_commands.calls, [])

    def test_dry_run_only_returns_commands_and_never_mutates_or_calls_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, arguments = self.make_runner(directory, dry_run=True)
            result = runner.run(limit=2)
            self.assertIn("commands", result)
            self.assertGreaterEqual(len(result["commands"]), 2)
            self.assertEqual(arguments["command_runner"].calls, [])
            self.assertFalse(arguments["ledger_path"].exists())
            self.assertFalse(arguments["artifact_dir"].exists())

    def test_storage_characterization_is_bounded_read_only_and_restore_advice_is_injected_immediately_before_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = pathlib.Path(directory) / "checkpoint"
            checkpoint.mkdir()
            pages = checkpoint / "pages.img"
            pages.write_bytes(b"checkpoint")
            advised = []
            timeline = []
            command_runner = Commands(timeline)

            def fadvise(paths):
                advised.append(tuple(paths))
                timeline.append(("fadvise", tuple(paths)))

            runner, arguments = self.make_runner(
                directory, checkpoint_files=(pages,), fadvise=fadvise, command_runner=command_runner
            )
            report = runner.characterize_storage(max_bytes=4, max_reads=3)
            self.assertLessEqual(report["bytes_read"], 4)
            self.assertLessEqual(report["read_operations"], 3)
            self.assertEqual(advised, [])
            runner.run(limit=2)
            restore_runs = [row for row in self.harness.ResultsLedger(arguments["ledger_path"]).read() if row["mode"] == "restore"]
            self.assertEqual(len(advised), len(restore_runs))
            self.assertTrue(all(call == (pages,) for call in advised))
            for index, entry in enumerate(timeline):
                if entry[0] == "command" and entry[1][0] == arguments["campaign"]["snapshotctl"]:
                    self.assertEqual(timeline[index - 1], ("fadvise", (pages,)))

    def test_runner_has_no_v0_harness_dependency_or_service_ingress_token_surface(self):
        source = LIVE.read_text()
        self.assertNotIn("verification/v0", source)
        self.assertNotIn("from v0", source)
        self.assertNotIn("kind: Service", source)
        self.assertNotIn("kind: Ingress", source)


if __name__ == "__main__":
    unittest.main()
