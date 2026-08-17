"""Offline command contracts for the V2 production collector and cache advisor."""

import hashlib
import importlib.util
import json
import os
import pathlib
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


V2_ROOT = pathlib.Path(__file__).resolve().parents[1]
PRODUCTION = V2_ROOT / "harness" / "v2_production.py"
NAMESPACE = "v2-live-test"
AGENT = "snapshot-agent"
POD = "v2-01-01"
ROOT = "/checkpoints/h-" + "a" * 61
PAGES = ROOT + "/pages-12.img"
REQUIRED_METRICS = {
    "pod_to_scheduled_s", "pod_to_restore_start_s", "criu_restore_s", "cuda_restore_s",
    "ready_s", "http_200_s", "first_token_s", "cgroup_io_stat", "diskstats",
    "node_page_cache_bytes", "node_memory_available_bytes", "psi_cpu", "psi_io",
    "psi_memory", "node_cpu_utilization", "gpu_memory_mib", "checkpoint_size_bytes",
    "pages_12_size_bytes", "rootfs_size_bytes", "metadata_size_bytes", "prepare_s",
    "sleep_s", "wake_s", "admission_closed", "harness_inflight", "vllm_running",
    "vllm_waiting", "tokens_per_second",
}


def load_production():
    if not PRODUCTION.is_file():
        raise FileNotFoundError(f"missing required V2 production module: {PRODUCTION}")
    spec = importlib.util.spec_from_file_location("v2_production_under_test", PRODUCTION)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PRODUCTION}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FixtureTransport:
    """Command fixture; all returned evidence is local and deterministic."""

    def __init__(self, *, fault=None, response=" 2.\n\nThe answer to 1+1 is 2.", unstable_uid=False):
        self.calls = []
        self.fault = fault
        self.response = response
        self.proc_stat_reads = 0
        self.pod_polls = 0
        self.unstable_uid = unstable_uid

    def __call__(self, argv, timeout_s=None):
        command = tuple(map(str, argv))
        self.calls.append((command, timeout_s))
        text = " ".join(command)
        if command[:6] == ("kubectl", "-n", NAMESPACE, "get", "pod", POD):
            self.pod_polls += 1
            conditions = [{"type": "PodScheduled", "status": "True", "lastTransitionTime": "2026-08-14T00:00:02Z"}]
            if self.pod_polls > 1:
                conditions.append({"type": "Ready", "status": "True", "lastTransitionTime": "2026-08-14T00:00:08Z"})
            return self.ok(command, json.dumps({
                "metadata": {
                    "uid": "uid-v2-01-01" if not self.unstable_uid or self.pod_polls == 1 else "uid-replaced",
                    "creationTimestamp": "2026-08-14T00:00:00Z",
                    "annotations": {"nvidia.com/snapshot-restore-status.server": "completed"},
                },
                "status": {"conditions": conditions, "containerStatuses": [{"name": "server", "containerID": "containerd://current"}]},
            }))
        if " get events " in f" {text} ":
            checkpoint_id = "h-" + "a" * 61
            return self.ok(command, json.dumps({"items": [
                {"involvedObject": {"uid": "uid-v2-01-01", "name": POD, "namespace": NAMESPACE},
                 "reason": "RestoreRequested", "message": "Restore requested from checkpoint %s for container server" % checkpoint_id},
                {"involvedObject": {"uid": "uid-v2-01-01", "name": POD, "namespace": NAMESPACE},
                 "reason": "RestoreSucceeded", "message": "Restore completed from checkpoint %s" % checkpoint_id},
            ]}))
        if " logs " in f" {text} ":
            if self.fault:
                return self.ok(command, self.fault + "\n")
            return self.ok(command, "\n".join((
                '2026-08-14T00:00:03Z\tINFO snapshot-agent\t=== Starting external restore ===\t{"pod":"other-ns/other-pod","checkpoint":"other"}',
                '2026-08-14T00:00:03Z Restore timing summary {"pod":"other-ns/other-pod","restore":{"phases":{"criu_restore_duration":"99s","cuda_duration":"99s"}}}',
                '2026-08-14T00:00:03Z\tINFO snapshot-agent\t=== Starting external restore ===\t{"pod":"v2-live-test/v2-01-01","namespace":"v2-live-test","container_id":"current","checkpoint_id":"h-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}',
                '2026-08-14T00:00:04Z Restore timing summary {"pod":"v2-live-test/v2-01-01","container_id":"current","checkpoint_id":"h-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","restore":{"phases":{"criu_restore_duration":"2.5s","cuda_duration":"1.5s"}}}',
            )))
        if command[:7] == ("kubectl", "-n", NAMESPACE, "exec", POD, "-c", "server") and "python3" in command:
            return self.ok(command, json.dumps({
                "response": self.response, "http_200_epoch_s": 1786665609.0,
                "first_token_epoch_s": 1786665610.0, "tokens_per_second": 42.0,
                "running": 0, "waiting": 0,
            }))
        if "/host/proc/meminfo" in text:
            phase = self.proc_stat_reads + 1
            return self.ok(command, "MemAvailable: %d kB\nCached: %d kB\nSReclaimable: 20 kB\nShmem: 5 kB\n" % (101 - phase, 29 + phase))
        if any(path in text for path in ("/host/proc/pressure/cpu", "/host/proc/pressure/io", "/host/proc/pressure/memory")):
            return self.ok(command, "some avg10=0.00 avg60=0.00 avg300=0.00 total=1\n")
        if "/sys/fs/cgroup/io.stat" in text:
            return self.ok(command, "253:0 rbytes=%d wbytes=0\n" % (100 + self.proc_stat_reads + 1))
        if "/host/proc/diskstats" in text:
            value = 100 + self.proc_stat_reads + 1
            return self.ok(command, "253 0 dm-0 1 0 %d 1 0 0 0 0 0 0 0 0\n7 6 loop6 1 0 %d 1 0 0 0 0 0 0 0 0\n8 0 sda 1 0 %d 1 0 0 0 0 0 0 0 0\n" % (value, value, value))
        if "/host/proc/stat" in text:
            self.proc_stat_reads += 1
            return self.ok(command, "cpu  10 0 0 %d 0 0 0 0\n" % (10 + self.proc_stat_reads))
        if "nvidia-smi" in text:
            return self.ok(command, "123\n")
        if "stat" in command:
            return self.ok(command, "regular file 4\n")
        if "dd" in command:
            return self.ok(command, "")
        return self.ok(command, "")

    @staticmethod
    def ok(command, stdout):
        return subprocess.CompletedProcess(command, 0, stdout, "")


class ProductionCollectorTests(unittest.TestCase):
    def setUp(self):
        self.module = load_production()
        self.run = {"run_id": POD, "block": 1, "sequence_in_block": 1, "opaque_arm": "B"}
        self.attestation = {
            "checkpoint_id": "h-" + "a" * 61, "compatibility_hash": "a" * 64,
            "checkpoint_size_bytes": 400, "pages_12_size_bytes": 300,
            "rootfs_size_bytes": 80, "metadata_size_bytes": 20,
        }

    def collector(self, directory, transport):
        return self.module.ProductionCollector(
            NAMESPACE, AGENT, pathlib.Path(directory) / "artifacts", self.attestation, transport, timeout_s=17
        )

    def test_subprocess_transport_uses_argv_capture_timeout_and_never_shell(self):
        with mock.patch.object(self.module.subprocess, "run") as run:
            run.return_value = subprocess.CompletedProcess(["tool"], 0, "ok", "")
            result = self.module.SubprocessTransport(timeout_s=9)(["tool", "literal;not-shell"])
        self.assertEqual(result.stdout, "ok")
        run.assert_called_once_with(["tool", "literal;not-shell"], text=True, capture_output=True, timeout=9, shell=False, check=False)
        self.assertEqual(self.module._epoch("2026-08-13T23:05:40.16886412Z"), 1786662340.168864)
        self.assertEqual(self.module._seconds("2m18.234300224s"), 138.234300224)
        for duration, expected in (("250ms", 0.25), ("1us", 0.000001), ("1ns", 0.000000001), ("1m2.5s", 62.5)):
            with self.subTest(duration=duration):
                self.assertEqual(self.module._seconds(duration), expected)
        for invalid in ("-1s", "2m", "2seconds", "", None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.module._seconds(invalid)

    def test_prepare_collects_timestamp_and_agent_host_before_without_querying_future_workload_pod(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = FixtureTransport()
            self.collector(directory, transport).prepare(self.run, "restore", POD)
        calls = [command for command, _ in transport.calls]
        self.assertFalse(any(command[:5] == ("kubectl", "-n", NAMESPACE, "get", "pod") for command in calls))
        agent_calls = [command for command in calls if command[:7] == ("kubectl", "-n", NAMESPACE, "exec", AGENT, "-c", "agent")]
        for path in (
            "/host/proc/stat", "/host/proc/meminfo", "/host/proc/pressure/cpu",
            "/host/proc/pressure/io", "/host/proc/pressure/memory", "/host/proc/diskstats",
            "/sys/fs/cgroup/io.stat",
        ):
            self.assertTrue(any(path in command for command in agent_calls), path)

    def test_collect_polls_pod_logs_agent_filters_target_and_uses_workload_loopback_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = FixtureTransport()
            collector = self.collector(directory, transport)
            collector.prepare(self.run, "restore", POD)
            observation = collector(self.run, "restore", POD)
            metrics = observation["metrics"]
            self.assertTrue(REQUIRED_METRICS.issubset(metrics))
            self.assertEqual((observation["pod_uid"], observation["pod_creation_epoch_s"]), ("uid-v2-01-01", 1786665600.0))
            self.assertEqual((metrics["criu_restore_s"], metrics["cuda_restore_s"]), (2.5, 1.5))
            self.assertEqual((metrics["vllm_running"], metrics["vllm_waiting"]), (0, 0))
            self.assertTrue(all(metrics[field] is None for field in ("prepare_s", "sleep_s", "wake_s")))
            calls = [command for command, _ in transport.calls]
            self.assertGreaterEqual(transport.pod_polls, 2)
            pod_lookup = next(index for index, call in enumerate(calls) if call[:5] == ("kubectl", "-n", NAMESPACE, "get", "pod"))
            stat_indices = [index for index, call in enumerate(calls) if "/host/proc/stat" in call]
            self.assertEqual(len(stat_indices), 3)
            self.assertLess(stat_indices[0], pod_lookup)
            self.assertGreater(stat_indices[1], pod_lookup)
            self.assertGreater(stat_indices[2], stat_indices[1])
            logs = next(command for command in calls if "logs" in command)
            self.assertEqual(logs[:7], ("kubectl", "-n", NAMESPACE, "logs", AGENT, "-c", "agent"))
            self.assertTrue(any(value.startswith("--since-time=") for value in logs))
            probe = next(command for command in calls if command[:7] == ("kubectl", "-n", NAMESPACE, "exec", POD, "-c", "server") and "python3" in command)
            self.assertEqual(probe[-2], "-c")
            self.assertTrue(probe[-1])
            for required in (
                "localhost:8000", "/health", "/v1/completions", "/metrics",
                "openai/gpt-oss-20b", "The answer to 1+1 is", "max_tokens", "128",
                "temperature", "stream", "urllib.request.Request", "urlopen",
                "data:", "time.time", "choices", "text", "tokens_per_second",
                "vllm:num_requests_running", "vllm:num_requests_waiting",
            ):
                self.assertIn(required, probe[-1])
            self.assertNotIn("'response':' 2'", probe[-1])
            self.assertNotIn('"response":" 2"', probe[-1])
            self.assertNotIn('"response": " 2"', probe[-1])
            gpu = next(command for command in calls if command[:7] == ("kubectl", "-n", NAMESPACE, "exec", POD, "-c", "server") and "nvidia-smi" in command)
            self.assertEqual(gpu[-2:], ("--query-compute-apps=used_memory", "--format=csv,noheader,nounits"))
            events = next(command for command in calls if "events" in command)
            self.assertIn("--field-selector", events)
            for ref in (observation["raw_events_ref"], observation["raw_logs_ref"], observation["raw_telemetry_ref"], observation["raw_response_ref"]):
                self.assertTrue((pathlib.Path(directory) / "artifacts" / ref).is_file(), ref)
        with tempfile.TemporaryDirectory() as directory:
            transport = FixtureTransport(unstable_uid=True)
            collector = self.collector(directory, transport)
            collector.prepare(self.run, "restore", POD)
            with self.assertRaises(ValueError):
                collector(self.run, "restore", POD)

    def test_collector_fails_closed_on_agent_fault_or_invalid_loopback_response(self):
        for fault in ("NVRM: Xid 79", "OOMKilled", "I/O error"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                collector = self.collector(directory, FixtureTransport(fault=fault))
                collector.prepare(self.run, "restore", POD)
                with self.assertRaises(ValueError):
                    collector(self.run, "restore", POD)
        with tempfile.TemporaryDirectory() as directory:
            collector = self.collector(directory, FixtureTransport(response="3"))
            collector.prepare(self.run, "restore", POD)
            with self.assertRaises(ValueError):
                collector(self.run, "restore", POD)

    def test_restore_collector_emits_delta_storage_metrics_and_three_recomputable_telemetry_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = FixtureTransport()
            collector = self.collector(directory, transport)
            collector.prepare(self.run, "restore", POD)
            observation = collector(self.run, "restore", POD)
            metrics = observation["metrics"]
            self.assertEqual(metrics["checkpoint_storage_read_bytes"], 1)
            self.assertEqual(metrics["checkpoint_storage_read_throughput_bytes_s"], 1.0)
            self.assertEqual(metrics["token_after_restore_summary_s"], 6.0)
            self.assertEqual(metrics["cgroup_io_stat"], {"253:0": {"rbytes": 1, "wbytes": 0}})
            self.assertEqual(set(metrics["diskstats"]), {"dm-0", "loop6", "sda"})
            self.assertEqual(metrics["diskstats"]["dm-0"]["sectors_read"], 1)
            self.assertEqual(metrics["node_page_cache_delta_bytes"], 1024)
            self.assertEqual(metrics["node_memory_available_delta_bytes"], -1024)
            telemetry = pathlib.Path(directory) / "artifacts" / observation["raw_telemetry_ref"]
            snapshots = json.loads(telemetry.read_text())
            self.assertTrue({"host_before", "storage_after", "final"}.issubset(snapshots))

    def test_restore_log_selects_one_current_container_pair_and_rejects_missing_or_ambiguous_pairs(self):
        current_id = "containerd://new-container-id"
        checkpoint_id = "h-" + "a" * 61

        class SamePodRestores(FixtureTransport):
            def __init__(self, log):
                super().__init__()
                self.log = log

            def __call__(self, argv, timeout_s=None):
                result = super().__call__(argv, timeout_s)
                command = tuple(map(str, argv))
                if command[:6] == ("kubectl", "-n", NAMESPACE, "get", "pod", POD):
                    pod = json.loads(result.stdout)
                    pod["status"]["containerStatuses"] = [{"name": "server", "containerID": current_id}]
                    return self.ok(command, json.dumps(pod))
                if " logs " in f" {' '.join(command)} ":
                    return self.ok(command, self.log)
                return result

        old_start = '2026-08-13T23:59:50Z\tINFO snapshot-agent\t=== Starting external restore ===\t{"pod":"v2-live-test/v2-01-01","container_id":"containerd://old-container-id","checkpoint_id":"%s"}' % checkpoint_id
        old_summary = '2026-08-13T23:59:51Z Restore timing summary {"pod":"v2-live-test/v2-01-01","container_id":"containerd://old-container-id","checkpoint_id":"%s","restore":{"phases":{"criu_restore_duration":"99s","cuda_duration":"99s"}}}' % checkpoint_id
        new_start = '2026-08-14T00:00:03Z\tINFO snapshot-agent\t=== Starting external restore ===\t{"pod":"v2-live-test/v2-01-01","container_id":"containerd://new-container-id","checkpoint_id":"%s"}' % checkpoint_id
        new_summary = '2026-08-14T00:00:04Z Restore timing summary {"pod":"v2-live-test/v2-01-01","container_id":"containerd://new-container-id","checkpoint_id":"%s","restore":{"phases":{"criu_restore_duration":"2.5s","cuda_duration":"1.5s"}}}' % checkpoint_id

        with tempfile.TemporaryDirectory() as directory:
            collector = self.collector(directory, SamePodRestores("\n".join((old_start, old_summary, new_start, new_summary))))
            collector.prepare(self.run, "restore", POD)
            self.assertEqual(collector(self.run, "restore", POD)["metrics"]["criu_restore_s"], 2.5)

        for name, log in {
            "missing": "\n".join((old_start, old_summary)),
            "ambiguous": "\n".join((
                new_start, new_summary, new_start,
                new_summary.replace('"cuda_duration":"1.5s"', '"cuda_duration":"9.5s"'),
            )),
        }.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                collector = self.collector(directory, SamePodRestores(log))
                collector.prepare(self.run, "restore", POD)
                with self.assertRaises(ValueError):
                    collector(self.run, "restore", POD)

    def test_restore_log_requires_matching_checkpoint_id_and_returns_start_and_summary_epochs(self):
        checkpoint_id = "h-" + "a" * 61
        start = (
            '2026-08-14T00:00:03Z\tINFO\t=== Starting external restore ===\t'
            '{"pod":"v2-live-test/v2-01-01","container_id":"current","checkpoint_id":"%s"}' % checkpoint_id
        )
        summary = (
            '2026-08-14T00:00:04Z Restore timing summary '
            '{"pod":"v2-live-test/v2-01-01","container_id":"current","checkpoint_id":"%s",'
            '"restore":{"phases":{"criu_restore_duration":"2.5s","cuda_duration":"1.5s"}}}' % checkpoint_id
        )
        result = self.module._restore_log(
            start + "\n" + summary, "v2-live-test/v2-01-01", "current", 1786665600.0, checkpoint_id
        )
        self.assertEqual(result[0:2], (1786665603.0, 1786665604.0))
        self.assertEqual(result[2]["criu_restore_duration"], "2.5s")
        for log in (
            start.replace('"checkpoint_id":"%s"' % checkpoint_id, '"checkpoint_id":"h-' + "0" * 61 + '"'),
            summary.replace('"checkpoint_id":"%s"' % checkpoint_id, '"checkpoint_id":"h-' + "0" * 61 + '"'),
        ):
            with self.subTest(log=log), self.assertRaises(ValueError):
                self.module._restore_log(
                    log + "\n" + (summary if log == start else start),
                    "v2-live-test/v2-01-01", "current", 1786665600.0, checkpoint_id,
                )

    def test_restore_events_require_one_uid_bound_requested_and_succeeded_message(self):
        checkpoint_id = "h-" + "a" * 61
        events = {"items": [
            {"involvedObject": {"uid": "uid-v2", "name": POD, "namespace": NAMESPACE},
             "message": "Successfully assigned", "reason": "Scheduled"},
            {"involvedObject": {"uid": "uid-v2", "name": POD, "namespace": NAMESPACE},
             "message": "Container image already present", "reason": "Pulled"},
            {"involvedObject": {"uid": "uid-v2", "name": POD, "namespace": NAMESPACE},
             "message": "Started container server", "reason": "Started"},
            {"involvedObject": {"uid": "uid-v2", "name": POD, "namespace": NAMESPACE},
             "message": "Restore requested from checkpoint %s for container server" % checkpoint_id,
             "reason": "RestoreRequested"},
            {"involvedObject": {"uid": "uid-v2", "name": POD, "namespace": NAMESPACE},
             "message": "Restore completed from checkpoint %s" % checkpoint_id,
             "reason": "RestoreSucceeded"},
        ]}
        self.assertTrue(self.module._restore_events(events, NAMESPACE, POD, "uid-v2", checkpoint_id))
        for mutate in (
            lambda value: value["items"].pop(),
            lambda value: value["items"].append(dict(value["items"][4])),
            lambda value: value["items"].__setitem__(3, {**value["items"][3], "message": "Restore requested from checkpoint h-" + "0" * 61 + " for container server"}),
        ):
            with self.subTest(mutate=mutate):
                candidate = json.loads(json.dumps(events))
                mutate(candidate)
                with self.assertRaises(ValueError):
                    self.module._restore_events(candidate, NAMESPACE, POD, "uid-v2", checkpoint_id)

        for message in (
            "prefix Restore completed from checkpoint %s" % checkpoint_id,
            "Restore completed from checkpoint %s suffix" % checkpoint_id,
            "Restore completed from checkpoint %s and h-%s" % (checkpoint_id, "0" * 61),
        ):
            with self.subTest(message=message):
                candidate = json.loads(json.dumps(events))
                candidate["items"][4]["message"] = message
                with self.assertRaises(ValueError):
                    self.module._restore_events(candidate, NAMESPACE, POD, "uid-v2", checkpoint_id)

    def test_restore_log_accepts_identical_json_duplicates_but_rejects_conflicting_identity_duplicates(self):
        checkpoint_id = "h-" + "a" * 61
        start = ('2026-08-14T00:00:03Z === Starting external restore === '
                 '{"pod":"v2-live-test/v2-01-01","container_id":"current","checkpoint_id":"%s","checkpoint_id":"%s"}' % (checkpoint_id, checkpoint_id))
        summary = ('2026-08-14T00:00:04Z Restore timing summary '
                   '{"pod":"v2-live-test/v2-01-01","container_id":"current","checkpoint_id":"%s","checkpoint_id":"%s","restore":{"phases":{"criu_restore_duration":"2.5s","cuda_duration":"1.5s"}}}' % (checkpoint_id, checkpoint_id))
        self.assertEqual(self.module._restore_log(start + "\n" + summary, NAMESPACE + "/" + POD, "current", 1786665600.0, checkpoint_id)[0], 1786665603.0)
        live_start = (
            '2026-08-14T00:00:03Z === Starting external restore === '
            '{"pod":"v2-live-test/v2-01-01","checkpoint_id":"%s","container_id":"current",'
            '"checkpoint_id":"%s","pod":"v2-01-01","namespace":"v2-live-test","container":"server"}'
            % (checkpoint_id, checkpoint_id)
        )
        self.assertEqual(
            self.module._restore_log(
                live_start + "\n" + summary,
                NAMESPACE + "/" + POD,
                "current",
                1786665600.0,
                checkpoint_id,
            )[0],
            1786665603.0,
        )
        conflicts = {
            "checkpoint_id": start.replace('"checkpoint_id":"%s","checkpoint_id":"%s"' % (checkpoint_id, checkpoint_id), '"checkpoint_id":"%s","checkpoint_id":"h-%s"' % (checkpoint_id, "0" * 61)),
            "pod": start.replace('"pod":"%s"' % (NAMESPACE + "/" + POD), '"pod":"%s","pod":"other/pod"' % (NAMESPACE + "/" + POD)),
            "container_id": start.replace('"container_id":"current"', '"container_id":"current","container_id":"other"'),
        }
        reverse_conflicts = {
            "checkpoint_id": start.replace('"checkpoint_id":"%s","checkpoint_id":"%s"' % (checkpoint_id, checkpoint_id), '"checkpoint_id":"h-%s","checkpoint_id":"%s"' % ("0" * 61, checkpoint_id)),
            "pod": start.replace('"pod":"%s"' % (NAMESPACE + "/" + POD), '"pod":"other/pod","pod":"%s"' % (NAMESPACE + "/" + POD)),
            "container_id": start.replace('"container_id":"current"', '"container_id":"other","container_id":"current"'),
        }
        for order, values in (("bad-last", conflicts), ("bad-first", reverse_conflicts)):
            for field, conflict in values.items():
                with self.subTest(order=order, field=field):
                    with self.assertRaises(ValueError):
                        self.module._restore_log(conflict + "\n" + summary, NAMESPACE + "/" + POD, "current", 1786665600.0, checkpoint_id)

    def test_restore_container_id_normalizes_containerd_prefix_and_rejects_bad_runtime_or_mismatch(self):
        checkpoint_id = "h-" + "a" * 61
        def log(container_id):
            return "\n".join((
                '2026-08-14T00:00:03Z\tINFO snapshot-agent\t=== Starting external restore ===\t{"pod":"v2-live-test/v2-01-01","container_id":"%s","checkpoint_id":"%s"}' % (container_id, checkpoint_id),
                '2026-08-14T00:00:04Z Restore timing summary {"pod":"v2-live-test/v2-01-01","container_id":"%s","checkpoint_id":"%s","restore":{"phases":{"criu_restore_duration":"2.5s","cuda_duration":"1.5s"}}}' % (container_id, checkpoint_id),
            ))

        class ContainerIdentityTransport(FixtureTransport):
            def __init__(self, runtime_id, agent_id):
                super().__init__()
                self.runtime_id = runtime_id
                self.agent_id = agent_id

            def __call__(self, argv, timeout_s=None):
                result = super().__call__(argv, timeout_s)
                command = tuple(map(str, argv))
                if command[:6] == ("kubectl", "-n", NAMESPACE, "get", "pod", POD):
                    pod = json.loads(result.stdout)
                    pod["status"]["containerStatuses"] = [{"name": "server", "containerID": self.runtime_id}]
                    return self.ok(command, json.dumps(pod))
                if " logs " in f" {' '.join(command)} ":
                    return self.ok(command, log(self.agent_id))
                return result

        with tempfile.TemporaryDirectory() as directory:
            collector = self.collector(directory, ContainerIdentityTransport("containerd://abc", "abc"))
            collector.prepare(self.run, "restore", POD)
            self.assertEqual(collector(self.run, "restore", POD)["metrics"]["cuda_restore_s"], 1.5)

        for runtime_id, agent_id in (("docker://abc", "abc"), ("containerd://abc", "other")):
            with self.subTest(runtime_id=runtime_id, agent_id=agent_id), tempfile.TemporaryDirectory() as directory:
                collector = self.collector(directory, ContainerIdentityTransport(runtime_id, agent_id))
                collector.prepare(self.run, "restore", POD)
                with self.assertRaises(ValueError):
                    collector(self.run, "restore", POD)

    def test_cold_collection_succeeds_without_target_restore_log_and_marks_restore_metrics_null(self):
        class ColdLogs(FixtureTransport):
            def __call__(self, argv, timeout_s=None):
                result = super().__call__(argv, timeout_s)
                command = tuple(map(str, argv))
                if " logs " in f" {' '.join(command)} ":
                    return self.ok(command, "2026-08-14T00:00:03Z ordinary agent line without restore evidence\n")
                return result

        with tempfile.TemporaryDirectory() as directory:
            collector = self.collector(directory, ColdLogs())
            collector.prepare(self.run, "cold", POD)
            observation = collector(self.run, "cold", POD)
            self.assertTrue(REQUIRED_METRICS.issubset(observation["metrics"]))
            self.assertTrue(all(observation["metrics"][field] is None for field in (
                "pod_to_restore_start_s", "criu_restore_s", "cuda_restore_s",
                "prepare_s", "sleep_s", "wake_s",
            )))

    def test_restore_start_allows_at_most_five_seconds_of_agent_clock_skew_before_pod_creation(self):
        checkpoint_id = "h-" + "a" * 61
        class SkewedRestore(FixtureTransport):
            def __init__(self, start_timestamp):
                super().__init__()
                self.start_timestamp = start_timestamp

            def __call__(self, argv, timeout_s=None):
                result = super().__call__(argv, timeout_s)
                command = tuple(map(str, argv))
                if command[:6] == ("kubectl", "-n", NAMESPACE, "get", "pod", POD):
                    pod = json.loads(result.stdout)
                    pod["status"]["containerStatuses"] = [{"name": "server", "containerID": "containerd://current"}]
                    return self.ok(command, json.dumps(pod))
                if " logs " in f" {' '.join(command)} ":
                    return self.ok(command, "\n".join((
                        '%s\tINFO snapshot-agent\t=== Starting external restore ===\t{"pod":"v2-live-test/v2-01-01","container_id":"current","checkpoint_id":"%s"}' % (self.start_timestamp, checkpoint_id),
                        '2026-08-14T00:00:01Z Restore timing summary {"pod":"v2-live-test/v2-01-01","container_id":"current","checkpoint_id":"%s","restore":{"phases":{"criu_restore_duration":"2.5s","cuda_duration":"1.5s"}}}' % checkpoint_id,
                    )))
                return result

        with tempfile.TemporaryDirectory() as directory:
            collector = self.collector(directory, SkewedRestore("2026-08-13T23:59:58Z"))
            collector.prepare(self.run, "restore", POD)
            self.assertEqual(collector(self.run, "restore", POD)["metrics"]["criu_restore_s"], 2.5)

        with tempfile.TemporaryDirectory() as directory:
            collector = self.collector(directory, SkewedRestore("2026-08-13T23:59:54Z"))
            collector.prepare(self.run, "restore", POD)
            with self.assertRaises(ValueError):
                collector(self.run, "restore", POD)

    def test_restore_waits_for_agent_summary_and_fails_closed_on_deadline_or_later_fault(self):
        checkpoint_id = "h-" + "a" * 61
        start = '2026-08-14T00:00:03Z\tINFO snapshot-agent\t=== Starting external restore ===\t{"pod":"v2-live-test/v2-01-01","container_id":"current","checkpoint_id":"%s"}' % checkpoint_id
        summary = '2026-08-14T00:00:04Z Restore timing summary {"pod":"v2-live-test/v2-01-01","container_id":"current","checkpoint_id":"%s","restore":{"phases":{"criu_restore_duration":"2.5s","cuda_duration":"1.5s"}}}' % checkpoint_id

        class DelayedLogs(FixtureTransport):
            def __init__(self, second_poll):
                super().__init__()
                self.second_poll = second_poll
                self.log_polls = 0
                self.since_values = []

            def __call__(self, argv, timeout_s=None):
                result = super().__call__(argv, timeout_s)
                command = tuple(map(str, argv))
                if command[:6] == ("kubectl", "-n", NAMESPACE, "get", "pod", POD):
                    pod = json.loads(result.stdout)
                    pod["status"]["containerStatuses"] = [{"name": "server", "containerID": "containerd://current"}]
                    return self.ok(command, json.dumps(pod))
                if " logs " in f" {' '.join(command)} ":
                    self.log_polls += 1
                    self.since_values.append(next(value for value in command if value.startswith("--since-time=")))
                    return self.ok(command, start if self.log_polls == 1 else self.second_poll)
                return result

        with tempfile.TemporaryDirectory() as directory:
            transport = DelayedLogs(start + "\n" + summary)
            collector = self.collector(directory, transport)
            collector.prepare(self.run, "restore", POD)
            self.assertEqual(collector(self.run, "restore", POD)["metrics"]["criu_restore_s"], 2.5)
            self.assertEqual(transport.log_polls, 2)
            self.assertEqual(len(set(transport.since_values)), 1)

        with tempfile.TemporaryDirectory() as directory:
            transport = DelayedLogs(start)
            collector = self.collector(directory, transport)
            collector.prepare(self.run, "restore", POD)
            clock = iter((0.0, 0.0, 0.0, 2.0))
            with mock.patch.object(self.module.time, "monotonic", side_effect=lambda: next(clock, 2.0)), mock.patch.object(self.module.time, "sleep"):
                with self.assertRaises(ValueError):
                    collector(self.run, "restore", POD)
            self.assertGreaterEqual(transport.log_polls, 2)

        with tempfile.TemporaryDirectory() as directory:
            transport = DelayedLogs("NVRM: Xid 79")
            collector = self.collector(directory, transport)
            collector.prepare(self.run, "restore", POD)
            with self.assertRaises(ValueError):
                collector(self.run, "restore", POD)
            self.assertEqual(transport.log_polls, 2)

    def test_restore_waits_for_completed_annotation_after_ready_and_rejects_deadline_or_uid_change(self):
        class AnnotationRace(FixtureTransport):
            def __init__(self, *, complete_on=None, replace_uid=False):
                super().__init__()
                self.complete_on = complete_on
                self.replace_uid = replace_uid

            def __call__(self, argv, timeout_s=None):
                result = super().__call__(argv, timeout_s)
                command = tuple(map(str, argv))
                if command[:6] == ("kubectl", "-n", NAMESPACE, "get", "pod", POD):
                    pod = json.loads(result.stdout)
                    if self.pod_polls >= 2:
                        pod["metadata"]["annotations"] = {}
                    if self.complete_on is not None and self.pod_polls >= self.complete_on:
                        pod["metadata"]["annotations"] = {"nvidia.com/snapshot-restore-status.server": "completed"}
                    if self.replace_uid and self.pod_polls >= 3:
                        pod["metadata"]["uid"] = "uid-replaced-after-ready"
                    return self.ok(command, json.dumps(pod))
                return result

        with tempfile.TemporaryDirectory() as directory:
            transport = AnnotationRace(complete_on=3)
            collector = self.collector(directory, transport)
            collector.prepare(self.run, "restore", POD)
            self.assertTrue(collector(self.run, "restore", POD)["restore_success"])
            self.assertGreaterEqual(transport.pod_polls, 3)

        with tempfile.TemporaryDirectory() as directory:
            transport = AnnotationRace()
            collector = self.collector(directory, transport)
            collector.prepare(self.run, "restore", POD)
            clock = iter((0.0, 0.0, 0.0, 2.0))
            with mock.patch.object(self.module.time, "monotonic", side_effect=lambda: next(clock, 2.0)), mock.patch.object(self.module.time, "sleep"):
                with self.assertRaises(ValueError):
                    collector(self.run, "restore", POD)

        with tempfile.TemporaryDirectory() as directory:
            transport = AnnotationRace(replace_uid=True)
            collector = self.collector(directory, transport)
            collector.prepare(self.run, "restore", POD)
            with self.assertRaises(ValueError):
                collector(self.run, "restore", POD)

    def test_restore_events_are_collected_only_after_completed_annotation_on_same_uid(self):
        class OrderedRestoreEvents(FixtureTransport):
            def __init__(self):
                super().__init__()
                self.completed_seen = False

            def __call__(self, argv, timeout_s=None):
                result = super().__call__(argv, timeout_s)
                command = tuple(map(str, argv))
                if command[:6] == ("kubectl", "-n", NAMESPACE, "get", "pod", POD):
                    pod = json.loads(result.stdout)
                    if self.pod_polls >= 2:
                        pod["metadata"]["annotations"] = {}
                    if self.pod_polls >= 3:
                        pod["metadata"]["annotations"] = {"nvidia.com/snapshot-restore-status.server": "completed"}
                        self.completed_seen = True
                    return self.ok(command, json.dumps(pod))
                if " get events " in f" {' '.join(command)} ":
                    if not self.completed_seen:
                        raise AssertionError("events queried before restore completion")
                    checkpoint_id = "h-" + "a" * 61
                    return self.ok(command, json.dumps({"items": [
                        {"involvedObject": {"uid": "uid-v2-01-01", "name": POD, "namespace": NAMESPACE},
                         "reason": "RestoreRequested", "message": "Restore requested from checkpoint %s for container server" % checkpoint_id},
                        {"involvedObject": {"uid": "uid-v2-01-01", "name": POD, "namespace": NAMESPACE},
                         "reason": "RestoreSucceeded", "message": "Restore completed from checkpoint %s" % checkpoint_id},
                    ]}))
                return result

        with tempfile.TemporaryDirectory() as directory:
            transport = OrderedRestoreEvents()
            collector = self.collector(directory, transport)
            collector.prepare(self.run, "restore", POD)
            observation = collector(self.run, "restore", POD)
            self.assertTrue(observation["restore_success"])
            raw_events = pathlib.Path(directory) / "artifacts" / observation["raw_events_ref"]
            self.assertIn("RestoreSucceeded", raw_events.read_text())


class ClusterPreflightTransport:
    """Offline reservation-state fixture for preflight only."""

    node = "node-a"
    reserves = (
        ("gpu-reserve-1", "reserve-uid-1", "registry.example/reserve@sha256:" + "1" * 64, "GPU-1"),
        ("gpu-reserve-2", "reserve-uid-2", "registry.example/reserve@sha256:" + "2" * 64, "GPU-2"),
        ("gpu-reserve-3", "reserve-uid-3", "registry.example/reserve@sha256:" + "3" * 64, "GPU-3"),
    )

    def __init__(self, *, fault=None):
        self.calls = []
        self.fault = fault

    @staticmethod
    def ok(command, stdout):
        return subprocess.CompletedProcess(command, 0, stdout, "")

    def _pod(self, name):
        reserve = next((item for item in self.reserves if item[0] == name), None)
        if reserve is not None:
            _, uid, image, uuid = reserve
            if self.fault == "reserve-uid" and name == "gpu-reserve-1":
                uid = "unexpected-reserve-uid"
            if self.fault == "reserve-image" and name == "gpu-reserve-1":
                image = "registry.example/reserve:latest"
            node = self.node if self.fault != "reserve-node" else "node-b"
            ready = "False" if self.fault == "reserve-not-ready" else "True"
            gpu = "0" if self.fault == "reserve-gpu" else "1"
            return {
                "metadata": {"uid": uid},
                "spec": {"nodeName": node, "containers": [{"name": "other" if self.fault == "reserve-container" else "server", "image": image,
                    "resources": {"requests": {"nvidia.com/gpu": gpu}, "limits": {"nvidia.com/gpu": gpu}}}]},
                "status": {"phase": "Running", "conditions": [{"type": "Ready", "status": ready}]},
            }
        if name == AGENT:
            return {
                "metadata": {"uid": "agent-uid" if self.fault != "agent-uid" else "other-agent-uid"},
                "spec": {"nodeName": self.node if self.fault != "agent-node" else "node-b"},
                "status": {"phase": "Running", "conditions": [{"type": "Ready", "status": "False" if self.fault == "agent-not-ready" else "True"}],
                           "containerStatuses": [{"name": "agent", "imageID":
                               "registry.example/snapshot-agent@sha256:" + "a" * 64
                               if self.fault != "agent-image" else "registry.example/snapshot-agent:latest"}]},
            }
        return None

    def __call__(self, argv, timeout_s=None):
        command = tuple(map(str, argv))
        self.calls.append((command, timeout_s))
        text = " ".join(command)
        if self.fault == "transport":
            return subprocess.CompletedProcess(command, 1, "", "transport fixture failure")
        if command[:3] == ("kubectl", "get", "node"):
            ready = "False" if self.fault == "node-not-ready" else "True"
            gpus = "3" if self.fault == "node-gpus" else "4"
            spec = {"unschedulable": True} if self.fault == "node-unschedulable" else {}
            return self.ok(command, json.dumps({
                "spec": spec,
                "status": {"conditions": [{"type": "Ready", "status": ready}],
                           "capacity": {"nvidia.com/gpu": gpus}, "allocatable": {"nvidia.com/gpu": gpus}},
            }))
        if command[:5] == ("kubectl", "-n", NAMESPACE, "get", "pod"):
            pod = self._pod(command[5])
            if pod is not None:
                return self.ok(command, json.dumps(pod))
        if command[:4] == ("kubectl", "get", "pods", "-A"):
            pods = [self._pod(name) | {"metadata": self._pod(name)["metadata"] | {"name": name}}
                    for name, *_ in self.reserves]
            pods.append(self._pod(AGENT) | {"metadata": self._pod(AGENT)["metadata"] | {"name": AGENT}})
            pods.append({"metadata": {"name": "completed-non-gpu", "namespace": "default"},
                         "spec": {"nodeName": self.node, "containers": [{"resources": {}}]},
                         "status": {"phase": "Succeeded"}})
            if self.fault == "v2-pod":
                pods.append({"metadata": {"name": "v2-leaked", "labels": {"poc.regolo.ai/lane": "v2-a"}},
                             "spec": {"nodeName": self.node}, "status": {"phase": "Running"}})
            if self.fault == "extra-consumer":
                pods.append({"metadata": {"name": "foreign-gpu", "namespace": "other"}, "spec": {"nodeName": self.node, "containers": [{"resources": {"requests": {"nvidia.com/gpu": "1"}, "limits": {"nvidia.com/gpu": "1"}}}]}, "status": {"phase": "Pending"}})
            return self.ok(command, json.dumps({"items": pods}))
        if command[:4] == ("kubectl", "-n", NAMESPACE, "logs") and command[-1] == "--tail=1":
            reserve = next((item for item in self.reserves if item[0] == command[4]), None)
            if reserve is not None:
                uuid = reserve[3] if self.fault != "reserve-log-uuid" else "GPU-outside"
                return self.ok(command, "GPU UUID: " + uuid + "\n")
        if "nvidia-smi" in text and "--query-gpu=uuid" in text:
            uuids = ("GPU-1", "GPU-2", "GPU-3", "GPU-4")
            if self.fault == "duplicate-uuid":
                uuids = ("GPU-1", "GPU-1", "GPU-3", "GPU-4")
            return self.ok(command, "\n".join(uuids) + "\n")
        if "nvidia-smi" in text and "--query-compute-apps=gpu_uuid" in text:
            values = ("GPU-1", "GPU-2", "GPU-3")
            if self.fault == "compute-empty":
                values = ()
            if self.fault == "consumer-uuid":
                values = ("GPU-1", "GPU-2", "GPU-4")
            return self.ok(command, "\n".join(values) + "\n")
        return self.ok(command, "")


class ProductionClusterPreflightTests(unittest.TestCase):
    def setUp(self):
        self.module = load_production()
        self.reserves = tuple({"name": name, "uid": uid, "image": image, "node": ClusterPreflightTransport.node, "container": "server",
                               "gpu_uuid": uuid} for name, uid, image, uuid in ClusterPreflightTransport.reserves)
        self.agent = {"name": AGENT, "uid": "agent-uid", "image": "registry.example/snapshot-agent@sha256:" + "a" * 64,
                      "node": ClusterPreflightTransport.node}

    def preflight(self, transport):
        return self.module.ProductionClusterPreflight(
            NAMESPACE, ClusterPreflightTransport.node, self.reserves, self.agent, transport, timeout_s=17
        )

    def test_cluster_preflight_accepts_only_exact_four_gpu_reserved_cluster_state(self):
        for fault in (None, "compute-empty"):
            with self.subTest(compute_apps=fault or "reserve-subset"):
                transport = ClusterPreflightTransport(fault=fault)
                self.assertTrue(self.preflight(transport)())
                calls = [command for command, _ in transport.calls]
                self.assertTrue(any(command[:3] == ("kubectl", "get", "node") for command in calls))
                self.assertIn(("kubectl", "get", "pods", "-A", "-o", "json"), calls)
                for reserve in self.reserves:
                    self.assertTrue(any(command[:6] == ("kubectl", "-n", NAMESPACE, "get", "pod", reserve["name"]) for command in calls))
                    self.assertIn(("kubectl", "-n", NAMESPACE, "logs", reserve["name"], "--tail=1"), calls)
                self.assertTrue(any(command[:6] == ("kubectl", "-n", NAMESPACE, "get", "pod", AGENT) for command in calls))
                self.assertTrue(any("--query-gpu=uuid" in command for command in calls))
                self.assertTrue(any("--query-compute-apps=gpu_uuid" in command for command in calls))

    def test_cluster_preflight_fails_closed_on_capacity_reservation_agent_consumer_or_uuid_drift(self):
        faults = (
            "node-not-ready", "node-unschedulable", "node-gpus", "reserve-uid", "reserve-image", "reserve-node",
            "reserve-not-ready", "reserve-gpu", "reserve-container", "agent-uid", "agent-image", "agent-node", "agent-not-ready",
            "extra-consumer", "v2-pod", "duplicate-uuid", "reserve-log-uuid", "consumer-uuid", "transport",
        )
        for fault in faults:
            with self.subTest(fault=fault):
                with self.assertRaises(ValueError):
                    self.preflight(ClusterPreflightTransport(fault=fault))()


LIVE_CHECKPOINT_ID = "h-" + "a" * 61
LIVE_CHECKPOINT_ROOT = "/checkpoints/" + LIVE_CHECKPOINT_ID + "/versions/1"
LIVE_PV_LOCAL_PATH = "/mnt/regolo-vllm-snapshot-luks/checkpoints"
LIVE_MANIFEST_SIZE = 8740


class CheckpointValidatorTransport:
    """Small, executable-looking fixture for attestation validation only."""

    def __init__(self, *, fault=None):
        self.calls = []
        self.fault = fault
        self.checkpoint_id = LIVE_CHECKPOINT_ID
        prefix = f"checkpointId: {self.checkpoint_id}\ncreatedAt: 2026-08-14T00:00:00Z\n"
        self.manifest = prefix + "#" * (LIVE_MANIFEST_SIZE - len(prefix))
        self.manifest_sha256 = hashlib.sha256(self.manifest.encode()).hexdigest()

    def __call__(self, argv, timeout_s=None):
        command = tuple(map(str, argv))
        self.calls.append((command, timeout_s))
        if self.fault == "transport":
            return subprocess.CompletedProcess(command, 1, "", "fixture transport failure")
        text = " ".join(command)
        if command[:6] == ("kubectl", "-n", NAMESPACE, "get", "pod", AGENT):
            value = {
                "metadata": {"uid": "agent-uid" if self.fault != "agent-uid" else "other-agent-uid"},
                "spec": {"nodeName": "node-a" if self.fault != "agent-node" else "node-b"},
                "status": {"containerStatuses": [{
                    "name": "agent",
                    "imageID": "registry.example/snapshot-agent@sha256:" + "b" * 64
                    if self.fault != "agent-image" else "registry.example/snapshot-agent:latest",
                }]},
            }
            return self.ok(command, json.dumps(value))
        if command[:6] == ("kubectl", "-n", NAMESPACE, "get", "pvc", "checkpoint-pvc"):
            value = {
                "metadata": {"uid": "pvc-uid" if self.fault != "pvc-uid" else "other-pvc-uid"},
                "spec": {"volumeName": "checkpoint-pv" if self.fault != "pvc-pv" else "other-pv"},
            }
            return self.ok(command, json.dumps(value))
        if command[:4] == ("kubectl", "get", "pv", "checkpoint-pv"):
            value = {
                "metadata": {"uid": "pv-uid" if self.fault != "pv-uid" else "other-pv-uid"},
                "spec": {
                    "persistentVolumeReclaimPolicy": "Retain" if self.fault != "reclaim" else "Delete",
                    "local": {"path": LIVE_PV_LOCAL_PATH if self.fault != "pv-path" else "/other"},
                    "claimRef": {"uid": "pvc-uid" if self.fault != "claim-uid" else "other-pvc-uid"},
                    "nodeAffinity": {"required": {"nodeSelectorTerms": [{"matchExpressions": [{
                        "key": "kubernetes.io/hostname", "operator": "In",
                        "values": ["node-a" if self.fault != "pv-node" else "node-b"],
                    }]}]}},
                },
            }
            return self.ok(command, json.dumps(value))
        if " stat " in f" {text} ":
            path = command[-1]
            values = {
                LIVE_CHECKPOINT_ROOT + "/pages-12.img": ("regular file", 300),
                LIVE_CHECKPOINT_ROOT + "/rootfs-diff.tar": ("regular file", 80),
                LIVE_CHECKPOINT_ROOT + "/manifest.yaml": ("regular file", LIVE_MANIFEST_SIZE),
            }
            kind, size = values.get(path, ("missing", 0))
            if self.fault == "symlink" and path.endswith("pages-12.img"):
                kind = "symbolic link"
            if self.fault == "nonregular" and path.endswith("rootfs-diff.tar"):
                kind = "directory"
            if self.fault == "size" and path.endswith("rootfs-diff.tar"):
                size += 1
            return self.ok(command, f"{kind} {size}\n")
        if " sha256sum " in f" {text} ":
            digest = self.manifest_sha256 if self.fault != "manifest-hash" else "0" * 64
            return self.ok(command, f"{digest}  {LIVE_CHECKPOINT_ROOT}/manifest.yaml\n")
        if " du " in f" {text} ":
            total = 9120 if self.fault != "du" else 9121
            return self.ok(command, f"{total}\t{LIVE_CHECKPOINT_ROOT}\n")
        if "find" in command:
            return self.ok(command, "f|pages-12.img|300\nf|rootfs-diff.tar|80\nf|manifest.yaml|8740\n")
        if command[-2:] == ("cat", LIVE_CHECKPOINT_ROOT + "/manifest.yaml"):
            manifest = self.manifest
            if self.fault == "manifest-id":
                manifest = "checkpointId: h-" + "0" * 61 + "\ncreatedAt: 2026-08-14T00:00:00Z\n"
            return self.ok(command, manifest)
        return self.ok(command, "")

    @staticmethod
    def ok(command, stdout):
        return subprocess.CompletedProcess(command, 0, stdout, "")


class ProductionCheckpointValidatorTests(unittest.TestCase):
    def setUp(self):
        self.module = load_production()

    @staticmethod
    def attestation(transport):
        inventory = [
            {"path": "manifest.yaml", "size": LIVE_MANIFEST_SIZE},
            {"path": "pages-12.img", "size": 300},
            {"path": "rootfs-diff.tar", "size": 80},
        ]
        return {
            "checkpoint": {
                "id": transport.checkpoint_id,
                "compatibility_hash": "a" * 64,
                "location": LIVE_CHECKPOINT_ROOT,
                "total_size_bytes": 9120,
                "pages_12_size_bytes": 300,
                "rootfs_size_bytes": 80,
                "metadata_size_bytes": LIVE_MANIFEST_SIZE,
                "manifest_sha256": transport.manifest_sha256,
                "inventory": {
                    "regular_file_count": 3,
                    "regular_file_size_bytes": 9120,
                    "inventory_sha256": hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                },
            },
            "agent": {
                "namespace": NAMESPACE,
                "name": AGENT,
                "uid": "agent-uid",
                "image": "registry.example/snapshot-agent@sha256:" + "b" * 64,
                "node": "node-a",
            },
            "pvc": {"name": "checkpoint-pvc", "uid": "pvc-uid", "pv": "checkpoint-pv"},
            "pv": {
                "uid": "pv-uid", "local_path": LIVE_PV_LOCAL_PATH, "claim_uid": "pvc-uid",
                "node": "node-a", "reclaim_policy": "Retain",
            },
        }

    def validator(self, transport):
        return self.module.ProductionCheckpointValidator(NAMESPACE, AGENT, transport, timeout_s=17)

    def attestation_file(self, directory, transport, value=None):
        path = pathlib.Path(directory) / "checkpoint-attestation.json"
        path.write_text(json.dumps(self.attestation(transport) if value is None else value, sort_keys=True))
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_checkpoint_validator_accepts_only_exact_attestation_and_binds_all_remote_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = CheckpointValidatorTransport()
            path, digest = self.attestation_file(directory, transport)
            self.assertTrue(self.validator(transport)(path, digest))
            calls = [command for command, _ in transport.calls]
            self.assertTrue(any(command[:6] == ("kubectl", "-n", NAMESPACE, "get", "pod", AGENT) for command in calls))
            self.assertTrue(any(command[:6] == ("kubectl", "-n", NAMESPACE, "get", "pvc", "checkpoint-pvc") for command in calls))
            self.assertTrue(any(command[:4] == ("kubectl", "get", "pv", "checkpoint-pv") for command in calls))
            for path in (
                LIVE_CHECKPOINT_ROOT + "/pages-12.img", LIVE_CHECKPOINT_ROOT + "/rootfs-diff.tar",
                LIVE_CHECKPOINT_ROOT + "/manifest.yaml",
            ):
                self.assertTrue(any("stat" in command and path in command for command in calls), path)
            self.assertTrue(any("du" in command and LIVE_CHECKPOINT_ROOT in command for command in calls))
            self.assertTrue(any(command[-2:] == ("cat", LIVE_CHECKPOINT_ROOT + "/manifest.yaml") for command in calls))
            hashes = [command for command in calls if "sha256sum" in command]
            self.assertEqual(len(hashes), 1)
            self.assertIn(LIVE_CHECKPOINT_ROOT + "/manifest.yaml", hashes[0])
            self.assertFalse(any(path in hashes[0] for path in (
                LIVE_CHECKPOINT_ROOT + "/pages-12.img", LIVE_CHECKPOINT_ROOT + "/rootfs-diff.tar",
            )))

    def test_checkpoint_validator_fails_closed_on_remote_identity_and_checkpoint_integrity_mismatch(self):
        faults = (
            "agent-uid", "agent-image", "agent-node", "pvc-uid", "pvc-pv", "pv-uid", "pv-path",
            "claim-uid", "pv-node", "reclaim", "symlink", "nonregular", "size", "manifest-hash",
            "du", "manifest-id", "transport",
        )
        for fault in faults:
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                transport = CheckpointValidatorTransport(fault=fault)
                path, digest = self.attestation_file(directory, transport)
                with self.assertRaises(ValueError):
                    self.validator(transport)(path, digest)

    def test_checkpoint_validator_fails_closed_on_malformed_or_digest_mismatched_attestation_before_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = CheckpointValidatorTransport()
            path, digest = self.attestation_file(directory, transport)
            with self.assertRaises(ValueError):
                self.validator(transport)(path, "0" * 64)
            self.assertEqual(transport.calls, [])

        for value in (
            {},
            {"checkpoint": self.attestation(CheckpointValidatorTransport())["checkpoint"]},
            {**self.attestation(CheckpointValidatorTransport()), "unexpected": True},
        ):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                transport = CheckpointValidatorTransport()
                path, digest = self.attestation_file(directory, transport, value)
                with self.assertRaises(ValueError):
                    self.validator(transport)(path, digest)
                self.assertEqual(transport.calls, [])

    def test_checkpoint_inventory_binds_every_immediate_regular_file_by_one_canonical_listing(self):
        class InventoryTransport(CheckpointValidatorTransport):
            listing = (
                "f|pages-12.img|300\n"
                "f|rootfs-diff.tar|80\n"
                "f|manifest.yaml|8740\n"
            )

            def __call__(self, argv, timeout_s=None):
                command = tuple(map(str, argv))
                if "find" in command:
                    self.calls.append((command, timeout_s))
                    return self.ok(command, self.listing)
                return super().__call__(argv, timeout_s)

        with tempfile.TemporaryDirectory() as directory:
            transport = InventoryTransport()
            attestation = self.attestation(transport)
            inventory = [
                {"path": "manifest.yaml", "size": 8740},
                {"path": "pages-12.img", "size": 300},
                {"path": "rootfs-diff.tar", "size": 80},
            ]
            canonical = json.dumps(sorted(inventory, key=lambda item: item["path"]), sort_keys=True, separators=(",", ":")).encode()
            attestation["checkpoint"]["inventory"] = {
                "regular_file_count": 3,
                "regular_file_size_bytes": 9120,
                "inventory_sha256": hashlib.sha256(canonical).hexdigest(),
            }
            path, digest = self.attestation_file(directory, transport, attestation)
            self.assertTrue(self.validator(transport)(path, digest))
            calls = [command for command, _ in transport.calls]
            find = next(command for command in calls if "find" in command)
            self.assertIn("-mindepth", find)
            self.assertIn("-maxdepth", find)
            self.assertIn("-printf", find)

        for listing in (
            "l|pages-12.img|300\n", "f|nested/name|1\n", "f|pages-12.img|300\nf|extra.img|1\n",
        ):
            with self.subTest(listing=listing), tempfile.TemporaryDirectory() as directory:
                transport = InventoryTransport()
                transport.listing = listing
                attestation = self.attestation(transport)
                attestation["checkpoint"]["inventory"] = {
                    "regular_file_count": 3, "regular_file_size_bytes": 9120,
                    "inventory_sha256": "0" * 64,
                }
                path, digest = self.attestation_file(directory, transport, attestation)
                with self.assertRaises(ValueError):
                    self.validator(transport)(path, digest)

    def test_checkpoint_inventory_is_mandatory_before_remote_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = CheckpointValidatorTransport()
            attestation = self.attestation(transport)
            attestation["checkpoint"].pop("inventory")
            path, digest = self.attestation_file(directory, transport, attestation)
            with self.assertRaises(ValueError):
                self.validator(transport)(path, digest)
            self.assertEqual(transport.calls, [])

        for field, value in (("location", "/other-checkpoint"), ("id", "h-" + "0" * 61)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                transport = CheckpointValidatorTransport()
                attestation = self.attestation(transport)
                attestation["checkpoint"][field] = value
                path, digest = self.attestation_file(directory, transport, attestation)
                with self.assertRaises(ValueError):
                    self.validator(transport)(path, digest)


class CacheAdvisorTests(unittest.TestCase):
    def setUp(self):
        self.module = load_production()

    def test_remote_allowlisted_paths_are_agent_stat_validated_and_storage_uses_four_bounded_dd_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = FixtureTransport()
            advisor = self.module.CacheAdvisor(NAMESPACE, AGENT, ROOT, transport, timeout_s=11)
            advisor.advise([PAGES])
            advice = [command for command, _ in transport.calls if "dd" in command]
            self.assertEqual(len(advice), 1)
            self.assertEqual(advice[0][advice[0].index("dd") + 1:], (f"if={PAGES}", "of=/dev/null", "iflag=nocache", "count=0", "status=none"))
            result = advisor.characterize_storage([PAGES], max_bytes=4, max_reads=3, artifact_dir=pathlib.Path(directory) / "artifacts")
            self.assertEqual([row["mode"] for row in result["reads"]], ["buffered-first", "buffered-repeat", "direct"])
            self.assertEqual(len(result["reads"]), 3)
            self.assertTrue(all(row["bytes"] <= 4 for row in result["reads"]))
            self.assertTrue((pathlib.Path(directory) / "artifacts" / result["raw_ref"]).is_file())
            calls = [command for command, _ in transport.calls]
            self.assertTrue(any("stat" in command and PAGES in command for command in calls))
            dd = [command for command in calls if "dd" in command][1:]
            self.assertEqual(len(dd), 4)
            self.assertIn("iflag=nocache", dd[0])
            self.assertIn("count=0", dd[0])
            self.assertEqual(dd[1][dd[1].index("dd") + 1:], (f"if={PAGES}", "of=/dev/null", "iflag=count_bytes", "count=4", "status=none"))
            self.assertEqual(dd[2][dd[2].index("dd") + 1:], (f"if={PAGES}", "of=/dev/null", "iflag=count_bytes", "count=4", "status=none"))
            self.assertEqual(dd[3][dd[3].index("dd") + 1:], (f"if={PAGES}", "of=/dev/null", "bs=4M", "iflag=direct,count_bytes", "count=4", "status=none"))
            self.assertTrue(all(f"if={PAGES}" in command and "of=/dev/null" in command for command in dd))
            for command in advice + dd:
                self.assertEqual(sum(value.startswith("iflag=") for value in command), 1)
                self.assertEqual(sum(value.startswith("count=") for value in command), 1)
            self.assertTrue(all(row["wall_s"] > 0 and row["throughput_bytes_s"] > 0 for row in result["reads"]))
            with self.assertRaises(ValueError):
                advisor.advise([ROOT + "/../escape.img"])

    def test_advise_inventory_reenumerates_and_advises_every_attested_regular_file_before_restore(self):
        class InventoryAdvice(FixtureTransport):
            def __init__(self, listing="f|b.img|2\nf|a.img|1\nf|manifest.yaml|3\n"):
                super().__init__()
                self.listing = listing

            def __call__(self, argv, timeout_s=None):
                command = tuple(map(str, argv))
                if "find" in command:
                    self.calls.append((command, timeout_s))
                    return self.ok(command, self.listing)
                if "stat" in command:
                    sizes = {ROOT + "/a.img": 1, ROOT + "/b.img": 2, ROOT + "/manifest.yaml": 3}
                    return self.ok(command, "regular file %d\n" % sizes[command[-1]])
                return super().__call__(argv, timeout_s)

        canonical = json.dumps(
            [{"path": "a.img", "size": 1}, {"path": "b.img", "size": 2}, {"path": "manifest.yaml", "size": 3}],
            sort_keys=True, separators=(",", ":"),
        ).encode()
        inventory = {"regular_file_count": 3, "regular_file_size_bytes": 6, "inventory_sha256": hashlib.sha256(canonical).hexdigest()}
        transport = InventoryAdvice()
        advisor = self.module.CacheAdvisor(NAMESPACE, AGENT, ROOT, transport, timeout_s=11)
        advisor.advise_inventory(inventory)
        dd = [command for command, _ in transport.calls if "dd" in command]
        self.assertEqual([command[command.index("dd") + 1] for command in dd], [
            "if=" + ROOT + "/a.img", "if=" + ROOT + "/b.img", "if=" + ROOT + "/manifest.yaml",
        ])
        self.assertTrue(all("iflag=nocache" in command and "count=0" in command for command in dd))

        for listing in ("l|a.img|1\n", "f|../escape|1\n", "d|nested|1\n"):
            with self.subTest(listing=listing):
                transport = InventoryAdvice(listing)
                with self.assertRaises(ValueError):
                    self.module.CacheAdvisor(NAMESPACE, AGENT, ROOT, transport, timeout_s=11).advise_inventory(inventory)
                self.assertFalse(any("dd" in command for command, _ in transport.calls))

    def test_characterization_captures_host_deltas_per_read_and_seals_private_raw_evidence(self):
        class StorageTelemetry(FixtureTransport):
            def __init__(self, *, regression=False, missing_device=False, malformed=False):
                super().__init__()
                self.snapshot = 0
                self.regression = regression
                self.missing_device = missing_device
                self.malformed = malformed

            def __call__(self, argv, timeout_s=None):
                command = tuple(map(str, argv))
                text = " ".join(command)
                if "/host/proc/meminfo" in text:
                    self.snapshot += 1
                    if self.malformed:
                        return self.ok(command, "MemAvailable: invalid kB\n")
                    return self.ok(command, "MemAvailable: %d kB\nCached: %d kB\nSReclaimable: 20 kB\nShmem: 5 kB\n" % (100 - self.snapshot, 30 + self.snapshot))
                if "/host/proc/pressure/io" in text:
                    return self.ok(command, "some avg10=0.00 avg60=0.00 avg300=0.00 total=%d\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=%d\n" % (self.snapshot, self.snapshot))
                if "/host/proc/diskstats" in text:
                    value = 100 - self.snapshot if self.regression else 100 + self.snapshot
                    lines = [
                        "253 0 dm-0 1 0 %d 1 0 0 0 0 0 0 0 0" % value,
                        "7 6 loop6 1 0 %d 1 0 0 0 0 0 0 0 0" % value,
                    ]
                    if not self.missing_device:
                        lines.append("8 0 sda 1 0 %d 1 0 0 0 0 0 0 0 0" % value)
                    return self.ok(command, "\n".join(lines) + "\n")
                if "/sys/fs/cgroup/io.stat" in text:
                    value = 100 - self.snapshot if self.regression else 100 + self.snapshot
                    return self.ok(command, "253:0 rbytes=%d wbytes=0\n" % value)
                return super().__call__(argv, timeout_s)

        with tempfile.TemporaryDirectory() as directory:
            transport = StorageTelemetry()
            advisor = self.module.CacheAdvisor(NAMESPACE, AGENT, ROOT, transport, timeout_s=11)
            artifact_dir = pathlib.Path(directory) / "artifacts"
            with mock.patch.object(self.module.os, "fsync", wraps=os.fsync) as fsync:
                result = advisor.characterize_storage([PAGES], max_bytes=4, max_reads=3, artifact_dir=artifact_dir)
            self.assertEqual(len(result["reads"]), 3)
            for row in result["reads"]:
                self.assertTrue({
                    "bytes", "wall_s", "throughput_bytes_s", "page_cache_delta_bytes",
                    "mem_available_delta_bytes", "psi_io_total_delta", "diskstats_delta", "cgroup_io_delta",
                }.issubset(row))
                self.assertGreater(row["wall_s"], 0)
                self.assertGreater(row["throughput_bytes_s"], 0)
                self.assertEqual(set(row["diskstats_delta"]), {"dm-0", "loop6", "sda"})
                self.assertEqual(set(row["psi_io_total_delta"]), {"some", "full"})
                self.assertTrue(all(value >= 0 for value in row["psi_io_total_delta"].values()))
            raw = artifact_dir / result["raw_ref"]
            self.assertEqual(stat.S_IMODE(raw.stat().st_mode), 0o600)
            self.assertGreaterEqual(fsync.call_count, 2)
            with self.assertRaises(ValueError):
                advisor.characterize_storage([PAGES], max_bytes=4, max_reads=3, artifact_dir=artifact_dir)

        for name, transport in (
            ("counter-regression", StorageTelemetry(regression=True)),
            ("missing-device", StorageTelemetry(missing_device=True)),
            ("malformed", StorageTelemetry(malformed=True)),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                advisor = self.module.CacheAdvisor(NAMESPACE, AGENT, ROOT, transport, timeout_s=11)
                with self.assertRaises(ValueError):
                    advisor.characterize_storage([PAGES], max_bytes=4, max_reads=3, artifact_dir=pathlib.Path(directory) / "artifacts")


if __name__ == "__main__":
    unittest.main()
