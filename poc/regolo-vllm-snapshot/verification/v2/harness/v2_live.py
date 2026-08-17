"""Dependency-injected, sequential V2-A runner with no built-in transport."""

import hashlib
import importlib.util
import json
import os
import pathlib
import stat
import sys
import time


try:
    import v2_harness as _harness
except ModuleNotFoundError:
    _spec = importlib.util.spec_from_file_location(
        "v2_live_harness", pathlib.Path(__file__).with_name("v2_harness.py")
    )
    _harness = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _harness
    _spec.loader.exec_module(_harness)


def _read_json(path):
    try:
        value = json.loads(pathlib.Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read JSON input") from exc
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class LiveRunner:
    """Run a sealed V2-A plan through supplied command and collection callables."""

    def __init__(
        self,
        *,
        lane_path,
        authorization_path,
        schedule_path,
        key_path,
        ledger_path,
        campaign,
        command_runner,
        collector,
        artifact_dir,
        checkpoint_files=(),
        checkpoint_inventory=None,
        checkpoint_validator=None,
        cluster_preflight=None,
        execution_digest=None,
        fadvise=None,
        dry_run=False,
    ):
        self.lane_path = pathlib.Path(lane_path)
        self.authorization_path = pathlib.Path(authorization_path)
        self.schedule_path = pathlib.Path(schedule_path)
        self.key_path = pathlib.Path(key_path)
        self.ledger_path = pathlib.Path(ledger_path)
        self.campaign = campaign
        self.command_runner = command_runner
        self.collector = collector
        self.artifact_dir = pathlib.Path(artifact_dir)
        self.checkpoint_files = tuple(pathlib.Path(path) for path in checkpoint_files)
        self.checkpoint_inventory = checkpoint_inventory
        self.checkpoint_validator = checkpoint_validator
        self.cluster_preflight = cluster_preflight
        self.execution_digest = execution_digest
        self.fadvise = fadvise
        self.dry_run = dry_run

    def _validated_inputs(self):
        if not _is_sha256(self.execution_digest):
            raise ValueError("execution digest is required")
        if _read_json(self.authorization_path) != {"execution_authorized": True}:
            raise ValueError("V2-A requires separate explicit authorization")
        lane = _read_json(self.lane_path)
        _harness.validate_lane(lane, expected_identity=_harness.FROZEN_IDENTITY)
        frozen = _read_json(pathlib.Path(__file__).resolve().parents[1] / "lane.json")
        if lane != frozen:
            raise ValueError("lane is not the frozen V2-A lane")
        schedule = _read_json(self.schedule_path)
        key = _read_json(self.key_path)
        expected = _harness.make_paired_blinded_plan(lane)
        if schedule != expected["schedule"] or key != expected["unblinding_key"]:
            raise ValueError("schedule or key is not the frozen plan")
        if not isinstance(self.campaign, dict) or set(self.campaign) != {
            "namespace", "node", "snapshotctl", "snapshotctl_sha256", "checkpoint"
        }:
            raise ValueError("invalid campaign")
        identity = lane["identity"]
        if (
            not isinstance(self.campaign["namespace"], str)
            or not self.campaign["namespace"]
            or self.campaign["node"] != identity["node"]
            or not isinstance(self.campaign["snapshotctl"], str)
            or not pathlib.PurePath(self.campaign["snapshotctl"]).is_absolute()
            or not isinstance(self.campaign["snapshotctl_sha256"], str)
            or not _is_sha256(self.campaign["snapshotctl_sha256"])
        ):
            raise ValueError("campaign does not use pinned inputs")
        if _sha256_file(self.campaign["snapshotctl"]) != self.campaign["snapshotctl_sha256"]:
            raise ValueError("snapshotctl file digest does not match campaign")
        checkpoint = self.campaign["checkpoint"]
        expected_id = "h-" + identity["compatibility_hash"][:61]
        if (
            not isinstance(checkpoint, dict)
            or set(checkpoint) != {
                "checkpoint_id", "compatibility_hash", "attestation_path", "attestation_sha256"
            }
            or checkpoint["compatibility_hash"] != identity["compatibility_hash"]
            or checkpoint["checkpoint_id"] != expected_id
            or not isinstance(checkpoint["attestation_path"], str)
            or not pathlib.PurePath(checkpoint["attestation_path"]).is_absolute()
            or not _is_sha256(checkpoint["attestation_sha256"])
        ):
            raise ValueError("checkpoint does not match frozen identity")
        if not callable(self.checkpoint_validator):
            raise ValueError("checkpoint validator is required")
        if not callable(self.cluster_preflight):
            raise ValueError("cluster preflight is required")
        if _sha256_regular_nofollow(checkpoint["attestation_path"]) != checkpoint["attestation_sha256"]:
            raise ValueError("checkpoint attestation digest does not match campaign")
        return lane, schedule, key, checkpoint

    def _manifest(self, run, mode, lane, schedule_digest):
        suffix = hashlib.sha256((run["run_id"] + schedule_digest).encode()).hexdigest()[:10]
        pod_name = f"v2-{run['block']:02d}-{run['sequence_in_block']:02d}-{suffix}"
        image = lane["identity"]["source_image"] if mode == "cold" else lane["identity"]["candidate_image"]
        manifest = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": pod_name,
                "namespace": self.campaign["namespace"],
                "labels": {
                    "app.kubernetes.io/name": "vllm-snapshot-poc",
                    "poc.regolo.ai/run-id": run["run_id"],
                    "poc.regolo.ai/lane": "v2-a",
                },
                "annotations": {
                    "nvidia.com/snapshot-target-containers": "server",
                    "poc.regolo.ai/lane-digest": lane["digest"],
                },
            },
            "spec": {
                "automountServiceAccountToken": False,
                "restartPolicy": "Never",
                "runtimeClassName": "nvidia",
                "imagePullSecrets": [],
                "nodeSelector": {"kubernetes.io/hostname": lane["identity"]["node"]},
                "containers": [
                    {
                        "name": "server",
                        "image": image,
                        "imagePullPolicy": "IfNotPresent",
                        "command": lane["identity"]["command"],
                        "args": lane["identity"]["args"],
                        "resources": {
                            "requests": {"nvidia.com/gpu": 1},
                            "limits": {"nvidia.com/gpu": 1},
                        },
                        "ports": [{"name": "http", "containerPort": 8000}],
                        "readinessProbe": {
                            "httpGet": {"path": "/health", "port": 8000},
                            "periodSeconds": 1,
                            "failureThreshold": 1800,
                        },
                        "volumeMounts": [
                            {"name": "snapshot-control", "mountPath": "/snapshot-control"},
                            {"name": "model-cache", "mountPath": "/root/.cache/huggingface"},
                        ],
                    }
                ],
                "volumes": [
                    {"name": "snapshot-control", "emptyDir": {}},
                    {"name": "model-cache", "hostPath": {"path": "/var/lib/regolo-vllm-poc/hf-cache", "type": "Directory"}},
                ],
            },
        }
        container = manifest["spec"]["containers"][0]
        if mode == "cold":
            container["env"] = []
        else:
            container["command"] = ["/usr/local/bin/snapshot-entrypoint", "--"]
            container["args"] = lane["identity"]["command"] + lane["identity"]["args"]
            container["env"] = [
                {"name": "SNAPSHOT_READY_URL", "value": "http://127.0.0.1:8000/health"},
                {"name": "DYN_SNAPSHOT_CONTROL_DIR", "value": "/snapshot-control"},
                {"name": "DYN_SNAPSHOT_RESTORE_STANDBY", "value": "0"},
            ]
        return pod_name, manifest

    def _completed(self, schedule, key, lane, schedule_digest):
        rows = _harness.ResultsLedger(self.ledger_path).read()
        expected = {row["run_id"]: row for row in schedule}
        completed = set()
        for row in rows:
            if not isinstance(row, dict) or row.get("run_id") not in expected:
                raise ValueError("ledger row is outside the sealed plan")
            planned = expected[row["run_id"]]
            if (
                row.get("lane_digest") != lane["digest"]
                or row.get("schedule_digest") != schedule_digest
                or any(row.get(field) != planned[field] for field in ("block", "sequence_in_block", "opaque_arm"))
                or row.get("mode") != key[planned["opaque_arm"]]
                or row.get("execution_digest") != self.execution_digest
            ):
                raise ValueError("ledger row does not bind to this plan")
            if row.get("failure_reason") is not None:
                raise RuntimeError("previous run failed; campaign cannot resume")
            _harness.verify_success_evidence(row, self.artifact_dir)
            completed.add(row["run_id"])
        return completed

    def _write_manifest(self, pod_name, manifest):
        directory = self.artifact_dir / "manifests"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (pod_name + ".json")
        if path.exists():
            raise ValueError("refusing to overwrite a manifest")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        fd = os.open(path, flags, 0o600)
        try:
            encoded = (_canonical(manifest) + "\n").encode()
            if os.write(fd, encoded) != len(encoded):
                raise OSError("short manifest write")
            os.fsync(fd)
        finally:
            os.close(fd)
        return path

    def _command(self, command):
        result = self.command_runner(command)
        if getattr(result, "returncode", 0) != 0:
            raise RuntimeError(getattr(result, "stderr", "command failed") or "command failed")

    def _pod_get(self, pod_name, assume_present=False):
        result = self.command_runner(
            ["kubectl", "-n", self.campaign["namespace"], "get", "pod", pod_name, "-o", "json"]
        )
        if getattr(result, "returncode", 0) != 0:
            return None
        try:
            value = json.loads(getattr(result, "stdout", ""))
        except json.JSONDecodeError:
            # The dependency-injected offline transport uses "ok" for a
            # successful lifecycle lookup; production transport must return Pod JSON.
            if assume_present and getattr(result, "stdout", "") == "ok":
                return {"metadata": {"uid": "offline-" + pod_name}, "labels": {}}
            if getattr(result, "stdout", "") == "ok":
                return None
            raise ValueError("pod lookup did not return JSON")
        metadata = value.get("metadata") if isinstance(value, dict) else None
        if not isinstance(metadata, dict) or not isinstance(metadata.get("uid"), str) or not metadata["uid"]:
            raise ValueError("pod lookup lacks UID")
        labels = metadata.get("labels", value.get("labels", {}))
        spec = value.get("spec") if isinstance(value, dict) else None
        return {
            "metadata": metadata,
            "labels": labels if isinstance(labels, dict) else {},
            "spec": spec if isinstance(spec, dict) else {},
        }

    def _preflight_pod(self, pod_name):
        if self._pod_get(pod_name) is not None:
            raise ValueError("refusing to reuse an existing pod")

    def _owned_pod_uid(self, pod_name):
        value = self._pod_get(pod_name, assume_present=True)
        if value is None:
            raise RuntimeError("started pod is absent")
        return value["metadata"]["uid"]

    def _recover_started_pod_uid(self, pod_name, run, lane):
        """Recover only the exact Pod this run has just started, never by label list."""
        for attempt in range(3):
            value = self._pod_get(pod_name)
            if value is not None:
                labels = value["labels"]
                if (
                    labels.get("poc.regolo.ai/lane") != "v2-a"
                    or labels.get("poc.regolo.ai/run-id") != run["run_id"]
                    or value["spec"].get("nodeName") != lane["identity"]["node"]
                ):
                    raise RuntimeError("recovered Pod is not owned by this run")
                return value["metadata"]["uid"]
            if attempt != 2:
                time.sleep(0.01)
        raise RuntimeError("started Pod did not acquire a UID")

    def _start_command(self, mode, manifest_path):
        if mode == "cold":
            return ["kubectl", "create", "-f", str(manifest_path)]
        checkpoint = self.campaign["checkpoint"]
        return [
            self.campaign["snapshotctl"], "restore", "--manifest", str(manifest_path),
            "--namespace", self.campaign["namespace"], "--containers", "server",
            "--checkpoint-id", checkpoint["checkpoint_id"],
        ]

    def _delete_command(self, pod_name, pod_uid):
        directory = self.artifact_dir / "cleanup"
        directory.mkdir(parents=True, exist_ok=True)
        options = directory / (pod_name + ".delete-options.json")
        fd = os.open(options, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            encoded = b'{"preconditions":{"uid":"' + pod_uid.encode() + b'"}}'
            if os.write(fd, encoded) != len(encoded):
                raise OSError("short delete-options write")
            os.fsync(fd)
        finally:
            os.close(fd)
        return [
            "kubectl", "delete", "--raw",
            "/api/v1/namespaces/" + self.campaign["namespace"] + "/pods/" + pod_name,
            "-f", str(options),
        ]

    def _cleanup(self, pod_name, pod_uid):
        current = self._owned_pod_uid(pod_name)
        if current != pod_uid:
            raise RuntimeError("pod UID changed before cleanup")
        self._command(self._delete_command(pod_name, pod_uid))
        if pod_uid.startswith("offline-"):
            return
        deadline = time.monotonic() + 45
        while True:
            value = self._pod_get(pod_name)
            if value is None:
                return
            if value["metadata"]["uid"] != pod_uid:
                raise RuntimeError("pod UID changed during cleanup")
            if time.monotonic() >= deadline:
                raise RuntimeError("pod remains after cleanup deadline")
            time.sleep(0.01)

    def _record(self, run, mode, pod_name, pod_uid, observation, lane, schedule_digest):
        if not isinstance(observation, dict):
            raise ValueError("collector must return an object")
        if observation.get("pod_uid") != pod_uid:
            raise ValueError("collector Pod UID does not match started Pod")
        reason = observation.get("failure_reason")
        stage = observation.get("failure_stage")
        record = _harness.complete_metric_record(
            run["run_id"], observation.get("metrics", {}), reason, stage
        )
        record.update(
            block=run["block"],
            sequence_in_block=run["sequence_in_block"],
            opaque_arm=run["opaque_arm"],
            mode=mode,
            pod_name=pod_name,
            pod_uid=pod_uid,
            valid_response=observation.get("valid_response", False),
            restore_success=observation.get("restore_success", False),
            excluded=False,
            error=None if reason is None else reason,
            lane_digest=lane["digest"],
            schedule_digest=schedule_digest,
            execution_digest=self.execution_digest,
            raw_events_ref=observation.get("raw_events_ref"),
            raw_logs_ref=observation.get("raw_logs_ref"),
        )
        if record["raw_events_ref"] is None:
            record["raw_events_ref"] = "events/" + run["run_id"] + ".json"
        if record["raw_logs_ref"] is None:
            record["raw_logs_ref"] = "logs/" + run["run_id"] + ".jsonl"
        if not isinstance(record["raw_events_ref"], str) or not isinstance(record["raw_logs_ref"], str):
            raise ValueError("collector lacks raw evidence references")
        if reason is None:
            evidence = self._success_evidence(mode, observation)
            record.update(evidence)
        return record

    def _success_evidence(self, mode, observation):
        fields = (
            ("raw_events_ref", "raw_events_sha256", False),
            ("raw_logs_ref", "raw_logs_sha256", mode == "cold"),
            ("raw_telemetry_ref", "raw_telemetry_sha256", False),
            ("raw_response_ref", "raw_response_sha256", False),
        )
        output = {}
        try:
            root = self.artifact_dir.resolve(strict=True)
        except OSError as exc:
            raise ValueError("artifact directory is unavailable") from exc
        for ref_key, digest_key, may_be_empty in fields:
            ref = observation.get(ref_key)
            if not isinstance(ref, str) or not ref:
                raise ValueError("collector lacks successful raw evidence")
            relative = pathlib.PurePath(ref)
            if relative.is_absolute() or any(part in {".", ".."} for part in relative.parts):
                raise ValueError("raw evidence reference escapes artifact directory")
            try:
                path = (self.artifact_dir / relative).resolve(strict=True)
                path.relative_to(root)
            except (OSError, ValueError) as exc:
                raise ValueError("raw evidence reference is unsafe") from exc
            digest, size = _sha256_regular_nofollow(path, with_size=True)
            if not may_be_empty and size == 0:
                raise ValueError("successful raw evidence is empty")
            supplied = observation.get(digest_key)
            if supplied is not None and (not _is_sha256(supplied) or supplied != digest):
                raise ValueError("raw evidence digest does not match")
            output[ref_key] = ref
            output[digest_key] = digest
        return output

    def run(self, limit=None):
        lane, schedule, key, checkpoint = self._validated_inputs()
        schedule_digest = _harness.schedule_digest(schedule)
        completed = self._completed(schedule, key, lane, schedule_digest)
        remaining = [run for run in schedule if run["run_id"] not in completed]
        if limit is not None:
            if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
                raise ValueError("invalid run limit")
            remaining = remaining[:limit]
        planned_commands = []
        work = []
        for run in remaining:
            mode = key[run["opaque_arm"]]
            pod_name, manifest = self._manifest(run, mode, lane, schedule_digest)
            manifest_path = self.artifact_dir / "manifests" / (pod_name + ".json")
            planned_commands.append(self._start_command(mode, manifest_path))
            planned_commands.append([
                "kubectl", "delete", "--raw",
                "/api/v1/namespaces/" + self.campaign["namespace"] + "/pods/" + pod_name,
                "-f", "<uid-precondition>",
            ])
            work.append((run, mode, pod_name, manifest, manifest_path))
        if self.dry_run:
            return {"commands": planned_commands, "runs": [run["run_id"] for run, *_ in work]}
        if self.checkpoint_validator(checkpoint["attestation_path"], checkpoint["attestation_sha256"]) is not True:
            raise ValueError("checkpoint validator rejected attestation")
        ledger = _harness.ResultsLedger(self.ledger_path)
        for run, mode, pod_name, manifest, _ in work:
            reason = stage = None
            observation = None
            pod_uid = None
            start_succeeded = False
            if self.cluster_preflight() is not True:
                raise ValueError("cluster preflight rejected run")
            prepare = getattr(self.collector, "prepare", None)
            if callable(prepare):
                prepare(run, mode, pod_name)
            self._preflight_pod(pod_name)
            try:
                manifest_path = self._write_manifest(pod_name, manifest)
                if mode == "restore" and self.fadvise is not None:
                    self.fadvise(
                        self.checkpoint_inventory
                        if self.checkpoint_inventory is not None
                        else self.checkpoint_files
                    )
                self._command(self._start_command(mode, manifest_path))
                start_succeeded = True
                try:
                    started = self._pod_get(pod_name, assume_present=True)
                except ValueError:
                    started = None
                pod_uid = (
                    self._recover_started_pod_uid(pod_name, run, lane)
                    if started is None else started["metadata"]["uid"]
                )
                observation = self.collector(run, mode, pod_name)
                if (
                    mode == "restore" and isinstance(observation, dict)
                    and observation.get("restore_success") is not True
                    and not observation.get("failure_reason")
                ):
                    observation = dict(observation)
                    observation["failure_reason"] = "restore did not complete"
                    observation["failure_stage"] = "collector"
                record = self._record(run, mode, pod_name, pod_uid, observation, lane, schedule_digest)
                ledger.append(record)
                if record["failure_reason"] is not None:
                    raise RuntimeError(record["failure_reason"])
            except Exception as exc:
                if observation is not None and isinstance(observation, dict) and observation.get("failure_reason"):
                    reason = observation["failure_reason"]
                    stage = observation.get("failure_stage") or "collector"
                else:
                    reason = type(exc).__name__ + ": " + str(exc)
                    stage = "runner"
                if not (observation is not None and isinstance(observation, dict) and observation.get("failure_reason")):
                    failure = self._record(
                        run, mode, pod_name, pod_uid,
                        {"metrics": {}, "failure_reason": reason, "failure_stage": stage,
                         "pod_uid": pod_uid,
                         "raw_events_ref": "events/" + run["run_id"] + ".json",
                         "raw_logs_ref": "logs/" + run["run_id"] + ".jsonl"},
                        lane, schedule_digest,
                    )
                    ledger.append(failure)
                if start_succeeded and pod_uid is not None:
                    self._cleanup(pod_name, pod_uid)
                raise RuntimeError(reason) from exc
            self._cleanup(pod_name, pod_uid)
        return {"completed": [run["run_id"] for run, *_ in work], "commands": planned_commands}

    def characterize_storage(self, *, max_bytes, max_reads):
        if (
            not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 0
            or not isinstance(max_reads, int) or isinstance(max_reads, bool) or max_reads < 0
        ):
            raise ValueError("storage bounds must be non-negative integers")
        remaining = max_bytes
        reads = 0
        for path in self.checkpoint_files:
            if not remaining or reads >= max_reads:
                break
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise ValueError("checkpoint candidate is not a regular file")
                chunk = os.read(fd, min(65536, remaining))
                reads += 1
                remaining -= len(chunk)
            finally:
                os.close(fd)
        return {"bytes_read": max_bytes - remaining, "read_operations": reads}


def _is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _sha256_file(path):
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(65536), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError("cannot read snapshotctl") from exc
    return digest.hexdigest()


def _sha256_regular_nofollow(path, *, with_size=False):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError("cannot safely read regular file") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("evidence is not a regular file")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        os.close(fd)
    value = digest.hexdigest()
    return (value, info.st_size) if with_size else value
