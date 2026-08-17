"""Local, fail-closed helpers for the proposed V2 verification campaign."""

import errno
import fcntl
import hashlib
import json
import math
import os
import pathlib
import random
import re
import stat
import statistics
import threading


REQUIRED_METRICS = frozenset(
    {
        "pod_to_scheduled_s",
        "pod_to_restore_start_s",
        "criu_restore_s",
        "cuda_restore_s",
        "ready_s",
        "http_200_s",
        "first_token_s",
        "cgroup_io_stat",
        "diskstats",
        "node_page_cache_bytes",
        "node_memory_available_bytes",
        "psi_cpu",
        "psi_io",
        "psi_memory",
        "node_cpu_utilization",
        "gpu_memory_mib",
        "checkpoint_size_bytes",
        "pages_12_size_bytes",
        "rootfs_size_bytes",
        "metadata_size_bytes",
        "prepare_s",
        "sleep_s",
        "wake_s",
        "admission_closed",
        "harness_inflight",
        "vllm_running",
        "vllm_waiting",
        "tokens_per_second",
        "token_after_restore_summary_s",
        "checkpoint_storage_read_bytes",
        "checkpoint_storage_read_throughput_bytes_s",
        "node_page_cache_delta_bytes",
        "node_memory_available_delta_bytes",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_RUN_ID = re.compile(r"^v2-[0-9]{2}(?:-[0-9]{2})?$")
_LOCKS = {}
_LOCKS_GUARD = threading.Lock()

FROZEN_IDENTITY = {
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


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_lane(value, expected_identity=None, authorization=None):
    """Validate the immutable identity and digest of a V2-A lane declaration."""
    if not isinstance(value, dict) or set(value) != {
        "baseline_group", "single_mutation", "seed", "identity", "workload", "digest"
    }:
        raise ValueError("invalid lane fields")
    if value["baseline_group"] != "v1-driver580-v3" or value["single_mutation"] != "observer_only_phase_collection":
        raise ValueError("unapproved lane")
    if not isinstance(value["seed"], int) or isinstance(value["seed"], bool) or value["seed"] < 1:
        raise ValueError("invalid lane seed")
    if expected_identity is not None and value["seed"] != 20260814:
        raise ValueError("V2-A seed is not the frozen seed")
    identity = value["identity"]
    anchor = FROZEN_IDENTITY if expected_identity is None else expected_identity
    if not isinstance(anchor, dict) or anchor != FROZEN_IDENTITY or identity != anchor:
        raise ValueError("identity does not match the frozen V2-A trust anchor")
    for field in ("source_image", "candidate_image"):
        if not re.fullmatch(r".+@sha256:[0-9a-f]{64}", identity[field]):
            raise ValueError("identity contains a mutable image reference")
    for field in ("baseline_protocol_sha256", "baseline_results_sha256", "baseline_phase_analysis_sha256", "compatibility_hash"):
        if not _SHA256.fullmatch(identity[field]):
            raise ValueError("identity contains a noncanonical digest")
    if not _REVISION.fullmatch(identity["dynamo_commit"]) or not _REVISION.fullmatch(identity["model_revision"]):
        raise ValueError("identity contains a noncanonical revision")
    if authorization is not None and (
        not isinstance(authorization, dict)
        or set(authorization) != {"execution_authorized"}
        or not isinstance(authorization["execution_authorized"], bool)
    ):
        raise ValueError("invalid external authorization")
    if not isinstance(value["workload"], dict) or set(value["workload"]) != {"kind", "prompt"}:
        raise ValueError("invalid workload fields")
    if not all(isinstance(value["workload"][key], str) and value["workload"][key] for key in ("kind", "prompt")):
        raise ValueError("invalid workload")
    if value["workload"]["kind"] != "synthetic":
        raise ValueError("unapproved workload")
    if not isinstance(value["digest"], str) or not _SHA256.fullmatch(value["digest"]):
        raise ValueError("invalid lane digest")
    unsigned = {key: item for key, item in value.items() if key != "digest"}
    if hashlib.sha256(_canonical(unsigned).encode()).hexdigest() != value["digest"]:
        raise ValueError("lane digest mismatch")
    return value


def make_paired_blinded_plan(lane):
    """Create one deterministic, two-arm schedule per each of twenty blocks."""
    validate_lane(lane)
    generator = random.Random(lane["seed"])
    modes = ["cold", "restore"]
    generator.shuffle(modes)
    key = dict(zip(("A", "B"), modes))
    schedule = []
    for block in range(1, 21):
        arms = ["A", "B"]
        generator.shuffle(arms)
        for sequence, arm in enumerate(arms, 1):
            schedule.append(
                {"run_id": f"v2-{block:02d}-{sequence:02d}", "block": block, "sequence_in_block": sequence, "opaque_arm": arm}
            )
    return {"schedule": schedule, "unblinding_key": key}


def schedule_digest(schedule):
    if not isinstance(schedule, list) or len(schedule) != 40:
        raise ValueError("schedule must contain exactly forty runs")
    return hashlib.sha256(_canonical(schedule).encode()).hexdigest()


def seal_plan(plan, directory):
    """Write the blinded schedule and its private unblinding key separately."""
    if not isinstance(plan, dict) or set(plan) != {"schedule", "unblinding_key"}:
        raise ValueError("invalid plan")
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    schedule_path = directory / "schedule.json"
    key_path = directory / "unblinding-key.json"
    schedule_path.write_text(_canonical(plan["schedule"]))
    fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        encoded = _canonical(plan["unblinding_key"]).encode()
        if os.write(fd, encoded) != len(encoded):
            raise OSError(errno.EIO, "key write failed")
        os.fsync(fd)
    finally:
        os.close(fd)
    return schedule_path, key_path


class ResultsLedger:
    """Append-only JSONL ledger with process and thread serialization."""

    def __init__(self, path):
        self.path = pathlib.Path(path)
        lock_key = str(self.path.resolve(strict=False))
        with _LOCKS_GUARD:
            self._lock = _LOCKS.setdefault(lock_key, threading.RLock())

    def _open(self, flags):
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(self.path, flags | nofollow, 0o600)
        metadata = os.fstat(fd)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_uid != os.geteuid()
        ):
            os.close(fd)
            raise ValueError("ledger must be a private, singly linked regular file owned by the caller")
        return fd

    @staticmethod
    def _decode(body):
        if not body:
            return []
        if not body.endswith(b"\n"):
            raise ValueError("truncated ledger")
        records = []
        for line in body[:-1].split(b"\n"):
            if not line:
                raise ValueError("blank ledger record")
            try:
                decoded = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("invalid ledger JSON") from exc
            if not isinstance(decoded, dict) or line.decode("utf-8") != _canonical(decoded):
                raise ValueError("non-canonical ledger record")
            records.append(decoded)
        ResultsLedger._verify(records)
        return records

    @staticmethod
    def _verify(records):
        previous = None
        seen_runs = set()
        for sequence, record in enumerate(records, 1):
            if set(("sequence", "previous_record_digest", "record_digest")) - set(record):
                raise ValueError("incomplete ledger chain")
            if record["sequence"] != sequence or not isinstance(record.get("run_id"), str) or record["run_id"] in seen_runs:
                raise ValueError("invalid ledger sequence")
            if record["previous_record_digest"] != previous:
                raise ValueError("broken ledger chain")
            digest = record["record_digest"]
            unsigned = {key: value for key, value in record.items() if key != "record_digest"}
            if not isinstance(digest, str) or not _SHA256.fullmatch(digest) or hashlib.sha256(_canonical(unsigned).encode()).hexdigest() != digest:
                raise ValueError("invalid ledger digest")
            seen_runs.add(record["run_id"])
            previous = digest

    def _read_fd(self, fd):
        os.lseek(fd, 0, os.SEEK_SET)
        chunks = []
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                return self._decode(b"".join(chunks))
            chunks.append(chunk)

    def append(self, record):
        if not isinstance(record, dict) or not isinstance(record.get("run_id"), str):
            raise ValueError("ledger record requires run_id")
        if {"sequence", "previous_record_digest", "record_digest"} & set(record):
            raise ValueError("ledger fields are reserved")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            fd = self._open(os.O_RDWR | os.O_APPEND | os.O_CREAT)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                records = self._read_fd(fd)
                if record["run_id"] in {item["run_id"] for item in records}:
                    raise ValueError("duplicate run_id")
                result = dict(record)
                result["sequence"] = len(records) + 1
                result["previous_record_digest"] = records[-1]["record_digest"] if records else None
                result["record_digest"] = hashlib.sha256(
                    _canonical(result).encode()
                ).hexdigest()
                encoded = (_canonical(result) + "\n").encode()
                written = 0
                while written < len(encoded):
                    count = os.write(fd, encoded[written:])
                    if count <= 0:
                        raise OSError(errno.EIO, "ledger write failed")
                    written += count
                os.fsync(fd)
                return result
            finally:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)

    def read(self):
        with self._lock:
            try:
                fd = self._open(os.O_RDONLY)
            except FileNotFoundError:
                return []
            try:
                fcntl.flock(fd, fcntl.LOCK_SH)
                return self._read_fd(fd)
            finally:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)


def evict_candidate_files(paths, allow_root):
    """Advise only regular, non-symlinked files below the approved checkpoint root."""
    root = pathlib.Path(allow_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("allow_root is not a directory")
    descriptors = []
    try:
        for item in paths:
            candidate = pathlib.Path(item)
            if not candidate.is_absolute():
                candidate = pathlib.Path.cwd() / candidate
            try:
                candidate.relative_to(root)
            except ValueError as exc:
                raise ValueError("candidate is outside allow_root") from exc
            if candidate.is_symlink() or not candidate.parent.resolve(strict=True).is_relative_to(root):
                raise ValueError("candidate symlink is not allowed")
            fd = os.open(candidate, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            descriptors.append(fd)
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("candidate is not a regular file")
        for fd in descriptors:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        for fd in descriptors:
            os.close(fd)


def parse_meminfo(text):
    values = {}
    required = {"MemAvailable", "Cached", "SReclaimable", "Shmem"}
    pattern = re.compile(r"^([A-Za-z_][A-Za-z0-9_()]*):[ \t]+([0-9]+)(?:[ \t]+(kB))?$")
    for line in text.splitlines():
        if not line:
            continue
        match = pattern.fullmatch(line)
        if not match:
            raise ValueError("invalid meminfo")
        name, raw, unit = match.groups()
        if name in required:
            if unit != "kB" or name in values:
                raise ValueError("invalid required meminfo field")
            values[name] = int(raw)
    if not required.issubset(values):
        raise ValueError("missing meminfo fields")
    page_cache = values["Cached"] + values["SReclaimable"] - values["Shmem"]
    if page_cache < 0:
        raise ValueError("negative page cache")
    return {"mem_available_bytes": values["MemAvailable"] * 1024, "page_cache_bytes": page_cache * 1024}


def parse_psi(text):
    output = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 5 or fields[0] not in {"some", "full"} or fields[0] in output:
            raise ValueError("invalid PSI")
        values = {}
        for field in fields[1:]:
            if "=" not in field:
                raise ValueError("invalid PSI field")
            key, raw = field.split("=", 1)
            if key in values or key not in {"avg10", "avg60", "avg300", "total"}:
                raise ValueError("invalid PSI field")
            if (key == "total" and not raw.isdigit()) or (key != "total" and not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", raw)):
                raise ValueError("invalid PSI value")
            try:
                value = int(raw) if key == "total" else float(raw)
            except ValueError as exc:
                raise ValueError("invalid PSI value") from exc
            if not _is_number(value) or value < 0:
                raise ValueError("invalid PSI value")
            values[key] = value
        if set(values) != {"avg10", "avg60", "avg300", "total"}:
            raise ValueError("incomplete PSI")
        output[fields[0]] = values
    if "some" not in output:
        raise ValueError("missing PSI some row")
    return output


def parse_io_stat(text):
    result = {}
    for line in text.splitlines():
        fields = line.split()
        devices = []
        while fields and re.fullmatch(r"[0-9]+:[0-9]+", fields[0]):
            devices.append(fields.pop(0))
        if not devices or len(set(devices)) != len(devices) or len(fields) < 2:
            raise ValueError("invalid io.stat")
        device = devices[-1]
        if device in result:
            raise ValueError("invalid io.stat")
        values = {}
        for field in fields:
            if field.count("=") != 1:
                raise ValueError("invalid io.stat field")
            key, raw = field.split("=", 1)
            if not re.fullmatch(r"[a-z_]+", key) or key in values or not raw.isdigit():
                raise ValueError("invalid io.stat field")
            values[key] = int(raw)
        if not {"rbytes", "wbytes"}.issubset(values):
            raise ValueError("incomplete io.stat")
        result[device] = values
    if not result:
        raise ValueError("empty io.stat")
    return result


def parse_diskstats(text):
    result = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 14 or not fields[0].isdigit() or not fields[1].isdigit() or not re.fullmatch(r"[A-Za-z0-9._-]+", fields[2]):
            raise ValueError("invalid diskstats")
        if fields[2] in result or any(not value.isdigit() for value in fields[3:]):
            raise ValueError("invalid diskstats")
        result[fields[2]] = {
            "major": int(fields[0]),
            "minor": int(fields[1]),
            "reads_completed": int(fields[3]),
            "reads_merged": int(fields[4]),
            "sectors_read": int(fields[5]),
            "milliseconds_reading": int(fields[6]),
            "writes_completed": int(fields[7]),
            "writes_merged": int(fields[8]),
            "sectors_written": int(fields[9]),
            "milliseconds_writing": int(fields[10]),
        }
    if not result:
        raise ValueError("empty diskstats")
    return result


def _cpu_totals(text):
    lines = [line for line in text.splitlines() if line.startswith("cpu ")]
    if len(lines) != 1:
        raise ValueError("missing or duplicate aggregate CPU row")
    fields = lines[0].split()[1:]
    if len(fields) < 4 or any(not value.isdigit() for value in fields):
        raise ValueError("invalid aggregate CPU row")
    counters = [int(value) for value in fields]
    idle = counters[3] + (counters[4] if len(counters) > 4 else 0)
    return sum(counters), idle


def cpu_utilization(before, after):
    total_before, idle_before = _cpu_totals(before)
    total_after, idle_after = _cpu_totals(after)
    total_delta = total_after - total_before
    idle_delta = idle_after - idle_before
    if total_delta <= 0 or idle_delta < 0 or idle_delta > total_delta:
        raise ValueError("CPU counters did not advance monotonically")
    return (total_delta - idle_delta) / total_delta


def parse_gpu_memory_mib(text):
    values = []
    for line in text.splitlines():
        try:
            value = float(line.strip())
        except ValueError as exc:
            raise ValueError("invalid GPU memory observation") from exc
        if not _is_number(value) or value < 0:
            raise ValueError("invalid GPU memory observation")
        values.append(value)
    if not values:
        raise ValueError("missing GPU memory observation")
    return sum(values)


def directory_sizes(paths):
    result = {}
    for label, root in paths.items():
        if not isinstance(label, str) or not label:
            raise ValueError("invalid directory label")
        root = pathlib.Path(root)
        if root.is_symlink() or not root.is_dir():
            raise ValueError("invalid directory")
        total = 0
        pending = [root]
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_symlink():
                        raise ValueError("symlink in measured directory")
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(pathlib.Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
                    else:
                        raise ValueError("non-regular directory entry")
        result[label] = total
    return result


class DrainGate:
    def __init__(self):
        self._lock = threading.RLock()
        self._admission_closed = False
        self._harness_inflight = None
        self._zero_epoch = None
        self._zero_samples = 0
        self._last_sample_seq = None
        self._last_observed_ns = None

    def close_admission(self):
        with self._lock:
            self._admission_closed = True

    def set_harness_inflight(self, count):
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("invalid harness inflight count")
        with self._lock:
            self._harness_inflight = count

    def observe_vllm(self, sample):
        if not isinstance(sample, dict) or set(sample) != {"epoch", "sample_seq", "observed_monotonic_ns", "running", "waiting"} or not isinstance(sample["epoch"], str) or not sample["epoch"]:
            raise ValueError("invalid vLLM sample")
        if any(not isinstance(sample[key], int) or isinstance(sample[key], bool) or sample[key] < 0 for key in ("sample_seq", "observed_monotonic_ns", "running", "waiting")):
            raise ValueError("invalid vLLM sample")
        with self._lock:
            if self._last_sample_seq is not None and sample["sample_seq"] <= self._last_sample_seq:
                raise ValueError("replayed or non-monotonic vLLM sample")
            if self._last_observed_ns is not None and sample["observed_monotonic_ns"] - self._last_observed_ns < 1_000_000_000:
                raise ValueError("vLLM samples are not separated by one second")
            self._last_sample_seq = sample["sample_seq"]
            self._last_observed_ns = sample["observed_monotonic_ns"]
            if sample["running"] or sample["waiting"]:
                self._zero_epoch = None
                self._zero_samples = 0
            elif self._zero_epoch == sample["epoch"]:
                self._zero_samples += 1
            else:
                self._zero_epoch = sample["epoch"]
                self._zero_samples = 1

    @property
    def is_drained(self):
        with self._lock:
            return self._admission_closed and self._harness_inflight == 0 and self._zero_samples >= 2


def complete_metric_record(run_id, metrics, failure_reason, failure_stage=None):
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id) or not isinstance(metrics, dict):
        raise ValueError("invalid metric record")
    if set(metrics) - REQUIRED_METRICS:
        raise ValueError("unknown metric")
    metrics = dict(metrics)
    legacy_cold = all(metrics.get(field) is None for field in ("pod_to_restore_start_s", "criu_restore_s", "cuda_restore_s"))
    for field in ("token_after_restore_summary_s", "checkpoint_storage_read_bytes", "checkpoint_storage_read_throughput_bytes_s"):
        metrics.setdefault(field, None if legacy_cold else 0)
    for field in ("node_page_cache_delta_bytes", "node_memory_available_delta_bytes"):
        metrics.setdefault(field, 0)
    if failure_reason is None:
        if set(metrics) != REQUIRED_METRICS:
            raise ValueError("successful record lacks metrics")
        restore_only = (
            "pod_to_restore_start_s", "criu_restore_s", "cuda_restore_s",
            "token_after_restore_summary_s", "checkpoint_storage_read_bytes",
            "checkpoint_storage_read_throughput_bytes_s",
        )
        kv_release_only = ("prepare_s", "sleep_s", "wake_s")
        optional_compatibility = {"node_page_cache_delta_bytes", "node_memory_available_delta_bytes"}
        if any(metrics[field] is None for field in REQUIRED_METRICS - set(restore_only) - set(kv_release_only) - optional_compatibility):
            raise ValueError("successful record lacks common metrics")
        for fields, message in (
            (restore_only, "restore-specific metrics must be all present or all inapplicable"),
            (kv_release_only, "KV-release metrics must be all present or all inapplicable"),
        ):
            values = [metrics[field] for field in fields]
            if not (all(value is None for value in values) or all(value is not None for value in values)):
                raise ValueError(message)
        if all(metrics[field] is not None for field in restore_only):
            for field in ("checkpoint_storage_read_bytes", "checkpoint_storage_read_throughput_bytes_s"):
                if not _is_number(metrics[field]) or metrics[field] <= 0:
                    raise ValueError("restore storage evidence must be strictly positive")
        if failure_stage is not None:
            raise ValueError("successful record has failure stage")
    elif not isinstance(failure_reason, str) or not failure_reason or not isinstance(failure_stage, str) or not failure_stage:
        raise ValueError("terminal failure requires reason and stage")
    result = {"run_id": run_id}
    result.update({field: metrics.get(field) for field in REQUIRED_METRICS})
    result["failure_reason"] = failure_reason
    result["failure_stage"] = failure_stage
    return result


def verify_success_evidence(row, artifact_dir):
    """Fail closed on any missing, unsafe, or altered successful raw evidence."""
    if not isinstance(row, dict):
        raise ValueError("invalid evidence row")
    if row.get("failure_reason") is not None:
        return True
    root = pathlib.Path(artifact_dir)
    try:
        root_info = os.lstat(root)
    except OSError as exc:
        raise ValueError("artifact directory is unavailable") from exc
    if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
        raise ValueError("artifact directory is unsafe")
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise ValueError("artifact directory is unavailable") from exc
    cold = row.get("mode") == "cold" or (
        row.get("mode") is None and all(
            row.get(field) is None for field in ("pod_to_restore_start_s", "criu_restore_s", "cuda_restore_s")
        )
    )
    fields = (
        ("raw_events_ref", "raw_events_sha256", False),
        ("raw_logs_ref", "raw_logs_sha256", cold),
        ("raw_telemetry_ref", "raw_telemetry_sha256", False),
        ("raw_response_ref", "raw_response_sha256", False),
    )
    for ref_field, digest_field, empty_ok in fields:
        ref, expected = row.get(ref_field), row.get(digest_field)
        if not isinstance(ref, str) or not ref or not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
            raise ValueError("successful row lacks bound raw evidence")
        relative = pathlib.PurePath(ref)
        if relative.is_absolute() or any(part in {".", ".."} for part in relative.parts):
            raise ValueError("raw evidence reference escapes artifact directory")
        path = root
        try:
            for part in relative.parts:
                path = path / part
                info = os.lstat(path)
                if stat.S_ISLNK(info.st_mode):
                    raise ValueError("raw evidence path contains a symlink")
            path.resolve(strict=True).relative_to(resolved_root)
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except (OSError, ValueError) as exc:
            raise ValueError("raw evidence reference is unsafe") from exc
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("raw evidence is not a regular file")
            digest = hashlib.sha256()
            size = 0
            while True:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                size += len(chunk)
                digest.update(chunk)
        finally:
            os.close(fd)
        if (not empty_ok and size == 0) or digest.hexdigest() != expected:
            raise ValueError("raw evidence content does not match ledger")
    return True


def _nearest_rank(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def _evaluate_records(records, schedule, lane_digest, unblinding_key, artifact_dir=None):
    reasons = []
    if (
        not isinstance(unblinding_key, dict)
        or set(unblinding_key) != {"A", "B"}
        or set(unblinding_key.values()) != {"cold", "restore"}
    ):
        raise ValueError("invalid unblinding key")
    if not isinstance(lane_digest, str) or not _SHA256.fullmatch(lane_digest):
        raise ValueError("invalid lane digest")
    try:
        expected_schedule_digest = schedule_digest(schedule)
    except ValueError as exc:
        raise ValueError("invalid sealed schedule") from exc
    expected_by_run = {
        row["run_id"]: row for row in schedule
        if isinstance(row, dict) and set(row) == {"run_id", "block", "sequence_in_block", "opaque_arm"}
    }
    if len(expected_by_run) != 40:
        raise ValueError("sealed schedule contains duplicate or invalid runs")
    if not isinstance(records, list) or len(records) != 40:
        reasons.append("exactly forty complete records are required")
    if artifact_dir is not None and isinstance(records, list):
        for record in records:
            verify_success_evidence(record, artifact_dir)
    run_ids = set()
    execution_digests = set()
    for record in records if isinstance(records, list) else ():
        if not isinstance(record, dict) or not REQUIRED_METRICS.issubset(record):
            reasons.append("missing required metric evidence")
            continue
        run_id = record.get("run_id")
        if not isinstance(run_id, str) or run_id in run_ids:
            reasons.append("invalid or duplicate run identifier")
        run_ids.add(run_id)
        if record.get("failure_reason") is not None:
            reasons.append(str(record["failure_reason"]))
        digest = record.get("execution_digest")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            reasons.append("missing execution digest")
        else:
            execution_digests.add(digest)
    paired_fields = {
        "block", "sequence_in_block", "opaque_arm", "restore_success", "valid_response",
        "excluded", "error", "lane_digest", "schedule_digest",
    }
    for record in records if isinstance(records, list) else ():
        if not paired_fields.issubset(record):
            reasons.append("incomplete paired gate evidence")
            continue
        if record["excluded"] is not False:
            reasons.append("statistical exclusion")
        if record["error"] is not None:
            reasons.append("unexplained error")
        if record["valid_response"] is not True:
            reasons.append("invalid response")
        expected = expected_by_run.get(record.get("run_id"))
        if expected is None or any(record.get(field) != expected[field] for field in ("block", "sequence_in_block", "opaque_arm")):
            reasons.append("record does not match sealed schedule")
        if record["lane_digest"] != lane_digest or record["schedule_digest"] != expected_schedule_digest:
            reasons.append("record identity digest mismatch")
        if (
            record["opaque_arm"] not in {"A", "B"}
            or not isinstance(record["block"], int)
            or isinstance(record["block"], bool)
            or record["block"] not in range(1, 21)
            or record["sequence_in_block"] not in (1, 2)
        ):
            reasons.append("invalid paired block")
        mode = unblinding_key.get(record.get("opaque_arm"))
        common_numeric = (
            "pod_to_scheduled_s", "ready_s", "http_200_s", "first_token_s",
            "node_page_cache_bytes", "node_memory_available_bytes",
            "node_cpu_utilization", "gpu_memory_mib", "checkpoint_size_bytes",
            "pages_12_size_bytes", "rootfs_size_bytes", "metadata_size_bytes",
            "harness_inflight", "vllm_running", "vllm_waiting", "tokens_per_second",
        )
        for field in common_numeric:
            if not _is_number(record[field]) or record[field] < 0:
                reasons.append("invalid " + field)
        for field in (
            "pod_to_restore_start_s", "criu_restore_s", "cuda_restore_s",
            "token_after_restore_summary_s",
        ):
            value = record[field]
            if (mode == "restore" and (not _is_number(value) or value < 0)) or (mode == "cold" and value is not None):
                reasons.append("invalid arm-specific " + field)
        for field in ("checkpoint_storage_read_bytes", "checkpoint_storage_read_throughput_bytes_s"):
            value = record[field]
            if (mode == "restore" and (not _is_number(value) or value <= 0)) or (mode == "cold" and value is not None):
                reasons.append("invalid arm-specific " + field)
        kv_release_values = [record[field] for field in ("prepare_s", "sleep_s", "wake_s")]
        if any(value is not None for value in kv_release_values) and any(
            not _is_number(value) or value < 0 for value in kv_release_values
        ):
            reasons.append("invalid applicable KV-release telemetry")
        for field in ("node_page_cache_delta_bytes", "node_memory_available_delta_bytes"):
            if not _is_number(record[field]):
                reasons.append("invalid " + field)
        if record["admission_closed"] is not True:
            reasons.append("admission is not closed")
        for field in ("harness_inflight", "vllm_running", "vllm_waiting"):
            if record[field] != 0:
                reasons.append(field + " is not zero")
        if not isinstance(record["cgroup_io_stat"], dict) or not isinstance(record["diskstats"], dict):
            reasons.append("invalid I/O telemetry")
        if any(not isinstance(record[field], dict) for field in ("psi_cpu", "psi_io", "psi_memory")):
            reasons.append("invalid PSI telemetry")
    if len(execution_digests) != 1:
        reasons.append("execution digest mismatch")
    blocks = {}
    for record in records if isinstance(records, list) else ():
        if isinstance(record.get("block"), int) and record.get("opaque_arm") in {"A", "B"}:
            blocks.setdefault(record["block"], []).append(record)
    if set(blocks) != set(range(1, 21)) or any(
        len(pair) != 2 or {row["opaque_arm"] for row in pair} != {"A", "B"} for pair in blocks.values()
    ):
        reasons.append("twenty complete A/B blocks are required")
    else:
        for rows in blocks.values():
            if any(not paired_fields.issubset(row) for row in rows):
                continue
            pair = {unblinding_key[row["opaque_arm"]]: row for row in rows}
            if pair["restore"]["restore_success"] is not True:
                reasons.append("restore failure")
            baseline = pair["cold"]["gpu_memory_mib"]
            candidate = pair["restore"]["gpu_memory_mib"]
            if not _is_number(baseline) or not _is_number(candidate) or baseline <= 0 or abs(candidate - baseline) / baseline > 0.05:
                reasons.append("paired GPU memory deviation exceeds five percent")
    candidate_rows = [
        record for record in records
        if isinstance(record, dict)
        and record.get("opaque_arm") in unblinding_key
        and unblinding_key[record["opaque_arm"]] == "restore"
    ]
    return reasons, blocks, candidate_rows


def evaluate_diagnosis_gate(records, schedule, lane_digest, unblinding_key, artifact_dir=None):
    """Require complete, correctly bound evidence without optimization targets."""
    reasons, _, candidate_rows = _evaluate_records(records, schedule, lane_digest, unblinding_key, artifact_dir)
    if len(candidate_rows) != 20:
        reasons.append("twenty restore observations are required")
    return {"passed": not reasons, "reasons": reasons}


def evaluate_optimized_gate(records, schedule, lane_digest, unblinding_key, artifact_dir=None):
    """Overlay promotion latency, memory, and derived throughput targets."""
    reasons, blocks, candidate_rows = _evaluate_records(records, schedule, lane_digest, unblinding_key, artifact_dir)
    candidate_summary = {}
    if len(candidate_rows) == 20 and all(
        _is_number(record.get(field)) for record in candidate_rows
        for field in ("first_token_s", "criu_restore_s")
    ):
        first_tokens = [record["first_token_s"] for record in candidate_rows]
        criu = [record["criu_restore_s"] for record in candidate_rows]
        candidate_summary = {
            "first_token_s": {
                "median": statistics.median(first_tokens),
                "p95": _nearest_rank(first_tokens, 0.95),
                "max": max(first_tokens),
            },
            "criu_restore_s": {"p95": _nearest_rank(criu, 0.95)},
        }
        if candidate_summary["first_token_s"]["median"] > 15:
            reasons.append("first-token median exceeds 15 seconds")
        if candidate_summary["first_token_s"]["p95"] > 25:
            reasons.append("first-token p95 exceeds 25 seconds")
        if candidate_summary["first_token_s"]["max"] > 40:
            reasons.append("first-token maximum exceeds 40 seconds")
        if candidate_summary["criu_restore_s"]["p95"] > 12:
            reasons.append("CRIU restore p95 exceeds 12 seconds")
    else:
        reasons.append("twenty numeric restore observations are required")
    if set(blocks) == set(range(1, 21)):
        for rows in blocks.values():
            if len(rows) != 2 or any(not _is_number(row.get("tokens_per_second")) for row in rows):
                reasons.append("paired throughput observations are required")
                continue
            pair = {unblinding_key[row["opaque_arm"]]: row for row in rows}
            if pair["cold"]["tokens_per_second"] <= 0 or pair["restore"]["tokens_per_second"] / pair["cold"]["tokens_per_second"] < 0.9:
                reasons.append("derived paired throughput ratio is below 90 percent")
    return {"passed": not reasons, "reasons": reasons, "candidate": candidate_summary}


evaluate_gate = evaluate_optimized_gate
