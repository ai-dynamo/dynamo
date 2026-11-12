#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only native LWS fixture snapshots. Part of #17951. Not a verdict."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

LWS_NAMES = ("recovery-target", "recovery-control")
GATES = {
    "recovery-target": "recovery-target-gates",
    "recovery-control": "recovery-control-gates",
}
LWS_RESOURCE = "leaderworkersets.leaderworkerset.x-k8s.io"
NAME_LABEL = "leaderworkerset.sigs.k8s.io/name"
LABELS = (
    NAME_LABEL,
    "leaderworkerset.sigs.k8s.io/group-index",
    "leaderworkerset.sigs.k8s.io/group-key",
    "leaderworkerset.sigs.k8s.io/worker-index",
    "leaderworkerset.sigs.k8s.io/template-revision-hash",
)
POD_CONDITIONS = ("Ready", "PodScheduled", "ContainersReady", "Initialized")
GATE_VALUES = ("healthy", "fail")
NOTE = "Sequential GETs, not one atomic snapshot. Not a recovery count or a pass."
# One SIGKILL of the owned kubectl group, then one wait of this length.
CLEANUP_WAIT_SECONDS = 2
STATE_FIELDS = {
    "terminated": ("reason", "exitCode", "startedAt", "finishedAt"),
    "waiting": ("reason",),
    "running": ("startedAt",),
}


class DeadlineReached(Exception):
    """No time remains in the overall duration."""


class CollectionFailure(Exception):
    """Nonzero failure. The message is a category, never an API body."""

    def __init__(self, category: str, operation: str, returncode: int | None = None):
        super().__init__(category)
        self.category = category
        self.operation = operation
        self.returncode = returncode


class Interrupted(Exception):
    """Ctrl-C after bounded kubectl cleanup. This is exit 130, not completion."""

    def __init__(self, cleanup: CollectionFailure | None = None):
        super().__init__("interrupted")
        self.cleanup = cleanup


@dataclass(frozen=True)
class Config:
    """kubectl target and the collection window."""

    context: str
    namespace: str
    output: Path
    duration: float
    interval: float
    command_timeout: float


def utc_now() -> str:
    """UTC timestamp with millisecond precision."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def picked(mapping: dict, keys: tuple[str, ...]) -> dict:
    """Copy only keys the API sent. Absence is not turned into zero."""
    return {key: mapping[key] for key in keys if key in mapping}


def bad(operation: str) -> None:
    """Reject a malformed object without copying it."""
    raise CollectionFailure("invalid_response", operation)


def metadata(document, kind, namespace, operation, name=None, label=None) -> dict:
    """Identity, allowlisted labels, and owner identity."""
    if document.get("kind") != kind:
        bad(operation)
    meta = document.get("metadata")
    if not isinstance(meta, dict) or meta.get("namespace") != namespace:
        bad(operation)
    if name is not None and meta.get("name") != name:
        bad(operation)
    actual, uid, version = (
        meta.get("name"),
        meta.get("uid"),
        meta.get("resourceVersion"),
    )
    if not all(isinstance(value, str) and value for value in (actual, uid, version)):
        bad(operation)
    raw_labels = meta.get("labels")
    if label is not None and (
        not isinstance(raw_labels, dict) or raw_labels.get(NAME_LABEL) != label
    ):
        bad(operation)
    projected = {
        "name": actual,
        "namespace": namespace,
        "uid": uid,
        "resourceVersion": version,
        **picked(meta, ("generation", "creationTimestamp", "deletionTimestamp")),
    }
    if "labels" in meta:
        if not isinstance(raw_labels, dict):
            bad(operation)
        kept = picked(raw_labels, LABELS)
        if kept:
            projected["labels"] = kept
    if "ownerReferences" not in meta:
        return projected
    refs = meta["ownerReferences"]
    if not isinstance(refs, list):
        bad(operation)
    owners = []
    for ref in refs:
        if not isinstance(ref, dict):
            bad(operation)
        owners.append(picked(ref, ("apiVersion", "kind", "name", "uid")))
    projected["ownerReferences"] = owners
    return projected


def conditions(status: dict, allowed: tuple[str, ...] | None, operation: str):
    """Selected condition fields, or None when the API omitted conditions."""
    if "conditions" not in status:
        return None
    rows = status["conditions"]
    if not isinstance(rows, list):
        bad(operation)
    selected, seen = [], set()
    fields = ("type", "status", "reason", "lastTransitionTime", "observedGeneration")
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("type"), str):
            bad(operation)
        if allowed is not None and row["type"] not in allowed:
            continue
        if row["type"] in seen:
            bad(operation)
        seen.add(row["type"])
        selected.append(picked(row, fields))
    return selected


def container_state(state: object, operation: str) -> dict | None:
    """Exit time, code, and reason for one current or previous state."""
    if not isinstance(state, dict):
        bad(operation)
    if not state:
        return None
    for phase, fields in STATE_FIELDS.items():
        if phase not in state:
            continue
        body = state[phase]
        if not isinstance(body, dict):
            bad(operation)
        return {"phase": phase, **picked(body, fields)}
    bad(operation)
    return None


def containers(pod: dict, spec_key: str, status_key: str, operation: str):
    """Image, image ID, readiness, and exit fields. No env, command, or args."""
    spec = pod.get("spec")
    status = pod.get("status") if isinstance(pod.get("status"), dict) else {}
    if not isinstance(spec, dict):
        bad(operation)
    if spec_key not in spec and status_key not in status:
        return None
    spec_rows, status_rows = spec.get(spec_key, []), status.get(status_key, [])
    if not isinstance(spec_rows, list) or not isinstance(status_rows, list):
        bad(operation)
    order, items, seen_status = [], {}, set()
    for row in spec_rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            bad(operation)
        if row["name"] in items:
            bad(operation)
        order.append(row["name"])
        items[row["name"]] = {"name": row["name"], **picked(row, ("image",))}
    for row in status_rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            bad(operation)
        if row["name"] in seen_status:
            bad(operation)
        seen_status.add(row["name"])
        item = items.setdefault(row["name"], {"name": row["name"]})
        if row["name"] not in order:
            order.append(row["name"])
        item.update(picked(row, ("imageID", "ready", "restartCount")))
        for key in ("state", "lastState"):
            if key in row:
                projected = container_state(row[key], operation)
                if projected is not None:
                    item[key] = projected
    return [items[name] for name in order]


def project_lws(document: dict, name: str, namespace: str, operation: str) -> dict:
    """Fixture LeaderWorkerSet policy and status, without the Pod template."""
    spec = document.get("spec")
    template = spec.get("leaderWorkerTemplate") if isinstance(spec, dict) else None
    if not isinstance(template, dict):
        bad(operation)
    spec_out = {
        **picked(spec, ("replicas", "startupPolicy")),
        **picked(template, ("size", "restartPolicy")),
    }
    strategy = spec.get("rolloutStrategy")
    if isinstance(strategy, dict):
        rolled = picked(strategy, ("type",))
        rolling = strategy.get("rollingUpdateConfiguration")
        if isinstance(rolling, dict):
            update = picked(rolling, ("partition", "maxUnavailable", "maxSurge"))
            if update:
                rolled["rollingUpdateConfiguration"] = update
        if rolled:
            spec_out["rolloutStrategy"] = rolled
    subgroup = template.get("subGroupPolicy")
    if isinstance(subgroup, dict):
        group = picked(subgroup, ("subGroupPolicyType", "subGroupSize"))
        if group:
            spec_out["subGroupPolicy"] = group
    network = spec.get("networkConfig")
    if isinstance(network, dict) and picked(network, ("subdomainPolicy",)):
        spec_out["networkConfig"] = picked(network, ("subdomainPolicy",))
    projected = {
        "kind": "LeaderWorkerSet",
        "metadata": metadata(document, "LeaderWorkerSet", namespace, operation, name),
        "spec": spec_out,
    }
    status = document.get("status")
    if "status" in document and not isinstance(status, dict):
        bad(operation)
    if isinstance(status, dict):
        status_out = picked(
            status,
            ("observedGeneration", "replicas", "readyReplicas", "updatedReplicas"),
        )
        rows = conditions(status, None, operation)
        if rows is not None:
            status_out["conditions"] = rows
        if status_out:
            projected["status"] = status_out
    return projected


def project_pod(document: dict, lws_name: str, namespace: str, operation: str) -> dict:
    """One Pod. Assigned node is spec.nodeName."""
    if not isinstance(document, dict):
        bad(operation)
    projected = {
        "kind": "Pod",
        "metadata": metadata(document, "Pod", namespace, operation, label=lws_name),
    }
    spec, status = document.get("spec"), document.get("status")
    if not isinstance(spec, dict):
        bad(operation)
    if "nodeName" in spec:
        projected["nodeName"] = spec["nodeName"]
    if isinstance(status, dict):
        projected.update(picked(status, ("phase",)))
        rows = conditions(status, POD_CONDITIONS, operation)
        if rows is not None:
            projected["conditions"] = rows
    elif "status" in document:
        bad(operation)
    for spec_key, status_key, out_key in (
        ("containers", "containerStatuses", "containers"),
        ("initContainers", "initContainerStatuses", "init_containers"),
    ):
        rows = containers(document, spec_key, status_key, operation)
        if rows is not None:
            projected[out_key] = rows
    return projected


def project_pods(document, lws_name, namespace, operation) -> list:
    """kubectl get -o json uses kind List. PodList is accepted too."""
    if not isinstance(document, dict) or document.get("kind") not in (
        "List",
        "PodList",
    ):
        bad(operation)
    items = document.get("items")
    if not isinstance(items, list):
        bad(operation)
    return [project_pod(item, lws_name, namespace, operation) for item in items]


def project_gate(document: dict, name: str, namespace: str, operation: str) -> dict:
    """Name, identity, and the two gates. Other data and bad values are dropped."""
    if not isinstance(document, dict) or document.get("kind") != "ConfigMap":
        bad(operation)
    meta = document.get("metadata")
    if not isinstance(meta, dict):
        bad(operation)
    if meta.get("name") != name or meta.get("namespace") != namespace:
        bad(operation)
    uid, version = meta.get("uid"), meta.get("resourceVersion")
    if not all(isinstance(value, str) and value for value in (uid, version)):
        bad(operation)
    data = document.get("data")
    if not isinstance(data, dict):
        raise CollectionFailure("invalid_gate", operation)
    leader, follower = data.get("leader"), data.get("follower")
    if leader not in GATE_VALUES or follower not in GATE_VALUES:
        raise CollectionFailure("invalid_gate", operation)
    return {
        "name": name,
        "namespace": namespace,
        "uid": uid,
        "resourceVersion": version,
        "leader": leader,
        "follower": follower,
    }


def signal_group(proc: subprocess.Popen) -> None:
    """SIGKILL only the process group of this start_new_session Popen."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        raise CollectionFailure("cleanup_failed", "reap/kubectl") from None


def wait_bounded(proc: subprocess.Popen) -> bool:
    """Wait one fixed interval. Ctrl-C during the wait is not consumed here."""
    try:
        proc.wait(timeout=CLEANUP_WAIT_SECONDS)
    except subprocess.TimeoutExpired:
        return False
    return True


def close_pipes(proc: subprocess.Popen) -> None:
    """Close this collector's pipe ends. Do not read them."""
    for pipe in (proc.stdout, proc.stderr, proc.stdin):
        if pipe is None:
            continue
        try:
            pipe.close()
        except OSError:
            raise CollectionFailure("cleanup_failed", "reap/kubectl") from None


def reap_kubectl(proc: subprocess.Popen) -> None:
    """SIGKILL the owned group, close local pipes, and wait at most 2 seconds."""
    signal_group(proc)
    close_pipes(proc)
    if not wait_bounded(proc):
        raise CollectionFailure("cleanup_failed", "reap/kubectl")


def stop_owned_kubectl(proc: subprocess.Popen, interrupted: bool) -> None:
    """Reap one group. Ctrl-C stays exit 130 and keeps any cleanup failure."""
    cleanup_error = None
    try:
        reap_kubectl(proc)
    except CollectionFailure as exc:
        cleanup_error = exc
    except KeyboardInterrupt:
        cleanup_error = CollectionFailure("cleanup_failed", "reap/kubectl")
        interrupted = True
    if interrupted:
        raise Interrupted(cleanup_error) from None
    if cleanup_error is not None:
        raise cleanup_error


def kubectl_get(config: Config, deadline: float, operation: str, resource: list):
    """One GET. Request and process timeouts are the time still left, at most."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise DeadlineReached()
    timeout = min(config.command_timeout, remaining)
    argv = [
        "kubectl",
        "--context",
        config.context,
        "--namespace",
        config.namespace,
        f"--request-timeout={max(1, int(timeout * 1000))}ms",
        "get",
        *resource,
        "-o",
        "json",
    ]
    try:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
    except FileNotFoundError:
        raise CollectionFailure("kubectl_not_found", operation) from None
    try:
        stdout, _stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        stop_owned_kubectl(proc, interrupted=False)
        if remaining <= config.command_timeout:
            raise DeadlineReached() from None
        raise CollectionFailure("command_timeout", operation) from None
    except KeyboardInterrupt:
        stop_owned_kubectl(proc, interrupted=True)
    if proc.returncode != 0:
        raise CollectionFailure("kubectl_failed", operation, proc.returncode) from None
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        raise CollectionFailure("invalid_json", operation, proc.returncode) from None
    if not isinstance(payload, dict):
        bad(operation)
    return payload


def has_rows(sample: dict) -> bool:
    """True after at least one projected object was stored."""
    return bool(
        sample["leader_worker_sets"]
        or sample["pods_by_leader_worker_set"]
        or sample["config_maps"]
    )


def fill_sample(config: Config, deadline: float, sample: dict) -> None:
    """Target set, its Pods, its gate, then the same three for the control."""
    for name in LWS_NAMES:
        operation = f"get/leaderworkerset/{name}"
        raw = kubectl_get(config, deadline, operation, [LWS_RESOURCE, name])
        sample["leader_worker_sets"].append(
            project_lws(raw, name, config.namespace, operation)
        )
        operation = f"get/pods/{name}"
        selector = f"{NAME_LABEL}={name}"
        raw = kubectl_get(config, deadline, operation, ["pods", "-l", selector])
        sample["pods_by_leader_worker_set"][name] = {
            "label_selector": selector,
            "items": project_pods(raw, name, config.namespace, operation),
        }
        gate = GATES[name]
        operation = f"get/configmap/{gate}"
        raw = kubectl_get(config, deadline, operation, ["configmap", gate])
        sample["config_maps"].append(
            project_gate(raw, gate, config.namespace, operation)
        )


def write_json(path: Path, payload: dict) -> None:
    """Replace a temp file in the output directory. Leave older snapshots."""
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)
    except (OSError, TypeError, ValueError):
        raise CollectionFailure("file_error", "write/output") from None


def run_collection(config: Config, started_at: str, started: float) -> None:
    """Stop at the duration. Do not reserve time for later GETs in the sample."""
    deadline = started + config.duration
    count, incomplete, sample, written = 0, None, None, False

    def stamp(complete: bool) -> None:
        sample["complete"] = complete
        sample["sample_finished_at_utc"] = utc_now()
        sample["elapsed_seconds"] = round(time.monotonic() - started, 3)

    def save_open_sample() -> None:
        if sample is not None and has_rows(sample) and not written:
            stamp(False)
            write_json(config.output / f"snapshot-{count:04d}.json", sample)

    try:
        while deadline - time.monotonic() > 0:
            count += 1
            written = False
            sample = {
                "sample_index": count,
                "sample_started_at_utc": utc_now(),
                "complete": False,
                "evidence_note": NOTE,
                "leader_worker_sets": [],
                "pods_by_leader_worker_set": {},
                "config_maps": [],
            }
            try:
                fill_sample(config, deadline, sample)
            except DeadlineReached:
                if not has_rows(sample):
                    count -= 1
                    break
                stamp(False)
                write_json(config.output / f"snapshot-{count:04d}.json", sample)
                incomplete = count
                break
            stamp(True)
            write_json(config.output / f"snapshot-{count:04d}.json", sample)
            written = True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(config.interval, remaining))
    except (KeyboardInterrupt, Interrupted):
        try:
            save_open_sample()
        except CollectionFailure:
            pass
        raise
    except CollectionFailure:
        try:
            save_open_sample()
        except CollectionFailure:
            pass
        raise
    if count == 0:
        raise CollectionFailure("no_observations", "collection")
    write_json(
        config.output / "collection.json",
        {
            "result": "completed",
            "reason": "duration_elapsed",
            "snapshot_count": count,
            "incomplete_sample_index": incomplete,
            "started_at_utc": started_at,
            "finished_at_utc": utc_now(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "parameters": {
                "context": config.context,
                "namespace": config.namespace,
                "duration_seconds": config.duration,
                "interval_seconds": config.interval,
                "command_timeout_seconds": config.command_timeout,
            },
            "statement": (
                "Duration elapsed after at least one observation. "
                "This file is not a recovery result."
            ),
        },
    )


def write_status(
    config, result, category, operation, returncode, started_at, started, cleanup=None
):
    """Failure or interruption record. No stdout, stderr, or gate text."""
    error = {"category": category, "operation": operation}
    if returncode is not None:
        error["returncode"] = returncode
    payload = {
        "result": result,
        "started_at_utc": started_at,
        "observation_time_utc": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "error": error,
    }
    if cleanup is not None:
        payload["cleanup_error"] = {
            "category": cleanup.category,
            "operation": cleanup.operation,
        }
    try:
        write_json(config.output / "error.json", payload)
    except CollectionFailure:
        print("error: file_error: write/error", file=sys.stderr)
    print(f"error: {category}: {operation}", file=sys.stderr)
    if cleanup is not None:
        print(f"error: {cleanup.category}: {cleanup.operation}", file=sys.stderr)


def positive(raw: str) -> float:
    """Parse a positive finite CLI number. Bounds are checked by the caller."""
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("not a finite number") from exc
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("not a finite number")
    return value


def context_token(raw: str) -> str:
    """One kube context token."""
    if raw.strip() != raw or not 1 <= len(raw) <= 256 or any(ord(c) < 32 for c in raw):
        raise argparse.ArgumentTypeError("invalid context")
    return raw


def namespace_name(raw: str) -> str:
    """DNS-1123 namespace label."""
    if raw.strip() != raw or not 1 <= len(raw) <= 63:
        raise argparse.ArgumentTypeError("invalid namespace")
    for index, char in enumerate(raw):
        ok = ("a" <= char <= "z") or ("0" <= char <= "9")
        if (char != "-" and not ok) or (index in (0, len(raw) - 1) and not ok):
            raise argparse.ArgumentTypeError("invalid namespace")
    return raw


def new_directory(raw: str) -> Path:
    """A path that does not exist, inside a directory that does."""
    path = Path(raw)
    if path.exists() or not path.parent.is_dir():
        raise argparse.ArgumentTypeError("output must be a new directory")
    return path


def parse_config(argv: list[str] | None) -> Config:
    """Required context, namespace, and new output directory."""
    parser = argparse.ArgumentParser(description="Read the native LWS CPU fixture.")
    parser.add_argument("--context", required=True, type=context_token)
    parser.add_argument("--namespace", required=True, type=namespace_name)
    parser.add_argument("--output", required=True, type=new_directory)
    parser.add_argument("--duration", type=positive, default=300)
    parser.add_argument("--interval", type=positive, default=5)
    parser.add_argument("--command-timeout", type=positive, default=15)
    args = parser.parse_args(argv)
    if not 1 <= args.duration <= 1800 or not 1 <= args.interval <= 60:
        raise ValueError("duration or interval is outside its limit")
    if not 1 <= args.command_timeout <= 60 or args.interval > args.duration:
        raise ValueError("command timeout or interval is outside its limit")
    return Config(
        args.context,
        args.namespace,
        args.output,
        args.duration,
        args.interval,
        args.command_timeout,
    )


def main(argv: list[str] | None = None) -> int:
    """Exit 130 on Ctrl-C. That path does not write collection.json."""
    try:
        config = parse_config(argv)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        config.output.mkdir()
    except OSError:
        print("error: file_error: create/output", file=sys.stderr)
        return 1
    started_at, started = utc_now(), time.monotonic()
    try:
        run_collection(config, started_at, started)
    except Interrupted as exc:
        write_status(
            config,
            "interrupted",
            "interrupted",
            "collection",
            None,
            started_at,
            started,
            exc.cleanup,
        )
        return 130
    except KeyboardInterrupt:
        write_status(
            config,
            "interrupted",
            "interrupted",
            "collection",
            None,
            started_at,
            started,
        )
        return 130
    except CollectionFailure as exc:
        write_status(
            config,
            "failed",
            exc.category,
            exc.operation,
            exc.returncode,
            started_at,
            started,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
