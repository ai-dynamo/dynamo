# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import json
import re
import select
import subprocess
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import yaml

from .model import (
    CLIENT_DEPLOYMENT,
    FRONTEND_DEPLOYMENT,
    INDEX_SCOPE,
    MOCKER_DEPLOYMENT,
    NAMESPACE,
    PART_OF,
    MatrixPoint,
)


SWEEP_DIR = Path(__file__).resolve().parent
MANIFEST_FILES = tuple(
    SWEEP_DIR / name
    for name in (
        "stack-config.yaml",
        "stack-data.yaml",
        "stack-policy.yaml",
        "stack-workload.yaml",
    )
)
EXPECTED_RESOURCES = {
    ("v1", "ConfigMap", "valkey-sweep-config"),
    ("v1", "PersistentVolumeClaim", "valkey-sweep-router-primary-data"),
    ("v1", "PersistentVolumeClaim", "valkey-sweep-router-replica-data"),
    ("v1", "PersistentVolumeClaim", "valkey-sweep-tokenizer-primary-state"),
    ("v1", "PersistentVolumeClaim", "valkey-sweep-tokenizer-replica-state"),
    ("v1", "Service", "valkey-sweep-frontend"),
    ("v1", "Service", "valkey-sweep-router-primary"),
    ("v1", "Service", "valkey-sweep-router-replica"),
    ("v1", "Service", "valkey-sweep-sentinel"),
    ("v1", "Service", "valkey-sweep-tokenizer-primary"),
    ("v1", "Service", "valkey-sweep-tokenizer-replica"),
    ("apps/v1", "Deployment", "valkey-sweep-client"),
    ("apps/v1", "Deployment", "valkey-sweep-frontend"),
    ("apps/v1", "Deployment", "valkey-sweep-mocker"),
    ("apps/v1", "StatefulSet", "valkey-sweep-router-primary"),
    ("apps/v1", "StatefulSet", "valkey-sweep-router-replica"),
    ("apps/v1", "StatefulSet", "valkey-sweep-sentinel"),
    ("apps/v1", "StatefulSet", "valkey-sweep-tokenizer-primary"),
    ("apps/v1", "StatefulSet", "valkey-sweep-tokenizer-replica"),
    ("policy/v1", "PodDisruptionBudget", "valkey-sweep-router"),
    ("policy/v1", "PodDisruptionBudget", "valkey-sweep-sentinel"),
    ("policy/v1", "PodDisruptionBudget", "valkey-sweep-tokenizer"),
    ("networking.k8s.io/v1", "NetworkPolicy", "valkey-sweep-client-egress"),
    ("networking.k8s.io/v1", "NetworkPolicy", "valkey-sweep-frontend-egress"),
    ("networking.k8s.io/v1", "NetworkPolicy", "valkey-sweep-frontend-ingress"),
    ("networking.k8s.io/v1", "NetworkPolicy", "valkey-sweep-mocker-egress"),
    ("networking.k8s.io/v1", "NetworkPolicy", "valkey-sweep-mocker-ingress"),
    ("networking.k8s.io/v1", "NetworkPolicy", "valkey-sweep-router-egress"),
    ("networking.k8s.io/v1", "NetworkPolicy", "valkey-sweep-router-ingress"),
    ("networking.k8s.io/v1", "NetworkPolicy", "valkey-sweep-sentinel-egress"),
    ("networking.k8s.io/v1", "NetworkPolicy", "valkey-sweep-sentinel-ingress"),
    ("networking.k8s.io/v1", "NetworkPolicy", "valkey-sweep-tokenizer-egress"),
    ("networking.k8s.io/v1", "NetworkPolicy", "valkey-sweep-tokenizer-ingress"),
}
GENERATED_SENTINEL_PVCS = {
    f"sentinel-data-valkey-sweep-sentinel-{ordinal}" for ordinal in range(3)
}
IMAGE_DIGEST_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/:-]{0,188}@sha256:[0-9a-f]{64}")
IMAGE_PLACEHOLDER = "${VALKEY_SWEEP_IMAGE}"
GROUP_IDENTITIES = {
    "dynamo-router": (
        "valkey-sweep-router-primary",
        "valkey-sweep-router-replica",
    ),
    "dynamo-tokenizer": (
        "valkey-sweep-tokenizer-primary",
        "valkey-sweep-tokenizer-replica",
    ),
}
CANONICAL_MASTER_HOSTS = {
    master_name: frozenset(
        host
        for identity in identities
        for host in (identity, f"{identity}-0.{identity}")
    )
    for master_name, identities in GROUP_IDENTITIES.items()
}
PROBE_API_PATHS = {
    "pod": ("/api/v1", "pods"),
    "networkpolicy": ("/apis/networking.k8s.io/v1", "networkpolicies"),
}


def _require_sweep_selector(selector: object, description: str) -> None:
    if not isinstance(selector, dict):
        raise ValueError(f"{description} is missing")
    labels = selector.get("matchLabels", selector)
    if (
        not isinstance(labels, dict)
        or labels.get("app.kubernetes.io/part-of") != PART_OF
    ):
        raise ValueError(f"{description} is not sweep-scoped")


def probe_api_path(namespace: str, resource: str) -> str:
    kind, separator, name = resource.partition("/")
    if not separator or not name or "/" in name or kind not in PROBE_API_PATHS:
        raise ValueError(f"unsupported temporary probe resource: {resource}")
    api_prefix, plural = PROBE_API_PATHS[kind]
    return f"{api_prefix}/namespaces/{quote(namespace, safe='')}/{plural}/{quote(name, safe='')}"


@contextlib.contextmanager
def kubectl_proxy() -> Iterator[str]:
    """Expose the authenticated API on loopback for one conditional delete."""
    process = subprocess.Popen(
        (
            "kubectl",
            "proxy",
            "--address=127.0.0.1",
            "--port=0",
            "--api-prefix=/",
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        if process.stdout is None:
            raise RuntimeError("kubectl proxy did not expose stdout")
        readable, _, _ = select.select((process.stdout,), (), (), 15)
        if not readable:
            raise RuntimeError("timed out waiting for kubectl proxy")
        startup = process.stdout.readline().strip()
        match = re.fullmatch(r"Starting to serve on 127\.0\.0\.1:(\d+)", startup)
        if match is None:
            raise RuntimeError(
                f"kubectl proxy did not report a loopback address: {startup!r}"
            )
        yield f"http://127.0.0.1:{match.group(1)}"
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def delete_with_uid_precondition(namespace: str, resource: str, uid: str) -> bool:
    """Delete exactly one temporary resource, retaining UID replacements."""
    payload = json.dumps(
        {
            "apiVersion": "v1",
            "kind": "DeleteOptions",
            "preconditions": {"uid": uid},
        }
    ).encode("utf-8")
    with kubectl_proxy() as endpoint:
        request = Request(
            f"{endpoint}{probe_api_path(namespace, resource)}",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="DELETE",
        )
        try:
            with urlopen(request, timeout=30):
                return True
        except HTTPError as error:
            if error.code == 404:
                return True
            if error.code == 409:
                return False
            raise RuntimeError(
                f"UID-preconditioned delete failed for {resource}: HTTP {error.code}"
            ) from error


def run_command(
    command: Sequence[str],
    *,
    input_text: str | None = None,
    timeout: float | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command), flush=True)
    return subprocess.run(
        list(command),
        check=check,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


class Cluster:
    def __init__(self, namespace: str):
        self.namespace = namespace

    def kubectl(
        self,
        arguments: Sequence[str],
        *,
        input_text: str | None = None,
        timeout: float | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return run_command(
            ("kubectl", "-n", self.namespace, *arguments),
            input_text=input_text,
            timeout=timeout,
            check=check,
        )

    def client_exec(
        self,
        arguments: Sequence[str],
        *,
        timeout: float | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return self.kubectl(
            ("exec", f"deployment/{CLIENT_DEPLOYMENT}", "--", *arguments),
            timeout=timeout,
            check=check,
        )

    def valkey(self, host: str, *arguments: str) -> list[str]:
        result = self.client_exec(
            ("valkey-cli", "--raw", "-h", host, "-p", "6379", *arguments),
            timeout=30,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def sentinel(self, ordinal: int, *arguments: str) -> list[str]:
        host = f"valkey-sweep-sentinel-{ordinal}.valkey-sweep-sentinel"
        result = self.client_exec(
            ("valkey-cli", "--raw", "-h", host, "-p", "26379", *arguments),
            timeout=30,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def delete_with_uid_precondition(self, resource: str, uid: str) -> bool:
        return delete_with_uid_precondition(self.namespace, resource, uid)


def render_stack(image: str, manifest_texts: Sequence[str] | None = None) -> str:
    if not IMAGE_DIGEST_RE.fullmatch(image):
        raise ValueError("image must be an immutable digest image reference")
    texts = manifest_texts or tuple(
        path.read_text(encoding="utf-8") for path in MANIFEST_FILES
    )
    documents = [
        document
        for content in texts
        for document in yaml.safe_load_all(content)
        if document
    ]
    inventory = {
        (
            document.get("apiVersion"),
            document.get("kind"),
            document.get("metadata", {}).get("name"),
        )
        for document in documents
    }
    if len(inventory) != len(documents) or inventory != EXPECTED_RESOURCES:
        raise ValueError(
            f"stack resource inventory differs from allowlist: "
            f"missing={sorted(EXPECTED_RESOURCES - inventory)}, "
            f"unexpected={sorted(inventory - EXPECTED_RESOURCES)}"
        )
    replaced = 0
    for document in documents:
        metadata = document["metadata"]
        if (
            metadata.get("namespace") != NAMESPACE
            or metadata.get("labels", {}).get("app.kubernetes.io/part-of") != PART_OF
        ):
            raise ValueError(
                f"resource is not sweep-scoped: {document['kind']}/{metadata['name']}"
            )
        document["metadata"]["namespace"] = NAMESPACE
        spec = document.get("spec", {})
        selector = spec.get("selector")
        if selector is not None:
            _require_sweep_selector(
                selector, f"selector for {document['kind']}/{metadata['name']}"
            )
        if document["kind"] == "NetworkPolicy":
            _require_sweep_selector(
                spec.get("podSelector"),
                f"network policy selector for {metadata['name']}",
            )
        for claim in spec.get("volumeClaimTemplates", []):
            if (
                claim.get("metadata", {})
                .get("labels", {})
                .get("app.kubernetes.io/part-of")
                != PART_OF
            ):
                raise ValueError(
                    f"generated PVC is not sweep-scoped: "
                    f"{document['kind']}/{metadata['name']}"
                )
        pod_spec = document.get("spec", {}).get("template", {}).get("spec", {})
        if pod_spec:
            if pod_spec.get("securityContext", {}).get("runAsNonRoot") is not True:
                raise ValueError(
                    f"pod template permits root execution: {metadata['name']}"
                )
            if pod_spec.get("securityContext", {}).get("seccompProfile") != {
                "type": "RuntimeDefault"
            }:
                raise ValueError(
                    f"pod template lacks RuntimeDefault seccomp: {metadata['name']}"
                )
            if pod_spec.get("automountServiceAccountToken") is not False:
                raise ValueError(
                    f"pod template mounts an API token: {metadata['name']}"
                )
            if any(
                pod_spec.get(field) for field in ("hostNetwork", "hostPID", "hostIPC")
            ):
                raise ValueError(
                    f"pod template uses host namespaces: {metadata['name']}"
                )
        containers = pod_spec.get("initContainers", []) + pod_spec.get("containers", [])
        for container in containers:
            if container.get("image") != IMAGE_PLACEHOLDER:
                raise ValueError(
                    f"unexpected image in {document['kind']}/{document['metadata']['name']}"
                )
            container["image"] = image
            replaced += 1
            security = container.get("securityContext", {})
            if security.get("allowPrivilegeEscalation") is not False or security.get(
                "privileged", False
            ):
                raise ValueError(
                    f"container lacks restricted security context: "
                    f"{metadata['name']}/{container.get('name')}"
                )
            if security.get("capabilities", {}).get("drop") != ["ALL"]:
                raise ValueError(
                    f"container retains Linux capabilities: "
                    f"{metadata['name']}/{container.get('name')}"
                )
    if replaced == 0:
        raise ValueError("stack manifests have no image placeholders")
    return yaml.safe_dump_all(documents, sort_keys=False)


def apply_stack(cluster: Cluster, image: str) -> None:
    assert_stack_ownership(cluster)
    assert_etcd_selector(cluster)
    cluster.kubectl(("apply", "-f", "-"), input_text=render_stack(image), timeout=180)
    for stateful_set in (
        "valkey-sweep-router-primary",
        "valkey-sweep-router-replica",
        "valkey-sweep-tokenizer-primary",
        "valkey-sweep-tokenizer-replica",
        "valkey-sweep-sentinel",
    ):
        cluster.kubectl(
            ("rollout", "status", f"statefulset/{stateful_set}", "--timeout=10m"),
            timeout=660,
        )
    cluster.kubectl(
        ("rollout", "status", f"deployment/{CLIENT_DEPLOYMENT}", "--timeout=10m"),
        timeout=660,
    )
    wait_for_ha(cluster)


def assert_stack_ownership(cluster: Cluster) -> None:
    references = [(kind, name) for _, kind, name in EXPECTED_RESOURCES] + [
        ("PersistentVolumeClaim", name) for name in GENERATED_SENTINEL_PVCS
    ]
    for kind, name in sorted(references):
        result = cluster.kubectl(
            ("get", f"{kind.lower()}/{name}", "-o", "json", "--ignore-not-found=true"),
            timeout=30,
        )
        if not result.stdout.strip():
            continue
        existing = json.loads(result.stdout)
        part_of = (
            existing.get("metadata", {})
            .get("labels", {})
            .get("app.kubernetes.io/part-of")
        )
        if part_of != PART_OF:
            raise RuntimeError(
                f"refusing to replace unowned {kind}/{name}: "
                f"app.kubernetes.io/part-of={part_of!r}"
            )


def assert_etcd_selector(cluster: Cluster) -> None:
    result = cluster.kubectl(("get", "service/etcd", "-o", "json"), timeout=30)
    service = json.loads(result.stdout)
    selector = service.get("spec", {}).get("selector", {})
    expected = {"app.kubernetes.io/component": "etcd"}
    if selector != expected:
        raise RuntimeError(
            f"the existing etcd Service selector must equal {expected}, "
            f"observed {selector}"
        )


def load_generator_capacity(cluster: Cluster, concurrency: int) -> dict[str, int | str]:
    result = cluster.client_exec(
        ("/bin/sh", "-c", "ulimit -n; cat /proc/sys/net/ipv4/ip_local_port_range"),
        timeout=30,
    )
    lines = result.stdout.splitlines()
    if len(lines) != 2:
        raise RuntimeError(f"invalid load-generator capacity response: {lines}")
    fd_limit_text = lines[0].strip()
    fd_limit = (2**63 - 1) if fd_limit_text == "unlimited" else int(fd_limit_text)
    port_start, port_end = (int(value) for value in lines[1].split())
    ephemeral_ports = port_end - port_start + 1
    required = concurrency + max(4096, concurrency // 4)
    if fd_limit < required or ephemeral_ports < required:
        raise RuntimeError(
            f"load generator cannot sustain C={concurrency} with headroom: "
            f"required={required}, fd_limit={fd_limit_text}, "
            f"ephemeral_port_range={port_start}-{port_end} ({ephemeral_ports})"
        )
    return {
        "fd_limit": fd_limit_text,
        "ephemeral_port_start": port_start,
        "ephemeral_port_end": port_end,
        "ephemeral_ports": ephemeral_ports,
        "required_connections_with_headroom": required,
    }


def sentinel_master(cluster: Cluster, master_name: str) -> tuple[str, int]:
    try:
        allowed_hosts = CANONICAL_MASTER_HOSTS[master_name]
    except KeyError as error:
        raise ValueError(f"unknown Sentinel master name: {master_name}") from error
    votes: list[tuple[str, int]] = []
    for ordinal in range(3):
        try:
            response = cluster.sentinel(
                ordinal, "SENTINEL", "get-master-addr-by-name", master_name
            )
            if len(response) == 2:
                votes.append((response[0], int(response[1])))
        except (ValueError, subprocess.SubprocessError, OSError):
            continue
    if not votes:
        raise RuntimeError(f"no Sentinel returned a master for {master_name}")
    master, count = Counter(votes).most_common(1)[0]
    if count < 2:
        raise RuntimeError(f"no Sentinel majority for {master_name}: {votes}")
    if master[0] not in allowed_hosts or master[1] != 6379:
        raise RuntimeError(
            f"Sentinel majority returned non-canonical master for {master_name}: {master}"
        )
    return master


def master_identity(master_name: str, host: str) -> str:
    try:
        identities = GROUP_IDENTITIES[master_name]
    except KeyError as error:
        raise ValueError(f"unknown Sentinel master name: {master_name}") from error
    matches = [
        identity
        for identity in identities
        if host == identity or host.startswith(f"{identity}-0.")
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Sentinel returned non-canonical host {host!r} for {master_name}"
        )
    return matches[0]


def selected_valkey_info(cluster: Cluster, host: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for section in ("replication", "clients", "memory", "stats"):
        for line in cluster.valkey(host, "INFO", section):
            if ":" not in line or line.startswith("#"):
                continue
            key, value = line.split(":", 1)
            if key in {
                "role",
                "connected_slaves",
                "master_link_status",
                "connected_clients",
                "blocked_clients",
                "used_memory",
                "used_memory_peak",
                "maxmemory",
                "maxmemory_policy",
                "total_commands_processed",
                "instantaneous_ops_per_sec",
                "rejected_connections",
                "total_error_replies",
            }:
                result[key] = value
    return result


def replication_snapshot(
    cluster: Cluster, master_name: str
) -> tuple[str, str, dict[str, dict[str, str]], tuple[str, int]]:
    identities = GROUP_IDENTITIES[master_name]
    elected = sentinel_master(cluster, master_name)
    elected_name = master_identity(master_name, elected[0])
    infos = {
        identity: selected_valkey_info(cluster, identity) for identity in identities
    }
    role_masters = [
        identity for identity, info in infos.items() if info.get("role") == "master"
    ]
    if role_masters != [elected_name]:
        raise RuntimeError(
            f"Sentinel election and identity roles disagree for {master_name}: "
            f"elected={elected}, roles={infos}"
        )
    replica_name = next(identity for identity in identities if identity != elected_name)
    if infos[replica_name].get("role") != "slave":
        raise RuntimeError(f"{master_name} has no replica role: {infos}")
    return elected_name, replica_name, infos, elected


def wait_for_ha(cluster: Cluster) -> None:
    deadline = time.monotonic() + 180
    last_error = "not queried"
    while time.monotonic() < deadline:
        try:
            snapshots = {
                master_name: replication_snapshot(cluster, master_name)
                for master_name in GROUP_IDENTITIES
            }
            quorum = [
                " ".join(cluster.sentinel(0, "SENTINEL", "CKQUORUM", master))
                for master in GROUP_IDENTITIES
            ]
            unhealthy = {
                master_name: infos
                for master_name, (elected, replica, infos, _) in snapshots.items()
                if int(infos[elected].get("connected_slaves", "0")) < 1
                or infos[replica].get("master_link_status") != "up"
            }
            if not unhealthy and all(value.startswith("OK") for value in quorum):
                return
            last_error = f"replication={unhealthy}, quorum={quorum}"
        except (RuntimeError, ValueError, subprocess.SubprocessError, OSError) as error:
            last_error = str(error)
        time.sleep(2)
    raise TimeoutError(f"HA Valkey topology did not become ready: {last_error}")


def index_key(runtime_namespace: str) -> str:
    def pct(value: str) -> str:
        return quote(value, safe="-._~")

    return (
        f"dynamo:kv-router:{pct(runtime_namespace)}:component-mocker:"
        f"scope-{INDEX_SCOPE}:block-size-16"
    )


def registered_mocker_stats(cluster: Cluster, runtime_namespace: str) -> list[int]:
    master = sentinel_master(cluster, "dynamo-router")
    lines = cluster.valkey(master[0], "DYNKV.STATS", index_key(runtime_namespace))
    return [int(value) for value in lines]


def wait_for_registered_mockers(
    cluster: Cluster, runtime_namespace: str, expected: int
) -> list[int]:
    key = index_key(runtime_namespace)
    deadline = time.monotonic() + 900
    last_stats: list[int] = []
    while time.monotonic() < deadline:
        try:
            last_stats = registered_mocker_stats(cluster, runtime_namespace)
            if len(last_stats) == 3 and last_stats[1] == expected:
                return last_stats
        except (RuntimeError, ValueError, subprocess.SubprocessError, OSError):
            pass
        time.sleep(2)
    raise TimeoutError(
        f"expected exactly {expected} registered mocker ranks for {key}, "
        f"last DYNKV.STATS={last_stats}"
    )


def wait_for_valkey_value(
    cluster: Cluster,
    host: str,
    arguments: Sequence[str],
    expected: list[str],
    *,
    timeout: float = 30,
) -> list[str]:
    deadline = time.monotonic() + timeout
    observed: list[str] = []
    while time.monotonic() < deadline:
        try:
            observed = cluster.valkey(host, *arguments)
            if observed == expected:
                return observed
        except (ValueError, subprocess.SubprocessError, OSError):
            pass
        time.sleep(0.25)
    raise TimeoutError(
        f"Valkey state did not converge on {host}: command={list(arguments)}, "
        f"expected={expected}, observed={observed}"
    )


def sentinel_policy_snapshot(cluster: Cluster) -> dict[str, dict[str, dict[str, str]]]:
    expected = {
        "quorum": "2",
        "down-after-milliseconds": "5000",
        "failover-timeout": "30000",
        "parallel-syncs": "1",
    }
    snapshot: dict[str, dict[str, dict[str, str]]] = {}
    for ordinal in range(3):
        witness: dict[str, dict[str, str]] = {}
        for master_name in CANONICAL_MASTER_HOSTS:
            response = cluster.sentinel(ordinal, "SENTINEL", "MASTER", master_name)
            if len(response) % 2:
                raise RuntimeError(
                    f"Sentinel {ordinal} returned malformed policy for {master_name}: {response}"
                )
            values = dict(zip(response[::2], response[1::2], strict=True))
            policy = {key: values.get(key, "") for key in expected}
            if policy != expected:
                raise RuntimeError(
                    f"Sentinel {ordinal} has stale policy for {master_name}: "
                    f"expected={expected}, observed={policy}"
                )
            witness[master_name] = policy
        snapshot[str(ordinal)] = witness
    return snapshot


def rollout(cluster: Cluster, deployment: str, timeout: str = "20m") -> None:
    cluster.kubectl(
        ("rollout", "status", f"deployment/{deployment}", f"--timeout={timeout}"),
        timeout=1260,
    )


def pods(cluster: Cluster, selector: str) -> list[dict[str, Any]]:
    return json.loads(
        cluster.kubectl(("get", "pods", f"-l={selector}", "-o", "json")).stdout
    )["items"]


def wait_for_no_pods(cluster: Cluster, deployment: str) -> None:
    deadline = time.monotonic() + 180
    selector = f"app.kubernetes.io/name={deployment}"
    while time.monotonic() < deadline:
        if not pods(cluster, selector):
            return
        time.sleep(1)
    raise TimeoutError(f"pods for {deployment} did not terminate")


def active_runtime_snapshot(cluster: Cluster) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for pod in pods(cluster, f"app.kubernetes.io/part-of={PART_OF}"):
        init_statuses = pod["status"].get("initContainerStatuses", [])
        container_statuses = pod["status"].get("containerStatuses", [])
        statuses = init_statuses + container_statuses
        image_ids = sorted({status.get("imageID", "") for status in statuses} - {""})
        if not image_ids:
            raise RuntimeError(
                f"active pod {pod['metadata']['name']} has no immutable image ID"
            )
        snapshot[pod["metadata"]["name"]] = {
            "uid": pod["metadata"]["uid"],
            "image_ids": image_ids,
            "restart_count": sum(status.get("restartCount", 0) for status in statuses),
            "phase": pod["status"].get("phase", "unknown"),
            "ready": any(
                condition.get("type") == "Ready" and condition.get("status") == "True"
                for condition in pod["status"].get("conditions", [])
            ),
            "containers_ready": bool(container_statuses)
            and all(status.get("ready") is True for status in container_statuses),
            "init_succeeded": all(
                status.get("state", {}).get("terminated", {}).get("exitCode") == 0
                for status in init_statuses
            ),
        }
    return dict(sorted(snapshot.items()))


def active_image_inventory(cluster: Cluster) -> dict[str, list[str]]:
    return {
        name: details["image_ids"]
        for name, details in active_runtime_snapshot(cluster).items()
    }


def verify_active_images(
    cluster: Cluster, binding: dict[str, Any], point: MatrixPoint
) -> dict[str, dict[str, Any]]:
    snapshot = active_runtime_snapshot(cluster)
    observed = {
        image_id for details in snapshot.values() for image_id in details["image_ids"]
    }
    expected = set(binding["image_ids"])
    if observed != expected:
        raise RuntimeError(
            f"active pod image IDs differ from campaign binding: "
            f"expected={sorted(expected)}, observed={sorted(observed)}"
        )
    fixed_pods = {
        "valkey-sweep-router-primary-0",
        "valkey-sweep-router-replica-0",
        "valkey-sweep-tokenizer-primary-0",
        "valkey-sweep-tokenizer-replica-0",
        "valkey-sweep-sentinel-0",
        "valkey-sweep-sentinel-1",
        "valkey-sweep-sentinel-2",
    }
    missing_fixed_pods = fixed_pods - snapshot.keys()
    client_count = sum(name.startswith(f"{CLIENT_DEPLOYMENT}-") for name in snapshot)
    frontend_count = sum(
        name.startswith(f"{FRONTEND_DEPLOYMENT}-") for name in snapshot
    )
    mocker_count = sum(name.startswith(f"{MOCKER_DEPLOYMENT}-") for name in snapshot)
    expected_count = len(fixed_pods) + 1 + point.frontends + point.mockers
    if (
        missing_fixed_pods
        or client_count != 1
        or frontend_count != point.frontends
        or mocker_count != point.mockers
        or len(snapshot) != expected_count
    ):
        raise RuntimeError(
            f"provenance inventory does not cover requested topology: "
            f"missing_fixed={sorted(missing_fixed_pods)}, clients={client_count}/1, "
            f"frontends={frontend_count}/{point.frontends}, "
            f"mockers={mocker_count}/{point.mockers}, "
            f"pods={len(snapshot)}/{expected_count}"
        )
    unhealthy = {
        name: details
        for name, details in snapshot.items()
        if details["phase"] != "Running"
        or not details["ready"]
        or not details["containers_ready"]
        or not details["init_succeeded"]
    }
    if unhealthy:
        raise RuntimeError(f"active pod inventory is not healthy: {unhealthy}")
    return snapshot
