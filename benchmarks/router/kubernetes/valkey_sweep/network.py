# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime proof that plaintext Valkey endpoints are NetworkPolicy-isolated."""

from __future__ import annotations

import json
import secrets
import subprocess
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import yaml

from .model import CLIENT_DEPLOYMENT, PART_OF


class NetworkCluster(Protocol):
    namespace: str

    def kubectl(
        self,
        arguments: Sequence[str],
        *,
        input_text: str | None = None,
        timeout: float | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]: ...

    def client_exec(
        self,
        arguments: Sequence[str],
        *,
        timeout: float | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]: ...

    def delete_with_uid_precondition(self, resource: str, uid: str) -> bool: ...


@dataclass
class CreatedProbeResource:
    """A resource registered for cleanup before its create request begins."""

    resource: str
    uid: str | None = None


def _probe_document_uid(document: object, resource: str, probe_id: str) -> str:
    if not isinstance(document, dict):
        raise RuntimeError(f"temporary probe response is not an object for {resource}")
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError(f"temporary probe response has no metadata for {resource}")
    labels = metadata.get("labels")
    uid = metadata.get("uid")
    if (
        not isinstance(labels, dict)
        or labels.get("app.kubernetes.io/part-of") != PART_OF
        or labels.get("valkey-sweep-probe-id") != probe_id
        or not isinstance(uid, str)
        or not uid
    ):
        raise RuntimeError(f"temporary probe ownership mismatch for {resource}")
    return uid


def _probe_resource_uid(
    cluster: NetworkCluster, resource: str, probe_id: str
) -> str | None:
    result = cluster.kubectl(
        ("get", resource, "-o", "json", "--ignore-not-found=true"), timeout=30
    )
    if not result.stdout.strip():
        return None
    return _probe_document_uid(json.loads(result.stdout), resource, probe_id)


def _create_probe_resource(
    cluster: NetworkCluster, resource: str, document: dict[str, Any], probe_id: str
) -> str:
    try:
        result = cluster.kubectl(
            ("create", "-f", "-", "-o", "json"),
            input_text=yaml.safe_dump(document, sort_keys=False),
            timeout=30,
        )
    except (subprocess.SubprocessError, OSError):
        uid = _probe_resource_uid(cluster, resource, probe_id)
        if uid is None:
            raise
        return uid

    try:
        return _probe_document_uid(json.loads(result.stdout), resource, probe_id)
    except (json.JSONDecodeError, RuntimeError):
        # A successful create may have reached the API server even when its response
        # is truncated by the client transport. Reconcile from the authoritative API.
        pass
    uid = _probe_resource_uid(cluster, resource, probe_id)
    if uid is None:
        raise RuntimeError(f"created temporary probe resource disappeared: {resource}")
    return uid


def _delete_probe_resource(
    cluster: NetworkCluster, resource: str, expected_uid: str | None, probe_id: str
) -> None:
    observed_uid = _probe_resource_uid(cluster, resource, probe_id)
    if observed_uid is None:
        return
    if expected_uid is not None and observed_uid != expected_uid:
        raise RuntimeError(
            f"refusing to delete replaced probe resource {resource}: "
            f"expected UID={expected_uid}, observed UID={observed_uid}"
        )
    delete_uid = expected_uid or observed_uid
    if not cluster.delete_with_uid_precondition(resource, delete_uid):
        raise RuntimeError(
            f"refusing to delete replaced probe resource {resource}: "
            f"expected UID={delete_uid}"
        )
    if _probe_resource_uid(cluster, resource, probe_id) is not None:
        raise RuntimeError(f"temporary probe resource was not deleted: {resource}")


def prove_network_isolation(cluster: NetworkCluster, image: str) -> dict[str, Any]:
    endpoints = (
        ("valkey-sweep-router-primary", 6379),
        ("valkey-sweep-tokenizer-primary", 6379),
        ("valkey-sweep-sentinel-0.valkey-sweep-sentinel", 26379),
    )
    connect_probe = (
        "import socket,sys; "
        "connection=socket.create_connection((sys.argv[1],int(sys.argv[2])),2); "
        "connection.close()"
    )
    deny_probe = """\
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
try:
    addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
except OSError as error:
    print(f"DNS_FAILURE:{type(error).__name__}")
    raise SystemExit(24)
for family, socktype, proto, _, address in addresses:
    connection = socket.socket(family, socktype, proto)
    connection.settimeout(2)
    try:
        connection.connect(address)
    except TimeoutError:
        continue
    except OSError as error:
        print(f"CONNECT_FAILURE:{error.errno}:{type(error).__name__}")
        raise SystemExit(25)
    finally:
        connection.close()
    print("NETWORK_REACHABLE")
    raise SystemExit(0)
print("EXPECTED_NETWORK_DENIAL")
raise SystemExit(23)
"""
    probe_id = secrets.token_hex(8)
    pod_name = f"valkey-sweep-network-probe-{probe_id}"
    policy_name = f"valkey-sweep-network-probe-{probe_id}"
    probe_labels = {
        "app.kubernetes.io/part-of": PART_OF,
        "app.kubernetes.io/component": "network-probe",
        "valkey-sweep-probe-id": probe_id,
    }
    pod = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": pod_name,
            "namespace": cluster.namespace,
            "labels": probe_labels,
        },
        "spec": {
            "automountServiceAccountToken": False,
            "restartPolicy": "Never",
            "nodeSelector": {"kubernetes.io/arch": "arm64"},
            "imagePullSecrets": [{"name": "nvcr-pull-secret"}],
            "securityContext": {
                "runAsNonRoot": True,
                "seccompProfile": {"type": "RuntimeDefault"},
            },
            "containers": [
                {
                    "name": "probe",
                    "image": image,
                    "command": [
                        "/bin/sh",
                        "-c",
                        "trap : TERM INT; sleep infinity & wait",
                    ],
                    "securityContext": {
                        "allowPrivilegeEscalation": False,
                        "capabilities": {"drop": ["ALL"]},
                    },
                    "resources": {
                        "requests": {"cpu": "10m", "memory": "32Mi"},
                        "limits": {"cpu": "1", "memory": "128Mi"},
                    },
                }
            ],
        },
    }
    policy = {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "NetworkPolicy",
        "metadata": {
            "name": policy_name,
            "namespace": cluster.namespace,
            "labels": probe_labels,
        },
        "spec": {
            "podSelector": {"matchLabels": probe_labels},
            "policyTypes": ["Egress"],
            "egress": [
                {
                    "to": [
                        {
                            "namespaceSelector": {
                                "matchLabels": {
                                    "kubernetes.io/metadata.name": "kube-system"
                                }
                            },
                            "podSelector": {"matchLabels": {"k8s-app": "kube-dns"}},
                        }
                    ],
                    "ports": [
                        {"protocol": "UDP", "port": 53},
                        {"protocol": "TCP", "port": 53},
                    ],
                },
                {
                    "to": [
                        {
                            "podSelector": {
                                "matchLabels": {"app.kubernetes.io/component": "etcd"}
                            }
                        }
                    ],
                    "ports": [{"protocol": "TCP", "port": 2379}],
                },
            ],
        },
    }
    created: list[CreatedProbeResource] = []
    try:
        policy_resource = f"networkpolicy/{policy_name}"
        created_policy = CreatedProbeResource(policy_resource)
        created.append(created_policy)
        created_policy.uid = _create_probe_resource(
            cluster, policy_resource, policy, probe_id
        )
        pod_resource = f"pod/{pod_name}"
        created_pod = CreatedProbeResource(pod_resource)
        created.append(created_pod)
        created_pod.uid = _create_probe_resource(cluster, pod_resource, pod, probe_id)
        cluster.kubectl(
            ("wait", "--for=condition=Ready", f"pod/{pod_name}", "--timeout=5m"),
            timeout=330,
        )
        cluster.kubectl(
            (
                "exec",
                f"pod/{pod_name}",
                "--",
                "/app/.venv/bin/python",
                "-c",
                connect_probe,
                f"etcd.{cluster.namespace}.svc.cluster.local",
                "2379",
            ),
            timeout=10,
        )
        for host, port in endpoints:
            cluster.client_exec(
                ("/app/.venv/bin/python", "-c", connect_probe, host, str(port)),
                timeout=10,
            )
            result = cluster.kubectl(
                (
                    "exec",
                    f"pod/{pod_name}",
                    "--",
                    "/app/.venv/bin/python",
                    "-c",
                    deny_probe,
                    host,
                    str(port),
                ),
                timeout=10,
                check=False,
            )
            if (
                result.returncode != 23
                or result.stdout.strip() != "EXPECTED_NETWORK_DENIAL"
            ):
                raise RuntimeError(
                    f"network isolation probe produced an ambiguous result for "
                    f"{host}:{port}: exit={result.returncode}, "
                    f"stdout={result.stdout!r}, stderr={result.stderr!r}"
                )
            cluster.client_exec(
                ("/app/.venv/bin/python", "-c", connect_probe, host, str(port)),
                timeout=10,
            )
    finally:
        cleanup_errors: list[Exception] = []
        for created_resource in reversed(created):
            try:
                _delete_probe_resource(
                    cluster,
                    created_resource.resource,
                    created_resource.uid,
                    probe_id,
                )
            except Exception as error:
                cleanup_errors.append(error)
        if cleanup_errors:
            raise RuntimeError(
                "failed to clean up temporary network probes"
            ) from cleanup_errors[0]
    return {
        "status": "passed",
        "positive_source": CLIENT_DEPLOYMENT,
        "negative_source": "ephemeral-network-probe",
        "negative_egress_control": "etcd:2379",
        "endpoints": [f"{host}:{port}" for host, port in endpoints],
    }
