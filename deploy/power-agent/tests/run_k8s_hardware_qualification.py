#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch the approved P2.8 Kubernetes hardware qualification fixture.

This is an operator-driven test harness, not a cluster installer. It creates
one uniquely named ConfigMap and one exclusive-GPU Job in the supplied
namespace, captures bounded evidence, and deletes only those exact resources.
All kubectl calls use argv arrays; shell execution is intentionally absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SOURCE_FILES = (
    "actuator.py",
    "power_agent.py",
    "managed_state.py",
    "pod_report.py",
    "podresources_identity.py",
    "podresources_api.py",
    "podresources_api_grpc.py",
    "tests/e2e_actuator_parity.py",
    "tests/e2e_gate_backend_probe.py",
    "tests/e2e_gate_entrypoint.py",
    "tests/e2e_k8s_hardware_qualification_entrypoint.py",
    "../../components/src/dynamo/power_gate/gate.py",
)
RESULT_PREFIX = "P2_8_QUALIFICATION_RESULT="
EVIDENCE_PREFIX = "P2_8_K8S_EVIDENCE="
HOSTENGINE_BUILD_MARKER = "Hostengine build info:"


def _kubectl(
    executable: str,
    context: str,
    namespace: str,
    *args: str,
    stdin: str | None = None,
    check: bool = True,
    deadline: float | None = None,
    timeout_seconds: float = 60,
) -> subprocess.CompletedProcess[str]:
    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("outer qualification deadline expired")
        timeout_seconds = min(timeout_seconds, max(1, remaining))
    request_timeout_seconds = max(1, int(timeout_seconds) - 1)
    command = [
        executable,
        "--context",
        context,
        "--namespace",
        namespace,
        f"--request-timeout={request_timeout_seconds}s",
        *args,
    ]
    try:
        result = subprocess.run(
            command,
            input=stdin,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        result = subprocess.CompletedProcess(
            command,
            124,
            stdout=stdout,
            stderr=stderr + f"\nkubectl timed out after {timeout_seconds:.1f}s",
        )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"kubectl failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def _source_payload(power_agent_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    data: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = power_agent_dir / relative
        content = path.read_text(encoding="utf-8")
        key = Path(relative).name
        if key in data:
            raise ValueError(f"duplicate ConfigMap key: {key}")
        data[key] = content
        hashes[relative] = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return data, hashes


def _parse_hostengine_version(output: str) -> str:
    if HOSTENGINE_BUILD_MARKER not in output:
        raise RuntimeError("dcgmi version output lacks Hostengine build info")
    hostengine_output = output.split(HOSTENGINE_BUILD_MARKER, 1)[1]
    match = re.search(r"^Version\s*:\s*(\S+)\s*$", hostengine_output, re.MULTILINE)
    if match is None:
        raise RuntimeError("dcgmi hostengine build info lacks Version")
    return match.group(1)


def _require_digest_image_id(image_id: str, subject: str) -> str:
    if re.search(r"@sha256:[0-9a-f]{64}$", image_id) is None:
        raise RuntimeError(f"{subject} lacks an immutable sha256 image ID")
    return image_id


def _verify_hostengine_endpoints(
    executable: str,
    context: str,
    namespace: str,
    service: str,
    container: str,
    port: int,
    expected_version: str,
    host: str,
    deadline: float | None = None,
) -> dict[str, Any]:
    expected_host = f"{service}.{namespace}.svc.cluster.local"
    if host != expected_host:
        raise RuntimeError(
            f"hostengine host {host!r} must be the verified Service {expected_host!r}"
        )
    service_object = _json_output(
        _kubectl(
            executable,
            context,
            namespace,
            "get",
            "service",
            service,
            "-o",
            "json",
            deadline=deadline,
        )
    )
    service_spec = service_object.get("spec", {})
    if service_spec.get("type") != "ClusterIP" or service_spec.get("clusterIP") in (
        None,
        "",
        "None",
    ):
        raise RuntimeError(
            f"{namespace}/{service} must be a non-headless ClusterIP Service"
        )
    if service_spec.get("internalTrafficPolicy") != "Local":
        raise RuntimeError(
            f"{namespace}/{service} must use internalTrafficPolicy Local"
        )
    matching_ports = [
        item
        for item in service_spec.get("ports", [])
        if item.get("port") == port and item.get("targetPort") == port
    ]
    if len(matching_ports) != 1:
        raise RuntimeError(
            f"{namespace}/{service} does not expose one exact {port}->{port} port"
        )
    endpoints = _json_output(
        _kubectl(
            executable,
            context,
            namespace,
            "get",
            "endpoints",
            service,
            "-o",
            "json",
            deadline=deadline,
        )
    )
    target_refs: dict[str, dict[str, str]] = {}
    for subset in endpoints.get("subsets", []):
        for address in subset.get("addresses", []):
            target = address.get("targetRef", {})
            if target.get("kind") != "Pod" or not target.get("name"):
                raise RuntimeError(
                    f"ready endpoint for {namespace}/{service} is not Pod-backed"
                )
            target_refs[target["name"]] = {
                "uid": target.get("uid", ""),
                "node": address.get("nodeName", ""),
            }
    if not target_refs:
        raise RuntimeError(f"{namespace}/{service} has no ready Pod endpoints")

    verified: list[dict[str, str]] = []
    for pod_name, endpoint in sorted(target_refs.items()):
        pod = _json_output(
            _kubectl(
                executable,
                context,
                namespace,
                "get",
                "pod",
                pod_name,
                "-o",
                "json",
                deadline=deadline,
            )
        )
        pod_uid = pod.get("metadata", {}).get("uid", "")
        if pod_uid != endpoint["uid"]:
            raise RuntimeError(
                f"hostengine endpoint {pod_name} changed UID during preflight"
            )
        statuses = {
            status.get("name"): status
            for status in pod.get("status", {}).get("containerStatuses", [])
        }
        status = statuses.get(container)
        if status is None or not status.get("ready"):
            raise RuntimeError(
                f"hostengine endpoint {namespace}/{pod_name} container "
                f"{container!r} is not ready"
            )
        version_result = _kubectl(
            executable,
            context,
            namespace,
            "exec",
            pod_name,
            "-c",
            container,
            "--",
            "dcgmi",
            "-v",
            "--host",
            f"localhost:{port}",
            deadline=deadline,
        )
        version = _parse_hostengine_version(
            version_result.stdout + "\n" + version_result.stderr
        )
        if version != expected_version:
            raise RuntimeError(
                f"DCGM hostengine {namespace}/{pod_name} version {version!r} != "
                f"expected {expected_version!r}"
            )
        image_id = _require_digest_image_id(
            status.get("imageID", ""), f"hostengine endpoint {namespace}/{pod_name}"
        )
        verified.append(
            {
                "pod": pod_name,
                "podUID": pod_uid,
                "node": pod.get("spec", {}).get("nodeName", endpoint["node"]),
                "container": container,
                "image": status.get("image", ""),
                "imageID": image_id,
                "version": version,
            }
        )

    image_ids = {item["imageID"] for item in verified}
    images = {item["image"] for item in verified}
    if len(image_ids) != 1 or len(images) != 1:
        raise RuntimeError(
            f"{namespace}/{service} ready endpoints are heterogeneous: "
            f"images={sorted(images)}, imageIDs={sorted(image_ids)}"
        )

    return {
        "status": "PASS",
        "verifiedAt": datetime.now(timezone.utc).isoformat(),
        "namespace": namespace,
        "service": service,
        "serviceUID": service_object.get("metadata", {}).get("uid", ""),
        "serviceResourceVersion": service_object.get("metadata", {}).get(
            "resourceVersion", ""
        ),
        "host": host,
        "port": port,
        "internalTrafficPolicy": "Local",
        "endpointResourceVersion": endpoints.get("metadata", {}).get(
            "resourceVersion", ""
        ),
        "expectedVersion": expected_version,
        "readyEndpointCount": len(verified),
        "endpoints": verified,
    }


def _bind_hostengine_postflight(
    preflight: dict[str, Any], postflight: dict[str, Any], node_name: str
) -> dict[str, Any]:
    if preflight.get("serviceUID") != postflight.get("serviceUID"):
        raise RuntimeError("DCGM Service changed UID during qualification")
    pre_by_uid = {item["podUID"]: item for item in preflight.get("endpoints", [])}
    post_by_uid = {item["podUID"]: item for item in postflight.get("endpoints", [])}
    if pre_by_uid != post_by_uid:
        raise RuntimeError("ready DCGM endpoint identity/version/digest set changed")
    local = [
        item
        for item in postflight.get("endpoints", [])
        if item.get("node") == node_name
    ]
    if len(local) != 1:
        raise RuntimeError(
            f"scheduled node {node_name!r} has {len(local)} ready DCGM endpoints"
        )
    return {
        "status": "PASS",
        "preflight": preflight,
        "postflight": postflight,
        "scheduledNodeEndpoint": local[0],
    }


def build_manifests(
    *,
    name: str,
    run_token: str,
    namespace: str,
    gpu_product: str,
    gpu_count: int,
    architecture: str,
    actuator_mode: str,
    runtime_image: str,
    dcgm_image: str | None,
    hostengine_host: str | None,
    hostengine_port: int,
    expected_hostengine_version: str | None,
    active_deadline_seconds: int,
    qualification_timeout_seconds: int,
    source_data: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if actuator_mode not in ("nvml", "nvml-dcgm-parity"):
        raise ValueError(f"unsupported actuator mode: {actuator_mode}")
    if actuator_mode == "nvml-dcgm-parity" and not all(
        (dcgm_image, hostengine_host, expected_hostengine_version)
    ):
        raise ValueError("parity mode requires DCGM image, host, and version")

    labels = {
        "app.kubernetes.io/name": "dynamo-power-qualification",
        "app.kubernetes.io/managed-by": "phase2-harness",
        "phase2.dynamo.nvidia.com/run": name,
        "phase2.dynamo.nvidia.com/run-token": run_token,
    }
    config_map = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": name, "namespace": namespace, "labels": labels},
        "data": source_data,
    }
    init_containers: list[dict[str, Any]] = [
        {
            "name": "python-dependencies",
            "image": runtime_image,
            "imagePullPolicy": "IfNotPresent",
            "command": ["python3", "-m", "pip", "install", "--quiet"],
            "args": [
                "--target=/deps",
                "pynvml==12.0.*",
                "kubernetes==30.*",
                "grpcio==1.83.*",
                "protobuf==7.35.*",
                "prometheus-client==0.20.*",
            ],
            "volumeMounts": [{"name": "python-deps", "mountPath": "/deps"}],
        }
    ]
    qualification_args = [
        "--expected-gpu-count",
        str(gpu_count),
        "--gpu-product",
        gpu_product,
        "--actuator-mode",
        actuator_mode,
        "--timeout-seconds",
        str(qualification_timeout_seconds),
    ]
    qualification_env = [
        {"name": "PYTHONPATH", "value": "/app:/deps"},
        {"name": "PYTHONUNBUFFERED", "value": "1"},
    ]
    qualification_mounts = [
        {"name": "source", "mountPath": "/app", "readOnly": True},
        {"name": "python-deps", "mountPath": "/deps", "readOnly": True},
    ]
    volumes = [
        {"name": "source", "configMap": {"name": name}},
        {"name": "python-deps", "emptyDir": {}},
    ]
    if actuator_mode == "nvml-dcgm-parity":
        init_containers.insert(
            0,
            {
                "name": "dcgm-vendor",
                "image": dcgm_image,
                "imagePullPolicy": "IfNotPresent",
                "command": ["/bin/bash", "-lc"],
                "args": [
                    "\n".join(
                        [
                            "set -euo pipefail",
                            "mkdir -p /vendor/python /vendor/lib",
                            "cp -a /usr/share/datacenter-gpu-manager-4/"
                            + "bindings/python3/*.py /vendor/python/",
                            "cp -a /usr/lib/*-linux-gnu/libdcgm.so* /vendor/lib/",
                            'dcgmi -v --host "${DCGM_HOST}:${DCGM_PORT}" '
                            + "> /vendor/dcgmi-version.txt",
                        ]
                    )
                    + "\n"
                ],
                "env": [
                    {"name": "DCGM_HOST", "value": str(hostengine_host)},
                    {"name": "DCGM_PORT", "value": str(hostengine_port)},
                ],
                "volumeMounts": [{"name": "dcgm-vendor", "mountPath": "/vendor"}],
            },
        )
        qualification_args[6:6] = [
            "--hostengine-host",
            str(hostengine_host),
            "--hostengine-port",
            str(hostengine_port),
            "--verified-hostengine-version",
            str(expected_hostengine_version),
        ]
        qualification_env[0]["value"] = "/app:/opt/dcgm/python:/deps"
        qualification_env.insert(
            1, {"name": "LD_LIBRARY_PATH", "value": "/opt/dcgm/lib"}
        )
        qualification_mounts.insert(
            1,
            {"name": "dcgm-vendor", "mountPath": "/opt/dcgm", "readOnly": True},
        )
        volumes.insert(1, {"name": "dcgm-vendor", "emptyDir": {}})

    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": name, "namespace": namespace, "labels": labels},
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": active_deadline_seconds,
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "restartPolicy": "Never",
                    "terminationGracePeriodSeconds": 120,
                    "nodeSelector": {
                        "kubernetes.io/arch": architecture,
                        "nvidia.com/gpu.product": gpu_product,
                    },
                    "tolerations": [
                        {
                            "key": "kubernetes.io/arch",
                            "operator": "Equal",
                            "value": architecture,
                            "effect": "NoSchedule",
                        },
                        {
                            "key": "nvidia.com/gpu",
                            "operator": "Exists",
                            "effect": "NoSchedule",
                        },
                    ],
                    "initContainers": init_containers,
                    "containers": [
                        {
                            "name": "qualification",
                            "image": runtime_image,
                            "imagePullPolicy": "IfNotPresent",
                            "command": [
                                "python3",
                                "/app/e2e_k8s_hardware_qualification_entrypoint.py",
                            ],
                            "args": qualification_args,
                            "env": qualification_env,
                            "securityContext": {"privileged": True, "runAsUser": 0},
                            "resources": {
                                "requests": {"nvidia.com/gpu": str(gpu_count)},
                                "limits": {"nvidia.com/gpu": str(gpu_count)},
                            },
                            "volumeMounts": qualification_mounts,
                        }
                    ],
                    "volumes": volumes,
                },
            },
        },
    }
    return config_map, job


def _json_output(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return json.loads(result.stdout)


def _wait_for_job(
    executable: str,
    context: str,
    namespace: str,
    name: str,
    deadline: float,
) -> dict[str, Any]:
    next_update = 0.0
    while time.monotonic() < deadline:
        job = _json_output(
            _kubectl(
                executable,
                context,
                namespace,
                "get",
                "job",
                name,
                "-o",
                "json",
                deadline=deadline,
            )
        )
        status = job.get("status", {})
        if status.get("succeeded", 0) >= 1:
            return job
        if status.get("failed", 0) >= 1:
            return job
        if time.monotonic() >= next_update:
            print(
                f"waiting for {namespace}/{name}: " f"active={status.get('active', 0)}",
                flush=True,
            )
            next_update = time.monotonic() + 30
        time.sleep(5)
    raise TimeoutError("qualification Job did not finish before the outer deadline")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _publish_successful_evidence(
    evidence_dir: Path,
    summary: dict[str, Any],
    cleanup: dict[str, Any],
) -> dict[str, Any]:
    cleanup_status = cleanup.get("status")
    if cleanup_status not in ("PASS", "RETAINED"):
        raise RuntimeError("refusing to publish PASS evidence before cleanup succeeds")
    payload = {
        **summary,
        "cleanup": cleanup,
        "evidenceStatus": (
            "PASS" if cleanup_status == "PASS" else "RETAINED_NOT_CLEANED"
        ),
    }
    _write_json(evidence_dir / "summary.json", payload)
    print(EVIDENCE_PREFIX + json.dumps(payload, sort_keys=True))
    return payload


def _cleanup_resources(
    executable: str,
    context: str,
    namespace: str,
    run_token: str,
    attempted_kinds: list[str],
) -> dict[str, Any]:
    resources: list[dict[str, Any]] = []
    selector = f"phase2.dynamo.nvidia.com/run-token={run_token}"
    cleanup_order = [
        kind for kind in ("job", "pod", "configmap") if kind in attempted_kinds
    ]
    for kind in cleanup_order:
        long_cleanup = kind in ("job", "pod")
        delete_timeout = "150s" if long_cleanup else "30s"
        subprocess_timeout = 165 if long_cleanup else 45
        extra_args: list[str] = []
        if kind == "job":
            extra_args.append("--cascade=foreground")
        if kind == "pod":
            extra_args.append("--grace-period=120")
        result = _kubectl(
            executable,
            context,
            namespace,
            "delete",
            kind,
            "-l",
            selector,
            "--ignore-not-found=true",
            "--wait=true",
            f"--timeout={delete_timeout}",
            *extra_args,
            check=False,
            timeout_seconds=subprocess_timeout,
        )
        record: dict[str, Any] = {
            "kind": kind,
            "selector": selector,
            "returnCode": result.returncode,
        }
        if result.returncode != 0:
            record["stderr"] = result.stderr[-2000:]
        verify = _kubectl(
            executable,
            context,
            namespace,
            "get",
            kind,
            "-l",
            selector,
            "-o",
            "name",
            check=False,
            timeout_seconds=30,
        )
        record["verificationReturnCode"] = verify.returncode
        if verify.returncode == 0 and verify.stdout.strip():
            record["remaining"] = verify.stdout.splitlines()
        if verify.returncode != 0:
            record["verificationStderr"] = verify.stderr[-2000:]
        resources.append(record)
    return {
        "status": (
            "PASS"
            if all(
                item["returnCode"] == 0
                and item["verificationReturnCode"] == 0
                and not item.get("remaining")
                for item in resources
            )
            else "FAIL"
        ),
        "runToken": run_token,
        "resources": resources,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kubectl", default="kubectl")
    parser.add_argument("--context", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--gpu-product", required=True)
    parser.add_argument("--gpu-count", type=int, required=True)
    parser.add_argument("--architecture", default="amd64")
    parser.add_argument(
        "--actuator-mode",
        choices=("nvml", "nvml-dcgm-parity"),
        default="nvml-dcgm-parity",
    )
    parser.add_argument("--dcgm-image")
    parser.add_argument("--runtime-image", default="python:3.12-slim-bookworm")
    parser.add_argument("--hostengine-host")
    parser.add_argument("--hostengine-port", type=int, default=5555)
    parser.add_argument("--hostengine-namespace", default="gpu-operator")
    parser.add_argument("--hostengine-service", default="nvidia-dcgm")
    parser.add_argument("--hostengine-container", default="nvidia-dcgm-ctr")
    parser.add_argument("--expected-hostengine-version")
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--keep-resources", action="store_true")
    args = parser.parse_args()

    if args.gpu_count < 1:
        parser.error("--gpu-count must be positive")
    if not args.gpu_product.strip() or args.gpu_product.strip() != args.gpu_product:
        parser.error(
            "--gpu-product must be an exact, whitespace-normalized label value"
        )
    if args.timeout_seconds < 420:
        parser.error("--timeout-seconds must be at least 420")
    if args.actuator_mode == "nvml-dcgm-parity":
        if not args.dcgm_image:
            parser.error("parity mode requires --dcgm-image")
        if not args.hostengine_host:
            parser.error("parity mode requires --hostengine-host")
        if not args.expected_hostengine_version:
            parser.error("parity mode requires --expected-hostengine-version")

    power_agent_dir = Path(__file__).resolve().parent.parent
    overall_deadline = time.monotonic() + args.timeout_seconds
    qualification_timeout_seconds = args.timeout_seconds - 300
    source_data, source_hashes = _source_payload(power_agent_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    run_token = uuid.uuid4().hex
    name = f"p2q-{timestamp}-{run_token[:6]}"
    config_map, job_manifest = build_manifests(
        name=name,
        run_token=run_token,
        namespace=args.namespace,
        gpu_product=args.gpu_product,
        gpu_count=args.gpu_count,
        architecture=args.architecture,
        actuator_mode=args.actuator_mode,
        dcgm_image=args.dcgm_image,
        runtime_image=args.runtime_image,
        hostengine_host=args.hostengine_host,
        hostengine_port=args.hostengine_port,
        expected_hostengine_version=args.expected_hostengine_version,
        active_deadline_seconds=qualification_timeout_seconds + 180,
        qualification_timeout_seconds=qualification_timeout_seconds,
        source_data=source_data,
    )

    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    for artifact in (
        "qualification.log",
        "job.json",
        "pod.json",
        "node.json",
        "summary.json",
        "cleanup.json",
        "hostengine.json",
    ):
        (args.evidence_dir / artifact).unlink(missing_ok=True)
    _write_json(args.evidence_dir / "job-manifest.json", job_manifest)
    _write_json(args.evidence_dir / "source-sha256.json", source_hashes)

    attempted_kinds: list[str] = []
    primary_error: BaseException | None = None
    successful_summary: dict[str, Any] | None = None
    cleanup: dict[str, Any] | None = None
    try:
        permissions = [("create", "jobs.batch"), ("create", "configmaps")]
        if not args.keep_resources:
            permissions.extend(
                [
                    ("delete", "jobs.batch"),
                    ("delete", "configmaps"),
                    ("delete", "pods"),
                ]
            )
        for verb, resource in permissions:
            allowed = _kubectl(
                args.kubectl,
                args.context,
                args.namespace,
                "auth",
                "can-i",
                verb,
                resource,
                deadline=overall_deadline,
            ).stdout.strip()
            if allowed != "yes":
                raise PermissionError(
                    f"kubectl auth can-i {verb} {resource}: {allowed}"
                )

        hostengine_evidence: dict[str, Any] | None = None
        if args.actuator_mode == "nvml-dcgm-parity":
            for verb, resource in (
                ("get", "endpoints"),
                ("get", "pods"),
                ("get", "services"),
                ("create", "pods/exec"),
            ):
                allowed = _kubectl(
                    args.kubectl,
                    args.context,
                    args.hostengine_namespace,
                    "auth",
                    "can-i",
                    verb,
                    resource,
                    deadline=overall_deadline,
                ).stdout.strip()
                if allowed != "yes":
                    raise PermissionError(
                        f"kubectl auth can-i {verb} {resource} "
                        f"-n {args.hostengine_namespace}: {allowed}"
                    )

            hostengine_evidence = _verify_hostengine_endpoints(
                args.kubectl,
                args.context,
                args.hostengine_namespace,
                args.hostengine_service,
                args.hostengine_container,
                args.hostengine_port,
                str(args.expected_hostengine_version),
                str(args.hostengine_host),
                overall_deadline,
            )
            _write_json(
                args.evidence_dir / "hostengine.json",
                {"status": "PREFLIGHT_PASS", "preflight": hostengine_evidence},
            )

        attempted_kinds.append("configmap")
        _kubectl(
            args.kubectl,
            args.context,
            args.namespace,
            "create",
            "-f",
            "-",
            stdin=json.dumps(config_map),
            deadline=overall_deadline,
        )
        attempted_kinds.append("job")
        attempted_kinds.append("pod")
        _kubectl(
            args.kubectl,
            args.context,
            args.namespace,
            "create",
            "-f",
            "-",
            stdin=json.dumps(job_manifest),
            deadline=overall_deadline,
        )

        job = _wait_for_job(
            args.kubectl,
            args.context,
            args.namespace,
            name,
            overall_deadline,
        )
        selector = f"phase2.dynamo.nvidia.com/run={name}"
        pods = _json_output(
            _kubectl(
                args.kubectl,
                args.context,
                args.namespace,
                "get",
                "pods",
                "-l",
                selector,
                "-o",
                "json",
                deadline=overall_deadline,
            )
        ).get("items", [])
        if len(pods) != 1:
            raise RuntimeError(f"expected one qualification Pod, found {len(pods)}")
        pod = pods[0]
        pod_name = pod["metadata"]["name"]
        node_name = pod.get("spec", {}).get("nodeName")
        if not node_name:
            raise RuntimeError("qualification Pod was never assigned a node")
        node = _json_output(
            _kubectl(
                args.kubectl,
                args.context,
                args.namespace,
                "get",
                "node",
                node_name,
                "-o",
                "json",
                deadline=overall_deadline,
            )
        )
        observed_product = node["metadata"]["labels"].get("nvidia.com/gpu.product")
        if observed_product != args.gpu_product:
            raise RuntimeError(
                f"scheduled node product {observed_product!r} != {args.gpu_product!r}"
            )
        logs = _kubectl(
            args.kubectl,
            args.context,
            args.namespace,
            "logs",
            pod_name,
            "-c",
            "qualification",
            check=False,
            deadline=overall_deadline,
        ).stdout
        (args.evidence_dir / "qualification.log").write_text(logs, encoding="utf-8")
        _write_json(args.evidence_dir / "job.json", job)
        _write_json(args.evidence_dir / "pod.json", pod)
        _write_json(args.evidence_dir / "node.json", node)

        statuses = [
            *pod.get("status", {}).get("initContainerStatuses", []),
            *pod.get("status", {}).get("containerStatuses", []),
        ]
        images = {
            status["name"]: {
                "image": status.get("image"),
                "imageID": status.get("imageID"),
            }
            for status in statuses
        }
        required_statuses = {"python-dependencies", "qualification"}
        if args.actuator_mode == "nvml-dcgm-parity":
            required_statuses.add("dcgm-vendor")
        if set(images) != required_statuses:
            raise RuntimeError(
                f"qualification container statuses {sorted(images)} != "
                f"{sorted(required_statuses)}"
            )
        for container_name, image in images.items():
            _require_digest_image_id(
                image.get("imageID") or "", f"qualification container {container_name}"
            )
        if (
            images["python-dependencies"]["imageID"]
            != images["qualification"]["imageID"]
        ):
            raise RuntimeError("runtime init and qualification images differ")

        common_summary = {
            "completedAt": datetime.now(timezone.utc).isoformat(),
            "actuatorMode": args.actuator_mode,
            "context": args.context,
            "namespace": args.namespace,
            "job": name,
            "pod": pod_name,
            "podUID": pod["metadata"]["uid"],
            "node": node_name,
            "nodeGPUProduct": observed_product,
            "images": images,
        }
        if args.actuator_mode == "nvml-dcgm-parity":
            if hostengine_evidence is None:
                raise RuntimeError("parity hostengine preflight evidence is missing")
            postflight = _verify_hostengine_endpoints(
                args.kubectl,
                args.context,
                args.hostengine_namespace,
                args.hostengine_service,
                args.hostengine_container,
                args.hostengine_port,
                str(args.expected_hostengine_version),
                str(args.hostengine_host),
                overall_deadline,
            )
            hostengine_binding = _bind_hostengine_postflight(
                hostengine_evidence, postflight, node_name
            )
            _write_json(args.evidence_dir / "hostengine.json", hostengine_binding)
            common_summary.update(
                {
                    "dcgmImage": args.dcgm_image,
                    "hostengine": f"{args.hostengine_host}:{args.hostengine_port}",
                    "hostengineBinding": hostengine_binding,
                }
            )
        if job.get("status", {}).get("succeeded", 0) < 1:
            failure_summary = {
                **common_summary,
                "qualification": {
                    "status": "FAIL",
                    "jobStatus": job.get("status", {}),
                },
            }
            _write_json(args.evidence_dir / "summary.json", failure_summary)
            raise RuntimeError(
                "qualification Job failed; captured logs and status under "
                f"{args.evidence_dir}"
            )

        result_lines = [
            line for line in logs.splitlines() if line.startswith(RESULT_PREFIX)
        ]
        if len(result_lines) != 1:
            raise RuntimeError(
                f"expected one {RESULT_PREFIX} marker, found {len(result_lines)}"
            )
        qualification = json.loads(result_lines[0][len(RESULT_PREFIX) :])
        if qualification.get("status") != "PASS":
            raise RuntimeError(f"qualification did not pass: {qualification}")

        successful_summary = {
            **common_summary,
            "qualification": qualification,
        }
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if args.keep_resources:
            print(
                f"retained qualification resource kinds: {attempted_kinds}",
                file=sys.stderr,
            )
            cleanup = {
                "status": "RETAINED",
                "runToken": run_token,
                "resourceKinds": attempted_kinds,
            }
            _write_json(args.evidence_dir / "cleanup.json", cleanup)
        else:
            cleanup = _cleanup_resources(
                args.kubectl,
                args.context,
                args.namespace,
                run_token,
                attempted_kinds,
            )
            _write_json(args.evidence_dir / "cleanup.json", cleanup)
            if cleanup["status"] != "PASS":
                cleanup_error = RuntimeError(
                    "qualification resource cleanup failed; see "
                    f"{args.evidence_dir / 'cleanup.json'}"
                )
                if primary_error is None:
                    raise cleanup_error
                print(f"P2_8_CLEANUP_ERROR={cleanup_error}", file=sys.stderr)

    if successful_summary is None or cleanup is None:
        raise RuntimeError("successful qualification evidence is incomplete")
    _publish_successful_evidence(args.evidence_dir, successful_summary, cleanup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
