# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Prometheus metrics capture for k8s sweeps.

Captures pre/post frontend /metrics snapshots for delta analysis.
Supports both direct HTTP (when endpoint is reachable) and kubectl-exec
(when only in-cluster DNS is available).
"""

from __future__ import annotations

import shlex
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Optional


def capture_metrics(
    endpoint: str,
    dest: Path,
    namespace: Optional[str] = None,
    pod_label: Optional[str] = None,
) -> None:
    """Capture frontend /metrics to a file.

    Tries direct HTTP first. If that fails and namespace + pod_label are
    provided, falls back to kubectl exec curl from the frontend pod.

    Args:
        endpoint: Frontend endpoint (host:port) -- may be in-cluster DNS.
        dest: Destination file path.
        namespace: K8s namespace (for kubectl exec fallback).
        pod_label: Pod label selector (for kubectl exec fallback).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Try direct HTTP first (works if port-forwarded or on same network)
    body = _try_http(endpoint)

    # Fallback: kubectl exec into the frontend pod to curl metrics
    if body is None and namespace and pod_label:
        body = _try_kubectl_exec(endpoint, namespace, pod_label)

    # Fallback 2: kubectl run a temporary pod to curl
    if body is None and namespace:
        body = _try_kubectl_run(endpoint, namespace)

    if body and body.strip():
        dest.write_text(body)
        line_count = len(body.strip().split("\n"))
        print(f"  Metrics captured -> {dest.name} ({line_count} lines)")
    else:
        msg = f"# metrics capture failed at {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
        dest.write_text(msg)
        print(f"  WARNING: could not capture metrics from {endpoint}")


def _try_http(endpoint: str) -> Optional[str]:
    """Try fetching metrics via direct HTTP."""
    try:
        req = urllib.request.Request(f"http://{endpoint}/metrics")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read().decode()
    except Exception:
        return None


def _try_kubectl_exec(
    endpoint: str,
    namespace: str,
    pod_label: str,
) -> Optional[str]:
    """Fetch metrics by exec-ing curl inside a running pod."""
    try:
        # Get a pod name from the label selector
        result = subprocess.run(
            [
                "kubectl",
                "-n",
                namespace,
                "get",
                "pod",
                "-l",
                pod_label,
                "-o",
                "jsonpath={.items[0].metadata.name}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        pod_name = result.stdout.strip()
        if not pod_name:
            return None

        # Exec curl inside the pod (curl may not be available; try wget too)
        safe_endpoint = shlex.quote(endpoint)
        result = subprocess.run(
            [
                "kubectl",
                "-n",
                namespace,
                "exec",
                pod_name,
                "--",
                "sh",
                "-c",
                f"curl -sf http://{safe_endpoint}/metrics 2>/dev/null || "
                f"wget -qO- http://{safe_endpoint}/metrics 2>/dev/null || "
                f'python3 -c "import urllib.request,sys; print(urllib.request.urlopen(sys.argv[1]).read().decode())" http://{safe_endpoint}/metrics 2>/dev/null',
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    except Exception:
        pass
    return None


def _try_kubectl_run(endpoint: str, namespace: str) -> Optional[str]:
    """Fetch metrics via a one-shot kubectl run --rm pod."""
    try:
        result = subprocess.run(
            [
                "kubectl",
                "run",
                "metrics-fetch",
                "--rm",
                "-i",
                "--restart=Never",
                "-n",
                namespace,
                "--image=curlimages/curl:latest",
                "--",
                "-sf",
                f"http://{endpoint}/metrics",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    except Exception:
        pass
    return None
