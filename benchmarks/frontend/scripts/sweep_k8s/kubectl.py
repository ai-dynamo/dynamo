# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Safe kubectl subprocess helpers.

All k8s interactions go through this module for consistent error handling
and namespace scoping.
"""

from __future__ import annotations

import json
import subprocess
import time
from typing import Any, Dict, List, Optional


def run_kubectl(
    args: List[str],
    namespace: Optional[str] = None,
    capture: bool = True,
    check: bool = True,
    timeout: int = 60,
    input_data: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """Run a kubectl command with namespace scoping and error handling.

    Args:
        args: kubectl arguments (e.g., ["get", "pods"]).
        namespace: K8s namespace (prepended as -n <namespace>).
        capture: Whether to capture stdout/stderr.
        check: Whether to raise on non-zero exit.
        timeout: Command timeout in seconds.
        input_data: Optional stdin input.

    Returns:
        CompletedProcess result.
    """
    cmd = ["kubectl"]
    if namespace:
        cmd.extend(["-n", namespace])
    cmd.extend(args)

    result = subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        check=False,
        timeout=timeout,
        input=input_data,
    )
    if check and result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else ""
        print(f"  kubectl error (rc={result.returncode}): {' '.join(args[:4])}")
        if stderr:
            print(f"    {stderr}")
        result.check_returncode()
    return result


def get_json(
    resource: str,
    name: str,
    namespace: str,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Get a k8s resource as a parsed JSON dict."""
    result = run_kubectl(
        ["get", resource, name, "-o", "json"],
        namespace=namespace,
        timeout=timeout,
    )
    return json.loads(result.stdout)


def patch_json(
    resource: str,
    name: str,
    namespace: str,
    patch: List[Dict[str, Any]],
    timeout: int = 30,
) -> None:
    """Apply a JSON patch to a k8s resource."""
    patch_str = json.dumps(patch)
    run_kubectl(
        ["patch", resource, name, "--type=json", f"-p={patch_str}"],
        namespace=namespace,
        timeout=timeout,
    )


def patch_merge(
    resource: str,
    name: str,
    namespace: str,
    patch: Dict[str, Any],
    timeout: int = 30,
) -> None:
    """Apply a strategic merge patch to a k8s resource."""
    patch_str = json.dumps(patch)
    run_kubectl(
        ["patch", resource, name, "--type=merge", f"-p={patch_str}"],
        namespace=namespace,
        timeout=timeout,
    )


def wait_pod(
    label_selector: str,
    namespace: str,
    condition: str = "Ready",
    timeout: int = 300,
) -> None:
    """Wait for pod(s) matching a label selector to reach a condition."""
    run_kubectl(
        [
            "wait",
            "pod",
            "-l",
            label_selector,
            f"--for=condition={condition}",
            f"--timeout={timeout}s",
        ],
        namespace=namespace,
        timeout=timeout + 10,
    )


def delete_pod(
    name: str,
    namespace: str,
    grace_period: int = 5,
) -> None:
    """Delete a pod by name."""
    run_kubectl(
        ["delete", "pod", name, f"--grace-period={grace_period}"],
        namespace=namespace,
        check=False,
    )


def get_pod_name(
    label_selector: str,
    namespace: str,
) -> Optional[str]:
    """Get the name of the first pod matching a label selector."""
    result = run_kubectl(
        [
            "get",
            "pod",
            "-l",
            label_selector,
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ],
        namespace=namespace,
        check=False,
    )
    name = result.stdout.strip()
    return name if name else None


def pod_exists(name: str, namespace: str) -> bool:
    """Check if a pod exists."""
    result = run_kubectl(
        ["get", "pod", name],
        namespace=namespace,
        check=False,
    )
    return result.returncode == 0


def apply_yaml(yaml_content: str, namespace: str) -> None:
    """Apply YAML content via kubectl apply -f -."""
    run_kubectl(
        ["apply", "-f", "-"],
        namespace=namespace,
        input_data=yaml_content,
    )


def apply_secret_literal(name: str, namespace: str, key: str, value: str) -> None:
    """Create or update an opaque Secret from a literal value."""
    secret_yaml = f"""apiVersion: v1
kind: Secret
metadata:
  name: {name}
type: Opaque
stringData:
  {key}: {json.dumps(value)}
"""
    apply_yaml(secret_yaml, namespace)


def wait_for_pod_deletion(
    name: str,
    namespace: str,
    timeout: int = 120,
) -> None:
    """Wait for a pod to be deleted."""
    waited = 0
    while pod_exists(name, namespace):
        time.sleep(5)
        waited += 5
        if waited >= timeout:
            print(f"  WARNING: pod {name} still present after {timeout}s")
            break
