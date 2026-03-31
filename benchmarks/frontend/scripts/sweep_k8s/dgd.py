# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
DynamoGraphDeployment helpers -- backend switch, restart, readiness.

Ported from sweep.sh functions: dgd_switch_backend, dgd_restart_frontend,
dgd_restart_graph, dgd_wait_all_ready.
"""

from __future__ import annotations

import json
import random
import subprocess
import time
import urllib.error
import urllib.request

from sweep_k8s.kubectl import (
    delete_pod,
    get_json,
    get_pod_name,
    patch_json,
    patch_merge,
    run_kubectl,
    wait_for_pod_deletion,
    wait_pod,
)

# Tokenizer backend name mapping for DGD env vars
TOKENIZER_BACKEND_MAP = {
    "hf": "default",
    "default": "default",
    "fast": "fast",
    "fastokens": "fast",
}


def dgd_label_selector(dgd_name: str, component_type: str) -> str:
    """Build a label selector for DGD-managed pods."""
    return (
        f"nvidia.com/dynamo-graph-deployment-name={dgd_name},"
        f"nvidia.com/dynamo-component-type={component_type}"
    )


def wait_model_ready(
    endpoint: str,
    model_name: str,
    max_wait: int = 300,
    namespace: str = "",
) -> None:
    """Wait for a model to be registered at the frontend /v1/models endpoint.

    Tries direct HTTP first. If the endpoint is not reachable from localhost
    (in-cluster DNS), falls back to kubectl run to check from inside the cluster.
    """
    print(f"  Waiting for model '{model_name}' at http://{endpoint}/v1/models...")
    waited = 0
    while True:
        # Try direct HTTP (works if endpoint is port-forwarded or localhost)
        try:
            req = urllib.request.Request(
                f"http://{endpoint}/v1/models",
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                models = data.get("data", [])
                if any(m.get("id") == model_name for m in models):
                    print(f"  Model ready (waited {waited}s)")
                    return
        except (urllib.error.URLError, json.JSONDecodeError, OSError, ValueError):
            pass

        # Fallback: kubectl-based check for in-cluster endpoints
        if namespace and _check_model_via_kubectl(endpoint, model_name, namespace):
            print(f"  Model ready via kubectl (waited {waited}s)")
            return

        time.sleep(5)
        waited += 5
        if waited >= max_wait:
            print(f"ERROR: Model not ready after {max_wait}s")
            raise TimeoutError(f"Model '{model_name}' not ready after {max_wait}s")
        if waited % 15 == 0:
            print(f"  Still waiting ({waited}s / {max_wait}s)...")


def _check_model_via_kubectl(
    endpoint: str,
    model_name: str,
    namespace: str,
) -> bool:
    """Check model readiness by running curl from inside the cluster."""
    pod_name = f"model-check-{int(time.time())}-{random.randint(0, 9999)}"
    try:
        result = subprocess.run(
            [
                "kubectl",
                "run",
                pod_name,
                "--rm",
                "-i",
                "--restart=Never",
                "-n",
                namespace,
                "--quiet",
                "--image=curlimages/curl:latest",
                "--",
                "-sf",
                f"http://{endpoint}/v1/models",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            models = data.get("data", [])
            return any(m.get("id") == model_name for m in models)
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        pass
    return False


def dgd_wait_all_ready(
    dgd_name: str,
    namespace: str,
    endpoint: str,
    model_name: str,
    max_wait: int = 300,
) -> None:
    """Wait for all DGD worker pods to be Ready, then wait for model endpoint."""
    print("  Waiting for all worker pods to be Ready...")
    retries = 3
    for attempt in range(retries):
        try:
            wait_pod(
                dgd_label_selector(dgd_name, "worker"),
                namespace,
                timeout=max_wait,
            )
            break
        except subprocess.TimeoutExpired:
            raise
        except subprocess.CalledProcessError as e:
            if attempt < retries - 1:
                print(f"  kubectl error (attempt {attempt + 1}/{retries}), retrying...")
                time.sleep(5)
            else:
                raise RuntimeError(
                    f"Worker pods not ready after {retries} retries: {e}"
                ) from e

    wait_model_ready(endpoint, model_name, max_wait, namespace=namespace)


def dgd_switch_backend(
    dgd_name: str,
    namespace: str,
    endpoint: str,
    model_name: str,
    backend: str,
) -> None:
    """Switch tokenizer backend on a DynamoGraphDeployment.

    Patches the DGD spec to set DYN_TOKENIZER_BACKEND; the Grove operator
    recreates the frontend pod automatically.
    """
    mapped_backend = TOKENIZER_BACKEND_MAP.get(backend, backend)
    print(
        f"\n--- Switching DGD tokenizer backend -> {mapped_backend} (dgd={dgd_name}) ---"
    )

    # Find the index of DYN_TOKENIZER_BACKEND in the Frontend env array
    try:
        dgd_json = get_json("dgd", dgd_name, namespace)
        env_list = (
            dgd_json.get("spec", {})
            .get("services", {})
            .get("Frontend", {})
            .get("extraPodSpec", {})
            .get("mainContainer", {})
            .get("env", [])
        )
        idx = None
        for i, env_var in enumerate(env_list):
            if env_var.get("name") == "DYN_TOKENIZER_BACKEND":
                idx = i
                break
    except Exception:
        idx = None

    # Capture the current frontend pod name BEFORE patching so we track
    # the right pod for deletion (avoids racing with the operator).
    old_pod = get_pod_name(
        dgd_label_selector(dgd_name, "frontend"),
        namespace,
    )

    if idx is not None:
        patch_json(
            "dgd",
            dgd_name,
            namespace,
            [
                {
                    "op": "replace",
                    "path": f"/spec/services/Frontend/extraPodSpec/mainContainer/env/{idx}/value",
                    "value": mapped_backend,
                }
            ],
        )
    else:
        patch_json(
            "dgd",
            dgd_name,
            namespace,
            [
                {
                    "op": "add",
                    "path": "/spec/services/Frontend/extraPodSpec/mainContainer/env/-",
                    "value": {"name": "DYN_TOKENIZER_BACKEND", "value": mapped_backend},
                }
            ],
        )

    print("  DGD patched -- waiting for frontend pod replacement...")
    if old_pod:
        print(f"  Waiting for old pod {old_pod} to terminate...")
        wait_for_pod_deletion(old_pod, namespace, timeout=120)

    # Wait for new frontend pod to be Ready
    print("  Waiting for new frontend pod to be Ready...")
    wait_pod(
        dgd_label_selector(dgd_name, "frontend"),
        namespace,
        timeout=300,
    )

    dgd_wait_all_ready(dgd_name, namespace, endpoint, model_name)


def dgd_restart_frontend(
    dgd_name: str,
    namespace: str,
    endpoint: str,
    model_name: str,
) -> None:
    """Restart only the frontend component to reset metrics counters."""
    print("  Restarting frontend pod to reset metrics counters...")

    old_pod = get_pod_name(
        dgd_label_selector(dgd_name, "frontend"),
        namespace,
    )

    if old_pod:
        delete_pod(old_pod, namespace, grace_period=5)
        print(f"  Waiting for old pod {old_pod} to terminate...")
        # Wait for delete
        try:
            run_kubectl(
                ["wait", "pod", old_pod, "--for=delete", "--timeout=90s"],
                namespace=namespace,
                check=False,
            )
        except Exception:
            pass

    print("  Waiting for new frontend pod to be Ready...")
    wait_pod(
        dgd_label_selector(dgd_name, "frontend"),
        namespace,
        timeout=300,
    )

    dgd_wait_all_ready(dgd_name, namespace, endpoint, model_name)


def dgd_restart_graph(
    dgd_name: str,
    namespace: str,
    endpoint: str,
    model_name: str,
) -> None:
    """Trigger a full DGD restart through spec.restart.

    Every run starts from a clean graph deployment state.
    """
    restart_id = f"bench-{time.strftime('%Y%m%d-%H%M%S')}-{random.randint(0, 9999)}"
    print(f"  Restarting full DGD deployment (id={restart_id})...")

    # Discover service names from the DGD spec so the restart order is correct
    # for any backend (mocker, vllm, trtllm, etc.)
    try:
        dgd_spec = get_json("dgd", dgd_name, namespace, timeout=60)
        services = list(dgd_spec.get("spec", {}).get("services", {}).keys())
        # Put workers before frontend: restart workers first, then frontend
        frontend_names = [s for s in services if s.lower() == "frontend"]
        worker_names = [s for s in services if s.lower() != "frontend"]
        restart_order = worker_names + frontend_names
    except Exception:
        restart_order = ["Frontend"]

    print(f"  Restart order: {restart_order}")

    patch_merge(
        "dgd",
        dgd_name,
        namespace,
        {
            "spec": {
                "restart": {
                    "id": restart_id,
                    "strategy": {
                        "type": "Sequential",
                        "order": restart_order,
                    },
                }
            }
        },
    )

    waited = 0
    phase = "pending"
    while True:
        try:
            state_json = get_json("dgd", dgd_name, namespace, timeout=60)
            restart_status = state_json.get("status", {}).get("restart", {})
            observed = restart_status.get("observedID", "")
            phase = restart_status.get("phase", "")

            if observed == restart_id:
                if phase == "Completed":
                    print(f"  DGD restart completed (waited {waited}s)")
                    break
                elif phase in ("Failed", "Superseded"):
                    raise RuntimeError(
                        f"DGD restart {restart_id} ended with phase={phase}"
                    )
        except (KeyError, TypeError):
            pass
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            # Transient kubectl timeout -- retry
            print(f"  kubectl transient error, retrying... ({e.__class__.__name__})")

        time.sleep(5)
        waited += 5
        if waited >= 600:
            raise TimeoutError(f"Timed out waiting for DGD restart {restart_id}")
        print(f"  Waiting for DGD restart ({waited}s / 600s)... phase={phase}")

    dgd_wait_all_ready(dgd_name, namespace, endpoint, model_name)
