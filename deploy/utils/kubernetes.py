# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import sys
from pathlib import Path
from typing import List

PVC_ACCESS_POD_NAME = "pvc-access-pod"

K8S_SA_TOKEN = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")


def is_running_in_cluster() -> bool:
    """Return True if running inside a Kubernetes cluster."""
    # Prefer well-known env var; fall back to SA token presence
    return bool(os.environ.get("KUBERNETES_SERVICE_HOST")) or K8S_SA_TOKEN.exists()


def run_command(
    cmd: List[str], capture_output: bool = True, exit_on_error: bool = True
) -> subprocess.CompletedProcess:
    """Run a command and handle errors."""
    try:
        result = subprocess.run(
            cmd, capture_output=capture_output, text=True, check=True
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed: {' '.join(cmd)}")
        print(f"Exit code: {e.returncode}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        if exit_on_error:
            sys.exit(1)
        else:
            raise


def check_kubectl_access(namespace: str) -> None:
    """Check if kubectl can access the specified namespace."""
    print(f"Checking kubectl access to namespace '{namespace}'...")
    run_command(["kubectl", "get", "pods", "-n", namespace], capture_output=True)
    print("✓ kubectl access confirmed")


def ensure_clean_access_pod(namespace: str) -> str:
    """Ensure a clean PVC access pod deployment by deleting any existing pod first."""

    # Check if pod exists and delete it if it does
    try:
        result = subprocess.run(
            [
                "kubectl",
                "get",
                "pod",
                PVC_ACCESS_POD_NAME,
                "-n",
                namespace,
                "-o",
                "jsonpath={.metadata.name}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip() == PVC_ACCESS_POD_NAME:
            print(f"Found existing access pod '{PVC_ACCESS_POD_NAME}', deleting it...")
            run_command(
                [
                    "kubectl",
                    "delete",
                    "pod",
                    PVC_ACCESS_POD_NAME,
                    "-n",
                    namespace,
                    "--ignore-not-found",
                ],
                capture_output=False,
                exit_on_error=False,
            )
            print("✓ Existing access pod deleted")
    except Exception:
        pass  # Pod doesn't exist, which is fine

    try:
        return deploy_access_pod(namespace)
    except Exception as e:
        print(f"Deployment failed: {e}")
        print(
            "Pod left running for debugging. Use 'kubectl delete pod pvc-access-pod -n <namespace>' to clean up manually."
        )
        raise


def deploy_access_pod(namespace: str) -> str:
    """Deploy the PVC access pod and return pod name."""

    # Check if pod already exists and is running
    try:
        result = subprocess.run(
            [
                "kubectl",
                "get",
                "pod",
                PVC_ACCESS_POD_NAME,
                "-n",
                namespace,
                "-o",
                "jsonpath={.status.phase}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip() == "Running":
            print(f"✓ Access pod '{PVC_ACCESS_POD_NAME}' already running")
            return PVC_ACCESS_POD_NAME
    except Exception:
        pass  # Pod doesn't exist or isn't running

    print(f"Deploying access pod '{PVC_ACCESS_POD_NAME}' in namespace '{namespace}'...")

    pod_yaml_path = Path(__file__).parent / "manifests" / "pvc-access-pod.yaml"
    if not pod_yaml_path.exists():
        print(f"ERROR: Pod YAML not found at {pod_yaml_path}")
        sys.exit(1)

    run_command(
        ["kubectl", "apply", "-f", str(pod_yaml_path), "-n", namespace],
        capture_output=False,
    )

    print("Waiting for pod to be ready...")
    run_command(
        [
            "kubectl",
            "wait",
            f"pod/{PVC_ACCESS_POD_NAME}",
            "-n",
            namespace,
            "--for=condition=Ready",
            "--timeout=60s",
        ],
        capture_output=False,
        exit_on_error=False,
    )
    print("✓ Access pod is ready")
    return PVC_ACCESS_POD_NAME


def cleanup_access_pod(namespace: str) -> None:
    print("Cleaning up access pod...")
    try:
        run_command(
            [
                "kubectl",
                "delete",
                "pod",
                PVC_ACCESS_POD_NAME,
                "-n",
                namespace,
                "--ignore-not-found",
            ],
            capture_output=False,
            exit_on_error=False,
        )
        print("✓ Access pod deleted")
    except Exception as e:
        print(f"Warning: Failed to clean up access pod: {e}")
