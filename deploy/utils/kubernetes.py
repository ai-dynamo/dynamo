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

import subprocess
import sys
from pathlib import Path
from typing import List

PVC_ACCESS_POD_NAME = "pvc-access-pod"


def run_command(
    cmd: List[str], capture_output: bool = True
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
        sys.exit(1)


def check_kubectl_access(namespace: str) -> None:
    """Check if kubectl can access the specified namespace."""
    print(f"Checking kubectl access to namespace '{namespace}'...")
    run_command(["kubectl", "get", "pods", "-n", namespace], capture_output=True)
    print("✓ kubectl access confirmed")


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
        # Pod doesn't exist or isn't running
        pass

    print(f"Deploying access pod '{PVC_ACCESS_POD_NAME}' in namespace '{namespace}'...")

    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    pod_yaml_path = script_dir / "manifests" / "pvc-access-pod.yaml"

    if not pod_yaml_path.exists():
        print(f"ERROR: Pod YAML not found at {pod_yaml_path}")
        sys.exit(1)

    # Deploy the pod
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
    )
    print("✓ Access pod is ready")
    return PVC_ACCESS_POD_NAME


def cleanup_access_pod(namespace: str) -> None:
    print("Cleaning up access pod...")
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
    )
    print("✓ Access pod deleted")
