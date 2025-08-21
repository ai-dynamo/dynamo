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
import time
from pathlib import Path
from typing import List


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
    pod_name = "pvc-access-pod"

    # Check if pod already exists and is running
    try:
        result = subprocess.run(
            [
                "kubectl",
                "get",
                "pod",
                pod_name,
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
            print(f"✓ Access pod '{pod_name}' already running")
            return pod_name
    except Exception:
        # Pod doesn't exist or isn't running
        pass

    print(f"Deploying access pod '{pod_name}' in namespace '{namespace}'...")

    # Get the directory where this script is located
    script_dir = Path(__file__).parent.parent
    pod_yaml_path = script_dir / "deploy" / "pvc-access-pod.yaml"

    if not pod_yaml_path.exists():
        print(f"ERROR: Pod YAML not found at {pod_yaml_path}")
        sys.exit(1)

    # Deploy the pod
    run_command(
        ["kubectl", "apply", "-f", str(pod_yaml_path), "-n", namespace],
        capture_output=False,
    )

    print("Waiting for pod to be ready...")

    # Wait for pod to be ready (up to 60 seconds)
    for i in range(60):
        try:
            result = subprocess.run(
                [
                    "kubectl",
                    "get",
                    "pod",
                    pod_name,
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
                print("✓ Access pod is ready")
                return pod_name

        except Exception:
            pass

        time.sleep(1)
        if i % 10 == 0:
            print(f"  Still waiting... ({i+1}s)")

    print("ERROR: Access pod failed to become ready within 60 seconds")
    sys.exit(1)
