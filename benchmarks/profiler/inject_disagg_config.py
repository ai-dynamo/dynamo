#!/usr/bin/env python3

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

"""
Disagg Config Injection Script

This script copies a DynamoGraphDeployment disagg configuration file into the profiling PVC
so it can be used by the SLA profiler job. The profiler can then reference this config
using the DGD_CONFIG_FILE environment variable.

Usage:
    python3 inject_disagg_config.py --namespace <namespace> [--disagg-config <path>] [--target-path <path>]

Examples:
    # Use default disagg.yaml from components/backends/vllm/deploy/
    python3 inject_disagg_config.py --namespace <namespace>

    # Use custom disagg config
    python3 inject_disagg_config.py --namespace <namespace> --disagg-config ./my-custom-disagg.yaml

    # Use custom target path in PVC
    python3 inject_disagg_config.py --namespace <namespace> --target-path /profiling_results/custom-disagg.yaml
"""

import argparse
import sys
from pathlib import Path

from utils.kubernetes import check_kubectl_access, deploy_access_pod, run_command


def copy_disagg_config(
    namespace: str, disagg_config_path: Path, target_path: str
) -> None:
    """Copy the disagg config file into the PVC via the access pod."""
    pod_name = "pvc-access-pod"

    if not disagg_config_path.exists():
        print(f"ERROR: Disagg config file not found: {disagg_config_path}")
        sys.exit(1)

    print(f"Copying {disagg_config_path} to {target_path} in PVC...")

    # Copy file to pod
    run_command(
        [
            "kubectl",
            "cp",
            str(disagg_config_path),
            f"{namespace}/{pod_name}:{target_path}",
        ],
        capture_output=False,
    )

    # Verify the file was copied
    result = run_command(
        ["kubectl", "exec", pod_name, "-n", namespace, "--", "ls", "-la", target_path],
        capture_output=True,
    )

    print("✓ Disagg config successfully copied to PVC")
    print(f"File details: {result.stdout.strip()}")


def cleanup_access_pod(namespace: str, keep_pod: bool = True) -> None:
    """Optionally clean up the access pod."""
    if keep_pod:
        print("ℹ️  Access pod 'pvc-access-pod' left running for future use")
        print(
            f"   To access PVC: kubectl exec -it pvc-access-pod -n {namespace} -- /bin/bash"
        )
        print(f"   To delete pod: kubectl delete pod pvc-access-pod -n {namespace}")
    else:
        print("Cleaning up access pod...")
        run_command(
            [
                "kubectl",
                "delete",
                "pod",
                "pvc-access-pod",
                "-n",
                namespace,
                "--ignore-not-found",
            ],
            capture_output=False,
        )
        print("✓ Access pod deleted")


def main():
    parser = argparse.ArgumentParser(
        description="Inject disagg config into profiling PVC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--namespace",
        "-n",
        required=True,
        help="Kubernetes namespace containing the profiling PVC",
    )

    parser.add_argument(
        "--disagg-config",
        type=Path,
        default=Path("components/backends/vllm/deploy/disagg.yaml"),
        help="Path to disagg config file (default: components/backends/vllm/deploy/disagg.yaml)",
    )

    parser.add_argument(
        "--target-path",
        default="/profiling_results/disagg.yaml",
        help="Target path in PVC (default: /profiling_results/disagg.yaml)",
    )

    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete the access pod after copying (default: keep running)",
    )

    args = parser.parse_args()

    # Validate target_path to prevent directory traversal
    if not args.target_path.startswith("/profiling_results/"):
        print("ERROR: Target path must be within /profiling_results/")
        sys.exit(1)

    if ".." in args.target_path:
        print("ERROR: Target path cannot contain '..'")
        sys.exit(1)

    print("🚀 Disagg Config Injection")
    print("=" * 40)

    # Validate inputs
    check_kubectl_access(args.namespace)

    # Deploy access pod
    deploy_access_pod(args.namespace)

    # Copy disagg config
    copy_disagg_config(args.namespace, args.disagg_config, args.target_path)

    # Cleanup
    cleanup_access_pod(args.namespace, keep_pod=not args.cleanup)

    print("\n✅ Disagg config injection completed!")
    print(f"📁 Config available at: {args.target_path}")
    print(f"🔧 Set DGD_CONFIG_FILE={args.target_path} in your profiler job")


if __name__ == "__main__":
    main()
