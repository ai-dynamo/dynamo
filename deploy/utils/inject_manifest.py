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
Manifest Injection Script

Copies any Kubernetes manifest file into the PVC for later use by jobs.
Both the source manifest path and destination path in the PVC are required.

Usage:
    python3 inject_manifest.py --namespace <namespace> --src <local_manifest.yaml> --dest <absolute_path_in_pvc>

Examples:
    python3 inject_manifest.py --namespace <ns> --src ./my-disagg.yaml --dest /configs/disagg.yaml
    python3 inject_manifest.py --namespace <ns> --src ./my-agg.yaml    --dest /configs/agg.yaml
"""

import argparse
import sys
from pathlib import Path

from deploy.utils.kubernetes import (
    PVC_ACCESS_POD_NAME,
    check_kubectl_access,
    cleanup_access_pod,
    deploy_access_pod,
    run_command,
)


def copy_manifest(namespace: str, manifest_path: Path, target_path: str) -> None:
    """Copy a manifest file into the PVC via the access pod."""
    pod_name = PVC_ACCESS_POD_NAME

    if not manifest_path.exists():
        print(f"ERROR: Manifest file not found: {manifest_path}")
        sys.exit(1)

    print(f"Copying {manifest_path} to {target_path} in PVC...")

    # Ensure destination directory exists
    target_dir = str(Path(target_path).parent)
    run_command(
        ["kubectl", "exec", pod_name, "-n", namespace, "--", "mkdir", "-p", target_dir],
        capture_output=False,
    )

    # Copy file to pod
    run_command(
        [
            "kubectl",
            "cp",
            str(manifest_path),
            f"{namespace}/{pod_name}:{target_path}",
        ],
        capture_output=False,
    )

    # Verify the file was copied
    result = run_command(
        ["kubectl", "exec", pod_name, "-n", namespace, "--", "ls", "-la", target_path],
        capture_output=True,
    )

    print("✓ Manifest successfully copied to PVC")
    print(f"File details: {result.stdout.strip()}")


def main():
    parser = argparse.ArgumentParser(
        description="Inject a Kubernetes manifest into the PVC",
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
        "--src", required=True, type=Path, help="Path to manifest file to copy"
    )
    parser.add_argument(
        "--dest",
        required=True,
        help="Absolute target path in PVC (e.g., /profiling_results/agg.yaml)",
    )

    args = parser.parse_args()

    # Validate target_path to prevent directory traversal
    if not args.dest.startswith("/"):
        print(
            "ERROR: Target path must be an absolute path inside the PVC (start with '/')."
        )
        sys.exit(1)

    if ".." in args.dest:
        print("ERROR: Target path cannot contain '..'")
        sys.exit(1)

    print("🚀 Manifest Injection")
    print("=" * 40)

    # Validate inputs
    check_kubectl_access(args.namespace)

    # Deploy access pod
    deploy_access_pod(args.namespace)
    try:
        # Copy manifest
        copy_manifest(args.namespace, args.src, args.dest)
        print("\n✅ Manifest injection completed!")
        print(f"📁 File available at: {args.dest}")
    finally:
        # Cleanup even on failure
        cleanup_access_pod(args.namespace)


if __name__ == "__main__":
    main()
