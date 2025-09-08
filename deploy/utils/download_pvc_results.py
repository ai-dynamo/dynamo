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
PVC Results Download Script (generic)

Downloads files from a specified folder path inside a Kubernetes PVC into a local directory.
Creates an access pod, copies files, and exits. You can optionally exclude YAML configs.

Usage:
    python3 download_pvc_results.py --namespace <namespace> --output-dir <local_directory> \
        --folder /data/<folder/in/pvc> [--no-config]
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

try:
    from deploy.utils.kubernetes import (
        check_kubectl_access,
        cleanup_access_pod,
        ensure_clean_access_pod,
        run_command,
    )
except ModuleNotFoundError:
    # Allow running as a script: add repo root to sys.path
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from deploy.utils.kubernetes import (
        check_kubectl_access,
        cleanup_access_pod,
        ensure_clean_access_pod,
        run_command,
    )


def list_pvc_contents(
    namespace: str, pod_name: str, base_folder: str, skip_config: bool = False
) -> List[str]:
    """List contents of the PVC to identify files.

    Downloads all files under base_folder. If skip_config is True, excludes *.yaml and *.yml.
    """
    print("Scanning PVC contents...")

    # Build find command: all files
    find_cmd = [
        "kubectl",
        "exec",
        pod_name,
        "-n",
        namespace,
        "--",
        "find",
        base_folder,
        "-type",
        "f",
    ]

    # Exclude YAML files when requested
    if skip_config:
        find_cmd.extend(["-not", "-name", "*.yaml", "-not", "-name", "*.yml"])

    try:
        result = run_command(find_cmd, capture_output=True)
        files = [f.strip() for f in result.stdout.split("\n") if f.strip()]
        config_note = " (excluding config files)" if skip_config else ""
        print(f"Found {len(files)} files to download{config_note}")
        return files
    except subprocess.CalledProcessError:
        print("ERROR: Failed to list PVC contents")
        sys.exit(1)


def download_files(
    namespace: str, pod_name: str, files: List[str], output_dir: Path, base_folder: str
) -> None:
    """Download relevant files from PVC to local directory."""
    if not files:
        print("No files to download")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(files)} files to {output_dir}")

    downloaded = 0
    failed = 0

    for file_path in files:
        try:
            # Determine relative path and create local structure based on base_folder
            prefix = base_folder.rstrip("/") + "/"
            rel_path = (
                file_path[len(prefix) :]
                if file_path.startswith(prefix)
                else file_path.lstrip("/")
            )

            # Validate relative path
            if ".." in rel_path or rel_path.startswith("/"):
                print(f"  WARNING: Skipping potentially unsafe path: {file_path}")
                failed += 1
                continue

            local_file = output_dir / rel_path

            # Ensure the file is within output_dir
            if not local_file.resolve().is_relative_to(output_dir.resolve()):
                print(f"  WARNING: Skipping file outside output directory: {file_path}")
                failed += 1
                continue

            local_file.parent.mkdir(parents=True, exist_ok=True)

            # Download file
            run_command(
                [
                    "kubectl",
                    "cp",
                    f"{namespace}/{pod_name}:{file_path}",
                    str(local_file),
                ],
                capture_output=True,
            )

            downloaded += 1
            if downloaded % 5 == 0:  # Progress update every 5 files
                print(f"  Downloaded {downloaded}/{len(files)} files...")

        except subprocess.CalledProcessError as e:
            print(f"  WARNING: Failed to download {file_path}: {e}")
            failed += 1

    print(f"✓ Download completed: {downloaded} successful, {failed} failed")


def main():
    parser = argparse.ArgumentParser(
        description="Download profiling results from PVC to local directory",
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
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Local directory to download results to",
    )

    parser.add_argument(
        "--no-config",
        action="store_true",
        help="Skip downloading configuration files (*.yaml, *.yml)",
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Absolute folder path in the PVC to download, must start with /data/, e.g. /data/profiling_results or /data/benchmarking_results",
    )

    args = parser.parse_args()

    # Validate folder path starts with /data/
    if not args.folder.startswith("/data/"):
        print("❌ Error: Folder path must start with '/data/'")
        print(f"   Provided: {args.folder}")
        print("   Quick Fix: Add '/data/' prefix to your path")
        print("   Examples:")
        print("     /profiling_results → /data/profiling_results")
        print("     /benchmarking_results → /data/benchmarking_results")
        print("     /configs → /data/configs")
        sys.exit(1)

    print("📥 PVC Results Download")
    print("=" * 40)

    # Validate inputs
    check_kubectl_access(args.namespace)

    # Deploy access pod
    pod_name = ensure_clean_access_pod(args.namespace)
    try:
        # List and download files
        files = list_pvc_contents(args.namespace, pod_name, args.folder, args.no_config)
        download_files(args.namespace, pod_name, files, args.output_dir, args.folder)
    finally:
        # Cleanup
        cleanup_access_pod(args.namespace)

    print("\n✅ Download completed!")
    print(f"📁 Results available at: {args.output_dir.absolute()}")
    print("📄 See README.md for file descriptions")


if __name__ == "__main__":
    main()
