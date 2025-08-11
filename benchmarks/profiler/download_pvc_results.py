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
PVC Results Download Script

This script downloads all relevant profiling results from the profiling PVC to a local directory.
It creates the necessary access pod, downloads the files, and cleans up automatically.

Usage:
    python3 download_pvc_results.py --namespace <namespace> --output-dir <local_directory> [--no-config]

Examples:
    # Download to ./results directory
    python3 download_pvc_results.py --namespace <namespace> --output-dir ./results

    # Download to specific directory
    python3 download_pvc_results.py --namespace <namespace> --output-dir /home/user/profiling_data

    # Download without configuration files
    python3 download_pvc_results.py --namespace <namespace> --output-dir ./results --no-config
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List

from utils.kubernetes import check_kubectl_access, deploy_access_pod, run_command


def list_pvc_contents(
    namespace: str, pod_name: str, skip_config: bool = False
) -> List[str]:
    """List contents of the PVC to identify relevant files."""
    print("Scanning PVC contents...")

    # Build find command with optional config file exclusion
    find_cmd = [
        "kubectl",
        "exec",
        pod_name,
        "-n",
        namespace,
        "--",
        "find",
        "/profiling_results",
        "-type",
        "f",
        "-name",
        "*.png",
        "-o",
        "-name",
        "*.npz",
    ]

    # Add config file patterns if not skipping them
    if not skip_config:
        find_cmd.extend(
            [
                "-o",
                "-name",
                "*.yaml",
                "-o",
                "-name",
                "*.yml",
            ]
        )

    try:
        result = run_command(find_cmd, capture_output=True)

        files = [f.strip() for f in result.stdout.split("\n") if f.strip()]
        config_note = " (excluding config files)" if skip_config else ""
        print(f"Found {len(files)} relevant files to download{config_note}")
        return files

    except subprocess.CalledProcessError:
        print("ERROR: Failed to list PVC contents")
        sys.exit(1)


def download_files(
    namespace: str, pod_name: str, files: List[str], output_dir: Path
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
            # Determine relative path and create local structure
            rel_path = file_path.replace("/profiling_results/", "")

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


def download_summary_files(
    namespace: str, pod_name: str, output_dir: Path, skip_config: bool = False
) -> None:
    """Download key summary files that might not match the pattern."""
    summary_files = [
        "/profiling_results/prefill_performance.png",
        "/profiling_results/decode_performance.png",
    ]

    # Add config files if not skipping them
    if not skip_config:
        summary_files.append(
            "/profiling_results/disagg.yaml"
        )  # In case it was injected

    print("Downloading summary files...")

    for file_path in summary_files:
        try:
            # Check if file exists first using subprocess.run directly
            result = subprocess.run(
                [
                    "kubectl",
                    "exec",
                    pod_name,
                    "-n",
                    namespace,
                    "--",
                    "test",
                    "-f",
                    file_path,
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                # File doesn't exist, skip silently
                continue

            # File exists, download it
            rel_path = file_path.replace("/profiling_results/", "")

            # Validate relative path
            if ".." in rel_path or rel_path.startswith("/"):
                print(
                    f"  ⚠️  Skipped {file_path.split('/')[-1]}: potentially unsafe path"
                )
                continue

            local_file = output_dir / rel_path

            # Ensure the file is within output_dir
            if not local_file.resolve().is_relative_to(output_dir.resolve()):
                print(
                    f"  ⚠️  Skipped {file_path.split('/')[-1]}: outside output directory"
                )
                continue

            local_file.parent.mkdir(parents=True, exist_ok=True)

            run_command(
                [
                    "kubectl",
                    "cp",
                    f"{namespace}/{pod_name}:{file_path}",
                    str(local_file),
                ],
                capture_output=True,
            )

            print(f"  ✓ {rel_path}")

        except Exception as e:
            # File doesn't exist or failed to download, skip silently
            print(f"  ⚠️  Skipped {file_path.split('/')[-1]}: {e}")
            pass


def cleanup_access_pod(namespace: str, pod_name: str) -> None:
    """Clean up the access pod (let it auto-delete via activeDeadlineSeconds)."""
    print(f"ℹ️  Access pod '{pod_name}' will auto-delete in 5 minutes")
    print(f"   To delete immediately: kubectl delete pod {pod_name} -n {namespace}")


def generate_readme(output_dir: Path, file_count: int) -> None:
    """Generate a README file explaining the downloaded contents."""
    readme_content = f"""# Profiling Results

Downloaded {file_count} files from profiling PVC.

## File Structure

### Performance Plots
- `prefill_performance.png` - Main prefill performance across TP sizes
- `decode_performance.png` - Main decode performance across TP sizes

### Interpolation Data
- `selected_prefill_interpolation/raw_data.npz` - Prefill performance data
- `selected_prefill_interpolation/*.png` - Prefill interpolation plots
- `selected_decode_interpolation/raw_data.npz` - Decode performance data
- `selected_decode_interpolation/*.png` - Decode interpolation plots

### Configuration Files
- `disagg.yaml` - DynamoGraphDeployment configuration used for profiling

### Individual TP Results
- `prefill_tp*/` - Individual tensor parallelism profiling results
- `decode_tp*/` - Individual tensor parallelism profiling results

## Loading Data

To load the .npz data files in Python:

```python
import numpy as np

# Load prefill data
prefill_data = np.load('selected_prefill_interpolation/raw_data.npz')
print("Prefill data keys:", list(prefill_data.keys()))

# Load decode data
decode_data = np.load('selected_decode_interpolation/raw_data.npz')
print("Decode data keys:", list(decode_data.keys()))
```

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""

    readme_path = output_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)

    print("📝 Generated README.md with download summary")


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

    args = parser.parse_args()

    print("📥 PVC Results Download")
    print("=" * 40)

    # Validate inputs
    check_kubectl_access(args.namespace)

    # Deploy access pod
    pod_name = deploy_access_pod(args.namespace)

    # List and download files
    files = list_pvc_contents(args.namespace, pod_name, args.no_config)
    download_files(args.namespace, pod_name, files, args.output_dir)

    # Download additional summary files
    download_summary_files(args.namespace, pod_name, args.output_dir, args.no_config)

    # Generate README
    generate_readme(args.output_dir, len(files))

    # Cleanup info
    cleanup_access_pod(args.namespace, pod_name)

    print("\n✅ Download completed!")
    print(f"📁 Results available at: {args.output_dir.absolute()}")
    print("📄 See README.md for file descriptions")


if __name__ == "__main__":
    main()
