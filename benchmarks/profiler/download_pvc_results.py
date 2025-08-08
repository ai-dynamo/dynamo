#!/usr/bin/env python3

"""
PVC Results Download Script

This script downloads all relevant profiling results from the profiling PVC to a local directory.
It creates the necessary access pod, downloads the files, and cleans up automatically.

Usage:
    python3 download_pvc_results.py --namespace <namespace> --output-dir <local_directory>

Examples:
    # Download to ./results directory
    python3 download_pvc_results.py --namespace <namespace> --output-dir ./results

    # Download to specific directory
    python3 download_pvc_results.py --namespace <namespace> --output-dir /home/user/profiling_data
"""

import argparse
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
        result = run_command(
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
        )

        if result.stdout.strip() == "Running":
            print(f"✓ Access pod '{pod_name}' already running")
            return pod_name
    except subprocess.CalledProcessError:
        # Pod doesn't exist or isn't running
        pass

    print(f"Deploying access pod '{pod_name}' in namespace '{namespace}'...")

    # Get the directory where this script is located
    script_dir = Path(__file__).parent
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
            result = run_command(
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
            )

            if result.stdout.strip() == "Running":
                print("✓ Access pod is ready")
                return pod_name

        except subprocess.CalledProcessError:
            pass

        time.sleep(1)
        if i % 10 == 0:
            print(f"  Still waiting... ({i+1}s)")

    print("ERROR: Access pod failed to become ready within 60 seconds")
    sys.exit(1)


def list_pvc_contents(namespace: str, pod_name: str) -> List[str]:
    """List contents of the PVC to identify relevant files."""
    print("Scanning PVC contents...")

    try:
        result = run_command(
            [
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
                "-o",
                "-name",
                "*.yaml",
                "-o",
                "-name",
                "*.yml",
            ],
            capture_output=True,
        )

        files = [f.strip() for f in result.stdout.split("\n") if f.strip()]
        print(f"Found {len(files)} relevant files to download")
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
            local_file = output_dir / rel_path
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


def download_summary_files(namespace: str, pod_name: str, output_dir: Path) -> None:
    """Download key summary files that might not match the pattern."""
    summary_files = [
        "/profiling_results/prefill_performance.png",
        "/profiling_results/decode_performance.png",
        "/profiling_results/disagg.yaml",  # In case it was injected
    ]

    print("Downloading summary files...")

    for file_path in summary_files:
        try:
            # Check if file exists first
            run_command(
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
            )

            # File exists, download it
            rel_path = file_path.replace("/profiling_results/", "")
            local_file = output_dir / rel_path
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

        except subprocess.CalledProcessError:
            # File doesn't exist or failed to download, skip silently
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

    args = parser.parse_args()

    print("📥 PVC Results Download")
    print("=" * 40)

    # Validate inputs
    check_kubectl_access(args.namespace)

    # Deploy access pod
    pod_name = deploy_access_pod(args.namespace)

    # List and download files
    files = list_pvc_contents(args.namespace, pod_name)
    download_files(args.namespace, pod_name, files, args.output_dir)

    # Download additional summary files
    download_summary_files(args.namespace, pod_name, args.output_dir)

    # Generate README
    generate_readme(args.output_dir, len(files))

    # Cleanup info
    cleanup_access_pod(args.namespace, pod_name)

    print("\n✅ Download completed!")
    print(f"📁 Results available at: {args.output_dir.absolute()}")
    print("📄 See README.md for file descriptions")


if __name__ == "__main__":
    main()
