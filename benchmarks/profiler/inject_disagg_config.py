#!/usr/bin/env python3

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
import subprocess
import sys
import time
from pathlib import Path


def run_command(
    cmd: list[str], capture_output: bool = True
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


def deploy_access_pod(namespace: str) -> None:
    """Deploy the PVC access pod if it doesn't exist."""
    pod_name = "pvc-access-pod"

    # Check if pod already exists
    try:
        result = run_command(
            ["kubectl", "get", "pod", pod_name, "-n", namespace], capture_output=True
        )
        print(f"✓ Access pod '{pod_name}' already exists")
        return
    except subprocess.CalledProcessError:
        # Pod doesn't exist, deploy it
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
                return

        except subprocess.CalledProcessError:
            pass

        time.sleep(1)
        if i % 10 == 0:
            print(f"  Still waiting... ({i+1}s)")

    print("ERROR: Access pod failed to become ready within 60 seconds")
    sys.exit(1)


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
