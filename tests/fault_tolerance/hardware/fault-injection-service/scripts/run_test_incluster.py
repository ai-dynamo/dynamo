#!/usr/bin/env python3
"""Run fault injection tests in-cluster (no port-forwarding needed)

Usage:
    python scripts/run_test_incluster.py [test_file] [job_name]

Examples:
    python scripts/run_test_incluster.py
    python scripts/run_test_incluster.py examples/test_partition_worker_to_nats.py
    python scripts/run_test_incluster.py examples/test_partition_frontend_to_nats.py
    python scripts/run_test_incluster.py examples/my_custom_test.py my-job
"""
import base64
import subprocess
import sys
import time
from pathlib import Path


def main():
    base_dir = Path(__file__).parent.parent

    # Parse arguments
    test_file = sys.argv[1] if len(sys.argv) > 1 else "examples/test_partition_worker_to_nats.py"
    job_name = sys.argv[2] if len(sys.argv) > 2 else "fault-injection-test"

    # Resolve test file path
    test_path = base_dir / test_file if not Path(test_file).is_absolute() else Path(test_file)
    client_file = base_dir / "client/fault_injection_client.py"
    helpers_file = base_dir / "client/test_helpers.py"
    conftest_file = base_dir / "client/conftest.py"

    # Validate files exist
    if not test_path.exists():
        print(f"Error: Test file not found: {test_path}")
        print(f"\nUsage: python {Path(__file__).name} [test_file] [job_name]")
        print("\nExamples:")
        print("  python scripts/run_test_incluster.py")
        print("  python scripts/run_test_incluster.py examples/test_partition_worker_to_nats.py")
        print("  python scripts/run_test_incluster.py examples/test_partition_frontend_to_nats.py")
        print("  python scripts/run_test_incluster.py examples/my_custom_test.py my-job")
        return 1

    if not client_file.exists():
        print(f"Error: Client library not found: {client_file}")
        return 1

    if not helpers_file.exists():
        print(f"Error: Test helpers not found: {helpers_file}")
        return 1

    # Read files and base64 encode to avoid YAML parsing issues
    test_code = test_path.read_text()
    client_code = client_file.read_text()
    helpers_code = helpers_file.read_text()
    conftest_code = conftest_file.read_text() if conftest_file.exists() else ""

    test_code_b64 = base64.b64encode(test_code.encode()).decode()
    client_code_b64 = base64.b64encode(client_code.encode()).decode()
    helpers_code_b64 = base64.b64encode(helpers_code.encode()).decode()
    conftest_code_b64 = base64.b64encode(conftest_code.encode()).decode() if conftest_code else ""

    test_name = test_path.stem

    # Generate job YAML with base64 encoded Python code
    job = f"""apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
  namespace: fault-injection-system
  labels:
    test-file: {test_name}
spec:
  ttlSecondsAfterFinished: 3600
  backoffLimit: 0
  template:
    metadata:
      labels:
        kai-scheduler.nvidia.com/skip-admission: "true"
    spec:
      serviceAccountName: fault-injection-api
      restartPolicy: Never
      containers:
      - name: test
        image: python:3.12-slim
        command: ["/bin/bash", "-c"]
        args:
        - |
          pip install -q requests httpx pydantic pytest kubernetes && cd /tmp
          echo '{client_code_b64}' | base64 -d > fault_injection_client.py
          echo '{helpers_code_b64}' | base64 -d > test_helpers.py
          echo '{conftest_code_b64}' | base64 -d > conftest.py
          echo '{test_code_b64}' | base64 -d > test.py
          python test.py
        env:
        - name: API_URL
          value: "http://fault-injection-api:8080"
        - name: FRONTEND_URL
          value: "http://vllm-agg-frontend.dynamo-oviya.svc.cluster.local:8000"
        - name: APP_NAMESPACE
          value: "dynamo-oviya"
        - name: FORCE_COLOR
          value: "1"
"""

    # Run test
    print(f"Running test in-cluster: {test_path.name}")
    print(f"Job name: {job_name}\n")

    subprocess.run(
        ["kubectl", "delete", "job", job_name, "-n", "fault-injection-system"],
        stderr=subprocess.DEVNULL,
    )

    proc = subprocess.Popen(["kubectl", "apply", "-f", "-"], stdin=subprocess.PIPE, text=True)
    proc.communicate(job)

    # Wait for pod to be created (with retry logic)
    print("Waiting for pod to be created...", end="", flush=True)
    pod = None
    max_wait = 60  # seconds
    check_interval = 2  # seconds
    elapsed = 0

    while elapsed < max_wait:
        time.sleep(check_interval)
        elapsed += check_interval
        print(".", end="", flush=True)

        result = subprocess.run(
            [
                "kubectl",
                "get",
                "pod",
                "-n",
                "fault-injection-system",
                "-l",
                f"job-name={job_name}",
                "-o",
                "name",
            ],
            capture_output=True,
            text=True,
        )
        pod = result.stdout.strip()

        if pod:
            print(f" found! ({elapsed}s)")
            break
    else:
        print(f" timeout after {max_wait}s")

    if pod:
        # Strip 'pod/' prefix if present (from kubectl get pod -o name)
        pod_name = pod.replace("pod/", "")
        
        print(f"\n{'='*70}")
        subprocess.run(["kubectl", "logs", "-f", pod_name, "-n", "fault-injection-system"])
        print(f"{'='*70}\n")

        status = subprocess.run(
            [
                "kubectl",
                "get",
                "job",
                job_name,
                "-n",
                "fault-injection-system",
                "-o",
                "jsonpath={.status.conditions[0].type}",
            ],
            capture_output=True,
            text=True,
        ).stdout

        print(f"Job: {job_name}")
        print(f"Pod: {pod}")
        print(f"Cleanup: kubectl delete job {job_name} -n fault-injection-system\n")

        return 0 if status == "Complete" else 1

    print(f"\nError: Could not find pod for job {job_name}")
    print("\nDebugging information:")
    print("-" * 70)

    # Show job status
    print("\nJob status:")
    subprocess.run(["kubectl", "get", "job", job_name, "-n", "fault-injection-system"])

    # Show job description for conditions
    print("\nJob conditions:")
    subprocess.run(
        [
            "kubectl",
            "get",
            "job",
            job_name,
            "-n",
            "fault-injection-system",
            "-o",
            "jsonpath={.status.conditions}",
        ]
    )
    print()

    # Show recent events
    print("\nRecent events in namespace:")
    subprocess.run(
        [
            "kubectl",
            "get",
            "events",
            "-n",
            "fault-injection-system",
            "--sort-by=.lastTimestamp",
            "--field-selector",
            f"involvedObject.name={job_name}",
        ]
    )

    print("-" * 70)
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
