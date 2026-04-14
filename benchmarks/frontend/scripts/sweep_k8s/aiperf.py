# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
K8s aiperf Job launcher.

Runs aiperf as a k8s Job inside the same namespace as the DGD, using the
in-cluster service DNS endpoint. Uses python:3.12-slim with pip-installed
aiperf (same pattern as recipes/qwen3-235b-a22b-fp8/trtllm/agg/perf.yaml).

Artifacts are written inside the pod, then copied back to the local host
via kubectl cp.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Optional

from sweep_k8s.kubectl import run_kubectl

DEFAULT_HF_TOKEN_SECRET_NAME = "hf-token-secret"


def _build_aiperf_script(
    model_name: str,
    endpoint: str,
    concurrency: int,
    isl: int,
    osl: int = 256,
    benchmark_duration: Optional[int] = None,
    num_requests: Optional[int] = None,
    request_rate: Optional[int] = None,
    warmup_duration: Optional[int] = None,
    warmup_count: Optional[int] = None,
    export_level: str = "summary",
) -> str:
    """Build the shell script that runs inside the Job container."""
    # Build load-control args
    load_args = ""
    if benchmark_duration:
        load_args += f" --benchmark-duration {benchmark_duration}"
    if num_requests:
        load_args += f" --request-count {num_requests}"
    if request_rate:
        load_args += f" --request-rate {request_rate}"
    if not load_args.strip():
        auto_count = max(concurrency * 20, 640)
        load_args = f" --request-count {auto_count}"

    # Warmup args
    warmup_args = ""
    if warmup_duration:
        warmup_args = f" --warmup-duration {warmup_duration}"
    elif warmup_count:
        warmup_args = f" --warmup-request-count {warmup_count}"
    else:
        warmup_args = f" --warmup-request-count {concurrency}"

    return f"""set -e
apt-get update -qq && apt-get install -y -qq curl jq git procps 2>/dev/null
pip install --quiet git+https://github.com/ai-dynamo/aiperf.git@54cd6dc820bff8bfebc875da104e59d745e14f75
echo "aiperf installed"

# Wait for model
echo "Waiting for model '{model_name}' at http://{endpoint}/v1/models..."
while ! curl -sf "http://{endpoint}/v1/models" 2>/dev/null | \\
      jq -e --arg m "{model_name}" '.data[]? | select(.id == $m)' >/dev/null 2>&1; do
    echo "  Model not ready, sleeping 5s..."
    sleep 5
done
echo "Model ready!"

# Write artifacts to PVC so they persist after pod completion
ARTIFACT_DIR="${{ARTIFACT_PVC_DIR:-/model-cache/perf/${{JOB_NAME}}}}"
mkdir -p "$ARTIFACT_DIR"
echo "Running aiperf: c={concurrency} isl={isl} osl={osl}"
echo "Artifact dir: $ARTIFACT_DIR"
aiperf profile \\
    --artifact-dir "$ARTIFACT_DIR" \\
    --model "{model_name}" \\
    --tokenizer "{model_name}" \\
    --endpoint-type chat \\
    --endpoint /v1/chat/completions \\
    --streaming \\
    --url "http://{endpoint}" \\
    --synthetic-input-tokens-mean {isl} \\
    --synthetic-input-tokens-stddev 0 \\
    --output-tokens-mean {osl} \\
    --output-tokens-stddev 0 \\
    --extra-inputs "max_tokens:{osl}" \\
    --extra-inputs "min_tokens:{osl}" \\
    --extra-inputs "ignore_eos:true" \\
    --extra-inputs "repetition_penalty:1.0" \\
    --extra-inputs "temperature:0.0" \\
    --concurrency {concurrency} \\
    {load_args.strip()} \\
    {warmup_args.strip()} \\
    --num-dataset-entries 12800 \\
    --random-seed 100 \\
    --workers-max {concurrency} \\
    --record-processors 32 \\
    --export-level {export_level} \\
    --ui simple

echo "aiperf done. Artifacts:"
ls -la "$ARTIFACT_DIR"/
"""


def _indent(text: str, spaces: int) -> str:
    """Indent each line of text by N spaces."""
    prefix = " " * spaces
    return "\n".join(prefix + line for line in text.split("\n"))


def _build_job_yaml(
    job_name: str,
    namespace: str,
    script: str,
    image_pull_secret: str = "",
    hf_token_secret_name: str = DEFAULT_HF_TOKEN_SECRET_NAME,
) -> str:
    """Build the aiperf k8s Job YAML.

    Uses python:3.12-slim with pip-installed aiperf (same pattern as
    recipes/qwen3-235b-a22b-fp8/trtllm/agg/perf.yaml).
    """
    image_pull_secret_block = ""
    if image_pull_secret:
        image_pull_secret_block = f"""
      imagePullSecrets:
        - name: {image_pull_secret}"""

    return f"""apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
  namespace: {namespace}
  labels:
    app: sweep-aiperf
spec:
  backoffLimit: 0
  completions: 1
  parallelism: 1
  ttlSecondsAfterFinished: 600
  template:
    metadata:
      labels:
        app: sweep-aiperf
        job-name: {job_name}
    spec:
      restartPolicy: Never
{image_pull_secret_block}
      securityContext:
        sysctls:
          - name: net.ipv4.ip_local_port_range
            value: "1024 65000"
      containers:
        - name: aiperf
          image: python:3.12-slim
          imagePullPolicy: IfNotPresent
          securityContext:
            allowPrivilegeEscalation: false
          command:
            - /bin/bash
            - -c
            - |
{_indent(script, 14)}
          env:
            - name: HF_HOME
              value: /model-cache
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: {hf_token_secret_name}
                  key: HF_TOKEN
            - name: PYTHONUNBUFFERED
              value: "1"
            - name: AIPERF_HTTP_CONNECTION_LIMIT
              value: "512"
            - name: JOB_NAME
              value: {job_name}
            - name: ARTIFACT_PVC_DIR
              value: /model-cache/perf/{job_name}
          volumeMounts:
            - name: model-cache
              mountPath: /model-cache
      volumes:
        - name: model-cache
          persistentVolumeClaim:
            claimName: model-cache
"""


def _wait_for_job(
    job_name: str,
    namespace: str,
    timeout: int = 600,
) -> bool:
    """Poll for Job completion. Returns True if succeeded."""
    waited = 0
    while waited < timeout:
        try:
            result = run_kubectl(
                ["get", "job", job_name, "-o", "json"],
                namespace=namespace,
                check=False,
            )
            if result.returncode != 0:
                time.sleep(5)
                waited += 5
                continue

            job_data = json.loads(result.stdout)
            conditions = job_data.get("status", {}).get("conditions", [])
            for cond in conditions:
                if cond.get("type") == "Complete" and cond.get("status") == "True":
                    print(f"  aiperf Job completed (waited {waited}s)")
                    return True
                if cond.get("type") == "Failed" and cond.get("status") == "True":
                    print(f"  aiperf Job FAILED (waited {waited}s)")
                    _print_job_logs(job_name, namespace)
                    return False
        except (json.JSONDecodeError, subprocess.SubprocessError, OSError) as e:
            print(f"  Transient error polling job {job_name} in {namespace}: {e}")

        time.sleep(5)
        waited += 5
        if waited % 30 == 0:
            print(f"  aiperf Job running ({waited}s / {timeout}s)...")

    print(f"  aiperf Job timed out after {timeout}s")
    _print_job_logs(job_name, namespace)
    return False


def _print_job_logs(job_name: str, namespace: str, tail: int = 20) -> None:
    """Print last N lines of the Job pod logs."""
    result = run_kubectl(
        ["logs", f"job/{job_name}", f"--tail={tail}"],
        namespace=namespace,
        check=False,
    )
    if result.stdout:
        print(f"  --- Last {tail} lines of aiperf logs ---")
        for line in result.stdout.strip().split("\n"):
            print(f"    {line}")


def _get_job_pod_name(job_name: str, namespace: str) -> Optional[str]:
    """Get the pod name for a Job."""
    result = run_kubectl(
        [
            "get",
            "pods",
            "-l",
            f"job-name={job_name}",
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ],
        namespace=namespace,
        check=False,
    )
    name = result.stdout.strip()
    return name if name else None


def _copy_artifacts_from_pvc(
    job_name: str,
    namespace: str,
    local_dir: Path,
) -> bool:
    """Copy aiperf artifacts from the model-cache PVC to the local filesystem.

    Spins up a temporary busybox pod that mounts the PVC, uses kubectl cp
    to extract the artifacts, then deletes the pod.

    Returns True if artifacts were successfully copied and the expected
    profile_export_aiperf.json exists, False otherwise.
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    artifacts_ok = False
    helper_name = f"copy-{job_name[-20:]}"
    pvc_path = f"/model-cache/perf/{job_name}"

    try:
        # Create a helper pod to access the PVC
        helper_yaml = f"""apiVersion: v1
kind: Pod
metadata:
  name: {helper_name}
  namespace: {namespace}
spec:
  restartPolicy: Never
  containers:
    - name: copy
      image: busybox:latest
      command: ["sh", "-c", "echo ready && sleep 300"]
      volumeMounts:
        - name: model-cache
          mountPath: /model-cache
          readOnly: true
  volumes:
    - name: model-cache
      persistentVolumeClaim:
        claimName: model-cache
"""
        run_kubectl(["apply", "-f", "-"], namespace=namespace, input_data=helper_yaml)

        # Wait for helper pod to be ready
        for _ in range(30):
            result = run_kubectl(
                ["get", "pod", helper_name, "-o", "jsonpath={.status.phase}"],
                namespace=namespace,
                check=False,
            )
            if result.stdout.strip() == "Running":
                break
            time.sleep(2)

        # List what's on the PVC
        result = run_kubectl(
            ["exec", helper_name, "--", "ls", "-la", pvc_path],
            namespace=namespace,
            check=False,
        )
        if result.stdout:
            print(f"  PVC artifacts ({pvc_path}):")
            for line in result.stdout.strip().split("\n")[:6]:
                print(f"    {line}")

        # Copy artifacts locally
        subprocess.run(
            [
                "kubectl",
                "cp",
                f"{namespace}/{helper_name}:{pvc_path}/",
                str(local_dir) + "/",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        files = list(local_dir.glob("*"))
        print(f"  Copied {len(files)} artifact files to local")
        for f in sorted(files)[:5]:
            print(f"    {f.name} ({f.stat().st_size} bytes)")

        expected = local_dir / "profile_export_aiperf.json"
        if expected.exists() and expected.stat().st_size > 0:
            artifacts_ok = True
        else:
            print(f"  WARNING: expected artifact missing or empty: {expected.name}")

    except Exception as e:
        print(f"  WARNING: artifact copy failed: {e}")
    finally:
        # Cleanup helper pod
        run_kubectl(
            ["delete", "pod", helper_name, "--ignore-not-found", "--grace-period=0"],
            namespace=namespace,
            check=False,
        )

    return artifacts_ok


def run_aiperf(
    artifact_dir: Path,
    endpoint: str,
    model_name: str,
    concurrency: int,
    isl: int,
    namespace: str,
    image: str,
    run_id: str,
    osl: int = 256,
    benchmark_duration: Optional[int] = None,
    num_requests: Optional[int] = None,
    request_rate: Optional[int] = None,
    warmup_duration: Optional[int] = None,
    warmup_count: Optional[int] = None,
    export_level: str = "summary",
    image_pull_secret: str = "",
    hf_token_secret_name: str = DEFAULT_HF_TOKEN_SECRET_NAME,
    timeout: int = 600,
) -> bool:
    """Run aiperf as a k8s Job inside the namespace.

    Creates a Job with python:3.12-slim, installs aiperf via pip, runs the
    benchmark against the in-cluster service endpoint, then copies artifacts
    back to the local filesystem.

    Args:
        artifact_dir: Local directory for aiperf artifacts.
        endpoint: In-cluster frontend endpoint (service:port).
        model_name: Model name for aiperf --model.
        concurrency: Concurrency level.
        isl: Input sequence length.
        namespace: K8s namespace.
        image: Container image (unused -- uses python:3.12-slim).
        run_id: Unique run identifier (used in Job name).
        osl: Output sequence length.
        benchmark_duration: Optional benchmark duration in seconds.
        num_requests: Optional request count.
        request_rate: Optional request rate limit.
        warmup_duration: Optional warmup duration in seconds.
        warmup_count: Optional warmup request count.
        export_level: aiperf export level (summary, records, raw).
        image_pull_secret: Optional image pull secret for the Job pod.
        hf_token_secret_name: Secret name that stores HF_TOKEN.
        timeout: Job timeout in seconds.

    Returns:
        True if aiperf succeeded, False otherwise.
    """
    # Sanitize run_id for k8s naming (lowercase, no underscores, max 63 chars)
    safe_id = run_id.lower().replace("_", "-")[:40]
    ts = str(int(time.time()))[-6:]
    job_name = f"aiperf-{safe_id}-{ts}"

    print(f"  Creating aiperf Job: {job_name} (c={concurrency} isl={isl})")

    script = _build_aiperf_script(
        model_name=model_name,
        endpoint=endpoint,
        concurrency=concurrency,
        isl=isl,
        osl=osl,
        benchmark_duration=benchmark_duration,
        num_requests=num_requests,
        request_rate=request_rate,
        warmup_duration=warmup_duration,
        warmup_count=warmup_count,
        export_level=export_level,
    )

    job_yaml = _build_job_yaml(
        job_name=job_name,
        namespace=namespace,
        script=script,
        image_pull_secret=image_pull_secret,
        hf_token_secret_name=hf_token_secret_name,
    )

    # Create the Job
    try:
        run_kubectl(
            ["apply", "-f", "-"],
            namespace=namespace,
            input_data=job_yaml,
        )
    except Exception as e:
        print(f"  ERROR: Failed to create aiperf Job: {e}")
        return False

    # Wait for completion
    success = _wait_for_job(job_name, namespace, timeout=timeout)

    # Copy artifacts from PVC regardless of success (partial results may exist)
    artifacts_ok = _copy_artifacts_from_pvc(job_name, namespace, artifact_dir)
    if success and not artifacts_ok:
        print("  Job succeeded but artifacts missing -- marking as failure")
        success = False

    # Print logs on failure
    if not success:
        _print_job_logs(job_name, namespace, tail=30)

    # Clean up the Job
    run_kubectl(
        ["delete", "job", job_name, "--ignore-not-found"],
        namespace=namespace,
        check=False,
    )

    return success
