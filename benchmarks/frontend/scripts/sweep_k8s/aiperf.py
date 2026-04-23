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
    aiperf_extra: str = "",
) -> str:
    """Build the shell script that runs inside the Job container.

    ``aiperf_extra`` is appended verbatim after the canonical flags so
    callers can pass knobs like ``--prefill-concurrency 200``,
    ``--arrival-pattern gamma``, or additional ``--extra-inputs`` without
    editing this builder.
    """
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

    # Extra flags passthrough (set from K8sConfig.aiperf_extra). Empty
    # string is a no-op; otherwise the string is splat into the aiperf
    # command line as-is so users can chain multiple flags.
    extra_args = aiperf_extra.strip()

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
    --ui simple {extra_args}

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
    artifact_pvc_name: str = "model-cache",
    artifact_pvc_mount_path: str = "/model-cache",
) -> str:
    """Build the aiperf k8s Job YAML.

    Uses python:3.12-slim with pip-installed aiperf (same pattern as
    recipes/qwen3-235b-a22b-fp8/trtllm/agg/perf.yaml).

    ``artifact_pvc_name`` / ``artifact_pvc_mount_path`` control which PVC
    holds the aiperf exports + tokenizer cache. Historically hard-coded to
    ``model-cache``; overridable per namespace via ``K8sConfig``.
    ``_copy_artifacts_from_pvc`` uses the same two values to spin a helper
    pod and pull the artifacts back locally.
    """
    image_pull_secret_block = ""
    if image_pull_secret:
        image_pull_secret_block = f"""
      imagePullSecrets:
        - name: {image_pull_secret}"""

    mount = artifact_pvc_mount_path.rstrip("/")
    artifact_dir = f"{mount}/perf/{job_name}"

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
              value: {mount}
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
              value: {artifact_dir}
          volumeMounts:
            - name: artifact-pvc
              mountPath: {mount}
      volumes:
        - name: artifact-pvc
          persistentVolumeClaim:
            claimName: {artifact_pvc_name}
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
    artifact_pvc_name: str = "model-cache",
    artifact_pvc_mount_path: str = "/model-cache",
) -> bool:
    """Copy aiperf artifacts from the artifact PVC to the local filesystem.

    Spins up a temporary busybox pod that mounts the PVC, uses kubectl cp
    to extract the artifacts, then deletes the pod. The PVC is mounted
    read-only here.

    Returns True if artifacts were successfully copied and the expected
    profile_export_aiperf.json exists, False otherwise.
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    artifacts_ok = False
    helper_name = f"copy-{job_name[-20:]}"
    mount = artifact_pvc_mount_path.rstrip("/")
    pvc_path = f"{mount}/perf/{job_name}"

    def _stream_tar(src_pod: str) -> tuple[bool, str]:
        """Stream a tar of ``pvc_path`` from ``src_pod`` into ``local_dir``.

        Uses ``kubectl exec ... -- tar cf - -C <dir> .`` piped into local
        ``tar xf -``. This is substantially more robust than ``kubectl cp``
        against flaky API-server streaming, which routinely emits
        ``read message: unexpected EOF`` on large transfers. Returns
        (ok, last_stderr_tail).
        """
        last_err = ""
        max_attempts = 8
        for i in range(max_attempts):
            # kubectl exec streams tar output; pipe to local tar.
            kubectl_proc = subprocess.Popen(
                [
                    "kubectl",
                    "-n",
                    namespace,
                    "exec",
                    src_pod,
                    "--",
                    "tar",
                    "cf",
                    "-",
                    "-C",
                    pvc_path,
                    ".",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            tar_proc = subprocess.Popen(
                ["tar", "xf", "-", "-C", str(local_dir)],
                stdin=kubectl_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if kubectl_proc.stdout is not None:
                kubectl_proc.stdout.close()
            tar_out, tar_err = tar_proc.communicate(timeout=300)
            k_err = kubectl_proc.stderr.read() if kubectl_proc.stderr else b""
            kubectl_rc = kubectl_proc.wait()
            tar_rc = tar_proc.returncode

            if kubectl_rc == 0 and tar_rc == 0:
                return True, ""

            last_err = (
                f"kubectl rc={kubectl_rc} stderr={k_err.decode(errors='replace').strip()[-300:]}"
                f" | tar rc={tar_rc} stderr={tar_err.decode(errors='replace').strip()[-300:]}"
            )
            if i < max_attempts - 1:
                print(
                    f"  tar-stream from {src_pod} attempt {i + 1}/"
                    f"{max_attempts} failed: {last_err[-400:]}"
                )
                # Clean up any partial extraction before retrying so a
                # failed tar doesn't leave corrupt output around.
                for p in local_dir.glob("*"):
                    if p.is_dir():
                        import shutil
                        shutil.rmtree(p, ignore_errors=True)
                    else:
                        try:
                            p.unlink()
                        except OSError:
                            pass
                time.sleep(5)
        return False, last_err

    try:
        # Create a helper pod to access the PVC. Volume name is fixed to
        # ``artifact-pvc`` so helper-pod YAML is independent of the actual
        # PVC name (which is supplied via claimName).
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
        - name: artifact-pvc
          mountPath: {mount}
          readOnly: true
  volumes:
    - name: artifact-pvc
      persistentVolumeClaim:
        claimName: {artifact_pvc_name}
"""
        run_kubectl(["apply", "-f", "-"], namespace=namespace, input_data=helper_yaml)

        # Wait for helper pod Ready condition (container started, kubectl exec
        # targets are available). Running phase alone is not enough -- we've
        # seen ``kubectl cp`` fail right after phase=Running because the
        # container's file descriptors weren't yet attachable.
        run_kubectl(
            [
                "wait",
                f"pod/{helper_name}",
                "--for=condition=Ready",
                "--timeout=120s",
            ],
            namespace=namespace,
            check=False,
        )
        # Give tar inside busybox a beat to settle.
        time.sleep(2)

        # Wait for the final aiperf write to become visible on the PVC.
        # With RWX PVCs and cross-pod mounts the aiperf pod's final flush
        # isn't always visible to a freshly-mounted helper immediately --
        # poll for ``profile_export_aiperf.json`` before the copy instead
        # of doing a single optimistic ``kubectl cp`` that fails if the
        # file hasn't propagated yet.
        target_file = "profile_export_aiperf.json"
        pvc_ls_attempts = 15  # 15 × 2s = 30s upper bound
        pvc_ls_out = ""
        for attempt in range(pvc_ls_attempts):
            result = run_kubectl(
                ["exec", helper_name, "--", "ls", "-la", pvc_path],
                namespace=namespace,
                check=False,
            )
            pvc_ls_out = result.stdout or ""
            if target_file in pvc_ls_out:
                break
            if attempt == 0:
                print(
                    f"  PVC missing {target_file} yet; waiting for "
                    "post-aiperf filesystem sync..."
                )
            time.sleep(2)
        if pvc_ls_out:
            print(f"  PVC artifacts ({pvc_path}):")
            for line in pvc_ls_out.strip().split("\n")[:8]:
                print(f"    {line}")

        # Copy artifacts from the helper pod via tar-stream (more reliable
        # than ``kubectl cp`` on this cluster, which hits spurious tar-stream
        # EOFs even on tiny transfers).
        ok, err = _stream_tar(helper_name)
        if not ok:
            raise RuntimeError(f"tar-stream from helper failed: {err[-400:]}")
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
    timeout: int = 1200,
    aiperf_template: str = "",
    aiperf_extra: str = "",
    artifact_pvc_name: str = "model-cache",
    artifact_pvc_mount_path: str = "/model-cache",
    k8s_config: Optional[object] = None,
) -> bool:
    """Run aiperf as a k8s Job inside the namespace.

    Two modes:

    * **Built-in** (default): emits a Job running ``python:3.12-slim`` with
      aiperf pip-installed at pod start. Flags are assembled here; knobs
      not exposed as kwargs can still reach aiperf via ``aiperf_extra``
      (raw string appended to the ``aiperf profile`` command).
    * **Templated**: when ``aiperf_template`` is a non-empty path, the
      Job YAML is read from that file and rendered with
      ``string.Template.safe_substitute`` via
      :mod:`sweep_k8s.aiperf_template`. The template controls the entire
      Job spec (image, mounts, command, env, ...) -- it just must honour
      the ``ARTIFACT_PVC_NAME`` / ``ARTIFACT_PVC_DIR`` contract so the
      copy helper can retrieve the results.

    Args:
        artifact_dir: Local directory for aiperf artifacts.
        endpoint: In-cluster frontend endpoint (service:port).
        model_name: Model name for aiperf --model.
        concurrency: Concurrency level.
        isl: Input sequence length.
        namespace: K8s namespace.
        image: Container image under test (informational -- passed into
            the aiperf template as ``${IMAGE}`` so logs can record it).
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
        aiperf_template: Optional path to a user Job YAML template. When
            set, the built-in builder is bypassed.
        aiperf_extra: Raw flags appended to the aiperf command line in
            built-in mode.
        artifact_pvc_name: PVC mounted by the aiperf Job for artifact
            storage (default: ``model-cache``).
        artifact_pvc_mount_path: Mount path for ``artifact_pvc_name``.
        k8s_config: Optional ``K8sConfig`` reference; when ``aiperf_template``
            is used, the full K8sConfig is threaded into the template
            substitution so user templates can reference fields like
            ``${IMAGE_PULL_SECRET_BLOCK}``.

    Returns:
        True if aiperf succeeded, False otherwise.
    """
    # Sanitize run_id for k8s naming (lowercase, no underscores, max 63 chars)
    safe_id = run_id.lower().replace("_", "-")[:40]
    ts = str(int(time.time()))[-6:]
    job_name = f"aiperf-{safe_id}-{ts}"

    print(f"  Creating aiperf Job: {job_name} (c={concurrency} isl={isl})")

    if aiperf_template:
        # Templated path: render the user's Job YAML with the full variable
        # dict, bypass the built-in builder entirely.
        from pathlib import Path as _Path

        from sweep_k8s.aiperf_template import (
            build_aiperf_variables,
            render_aiperf_template,
        )

        if k8s_config is None:
            # Build a minimal stand-in so the template renderer still has
            # the PVC settings it expects. This keeps run_aiperf callable
            # from tests / ad-hoc scripts that do not have a SweepConfig.
            from sweep_core.models import K8sConfig as _K8sConfig

            k8s_config = _K8sConfig(
                namespace=namespace,
                image=image,
                image_pull_secret=image_pull_secret,
                aiperf_template=aiperf_template,
                aiperf_extra=aiperf_extra,
                artifact_pvc_name=artifact_pvc_name,
                artifact_pvc_mount_path=artifact_pvc_mount_path,
                export_level=export_level,
            )

        variables = build_aiperf_variables(
            job_name=job_name,
            run_id=run_id,
            namespace=namespace,
            endpoint=endpoint,
            model_name=model_name,
            concurrency=concurrency,
            isl=isl,
            osl=osl,
            benchmark_duration=benchmark_duration,
            num_requests=num_requests,
            request_rate=request_rate,
            warmup_request_count=warmup_count,
            warmup_duration=warmup_duration,
            export_level=export_level,
            k8s=k8s_config,  # type: ignore[arg-type]
            hf_token_secret_name=hf_token_secret_name,
            image=image,
        )
        template_path = _Path(aiperf_template)
        if not template_path.exists():
            print(f"  ERROR: aiperf template not found: {template_path}")
            return False
        print(f"  Rendering aiperf template: {template_path.name}")
        job_yaml = render_aiperf_template(template_path, variables)
    else:
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
            aiperf_extra=aiperf_extra,
        )

        job_yaml = _build_job_yaml(
            job_name=job_name,
            namespace=namespace,
            script=script,
            image_pull_secret=image_pull_secret,
            hf_token_secret_name=hf_token_secret_name,
            artifact_pvc_name=artifact_pvc_name,
            artifact_pvc_mount_path=artifact_pvc_mount_path,
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
    artifacts_ok = _copy_artifacts_from_pvc(
        job_name,
        namespace,
        artifact_dir,
        artifact_pvc_name=artifact_pvc_name,
        artifact_pvc_mount_path=artifact_pvc_mount_path,
    )
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
