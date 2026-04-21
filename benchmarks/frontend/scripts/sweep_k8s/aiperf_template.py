# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
aiperf Job YAML template rendering.

Companion to ``template.py`` (which renders the DGD deploy template). When
``K8sConfig.aiperf_template`` is set, the sweep runner bypasses
``aiperf._build_job_yaml`` and renders a user-supplied aiperf Job YAML file
instead. This unblocks:

  * using a pre-built aiperf image (rather than pip-installing at pod start)
  * a different artifact PVC (e.g. ``shared-model-cache``)
  * aiperf flags the built-in builder does not expose
    (``--arrival-pattern``, ``--prefill-concurrency``, custom headers,
    arbitrary ``--extra-inputs``)
  * auth, node selection, tolerations, or any other pod-spec customization

Template substitution uses ``string.Template.safe_substitute`` so missing
variables are left as literal ``${VAR}`` rather than raising KeyError. This
matches the deploy-template convention.

Supported variables (any can be referenced in the template):

  JOB_NAME                 -- unique Job metadata.name (derived from run_id)
  NAMESPACE                -- target k8s namespace
  MODEL_NAME               -- served model name, aiperf --model
  ENDPOINT                 -- in-cluster frontend endpoint (host:port)
  CONCURRENCY              -- aiperf --concurrency
  ISL / OSL                -- synthetic input / output token length
  BENCHMARK_DURATION       -- aiperf --benchmark-duration (sec) or ""
  NUM_REQUESTS             -- aiperf --request-count or ""
  REQUEST_RATE             -- aiperf --request-rate or ""
  WARMUP_REQUEST_COUNT     -- aiperf --warmup-request-count (default: CONCURRENCY)
  WARMUP_DURATION          -- aiperf --warmup-duration or ""
  EXPORT_LEVEL             -- summary | records | raw
  AIPERF_EXTRA             -- raw extra flags from K8sConfig.aiperf_extra
  RUN_ID                   -- human-readable run id (for labels / logs)
  IMAGE                    -- the deployment image under test (informational)
  HF_TOKEN_SECRET_NAME     -- secret holding HF_TOKEN
  IMAGE_PULL_SECRET_BLOCK  -- rendered imagePullSecrets: block or empty

  Storage (honour these if your template mounts a PVC for artifacts):
    ARTIFACT_PVC_NAME        -- PVC to mount (from --artifact-pvc-name)
    ARTIFACT_PVC_MOUNT_PATH  -- mountPath (from --artifact-pvc-mount-path)
    ARTIFACT_PVC_DIR         -- full on-pod path where aiperf should write
                                 artifacts: <mount>/perf/<JOB_NAME>

The contract for artifact retrieval is that the template MUST mount
``ARTIFACT_PVC_NAME`` and write profile exports under ``ARTIFACT_PVC_DIR``.
``sweep_k8s/aiperf.py::_copy_artifacts_from_pvc`` spins up a helper pod
mounting the same PVC to copy results back.
"""

from __future__ import annotations

import string
from pathlib import Path
from typing import Dict, Optional

from sweep_core.models import K8sConfig


def _indent_block(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(f"{prefix}{line}" if line else "" for line in text.splitlines())


def _image_pull_secret_block(image_pull_secret: str, indent: int = 6) -> str:
    if not image_pull_secret:
        return ""
    return _indent_block(
        f"imagePullSecrets:\n  - name: {image_pull_secret}",
        indent,
    )


def build_aiperf_variables(
    *,
    job_name: str,
    run_id: str,
    namespace: str,
    endpoint: str,
    model_name: str,
    concurrency: int,
    isl: int,
    osl: int,
    benchmark_duration: Optional[int],
    num_requests: Optional[int],
    request_rate: Optional[int],
    warmup_request_count: Optional[int],
    warmup_duration: Optional[int],
    export_level: str,
    k8s: K8sConfig,
    hf_token_secret_name: str,
    image: str = "",
) -> Dict[str, str]:
    """Assemble the variable dict for ``render_aiperf_template``.

    All values are stringified; ``None`` becomes empty string so the user's
    template can conditionally include flags, e.g.
    ``${BENCHMARK_DURATION:+--benchmark-duration ${BENCHMARK_DURATION}}``
    (shell-side conditional inside the Job's command).
    """
    effective_warmup = (
        warmup_request_count if warmup_request_count is not None else concurrency
    )
    mount_path = k8s.artifact_pvc_mount_path.rstrip("/")
    artifact_pvc_dir = f"{mount_path}/perf/{job_name}"

    return {
        "JOB_NAME": job_name,
        "RUN_ID": run_id,
        "NAMESPACE": namespace,
        "MODEL_NAME": model_name,
        "ENDPOINT": endpoint,
        "CONCURRENCY": str(concurrency),
        "ISL": str(isl),
        "OSL": str(osl),
        "BENCHMARK_DURATION": str(benchmark_duration) if benchmark_duration else "",
        "NUM_REQUESTS": str(num_requests) if num_requests else "",
        "REQUEST_RATE": str(request_rate) if request_rate else "",
        "WARMUP_REQUEST_COUNT": str(effective_warmup),
        "WARMUP_DURATION": str(warmup_duration) if warmup_duration else "",
        "EXPORT_LEVEL": export_level,
        "AIPERF_EXTRA": k8s.aiperf_extra,
        "IMAGE": image,
        "HF_TOKEN_SECRET_NAME": hf_token_secret_name,
        "IMAGE_PULL_SECRET_BLOCK": _image_pull_secret_block(k8s.image_pull_secret),
        "ARTIFACT_PVC_NAME": k8s.artifact_pvc_name,
        "ARTIFACT_PVC_MOUNT_PATH": mount_path,
        "ARTIFACT_PVC_DIR": artifact_pvc_dir,
    }


def render_aiperf_template(template_path: Path, variables: Dict[str, str]) -> str:
    """Read an aiperf Job template and substitute ``${VAR}`` placeholders.

    Uses ``safe_substitute`` so missing variables are left as-is rather
    than raising. This allows templates to contain shell ``${FOO}`` refs
    that are resolved at pod runtime without listing them in the dict.
    """
    raw = template_path.read_text()
    return string.Template(raw).safe_substitute(variables)
