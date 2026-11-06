# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
K8sDgdExecutor -- DynamoGraphDeployment-based executor for k8s sweeps.

Handles DGD backend switching, restart strategies, metrics capture,
and aiperf invocation against a k8s-deployed frontend.

When --deploy-template is provided, uses template rendering instead of
DGD patching. This enables arbitrary backend deployments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from sweep_core.models import DeployDimension, RunResult, RunSpec, SweepConfig
from sweep_k8s import aiperf as k8s_aiperf
from sweep_k8s import dgd as k8s_dgd
from sweep_k8s import template as k8s_template
from sweep_k8s.kubectl import apply_secret_literal
from sweep_k8s.metrics import capture_metrics


class K8sDgdExecutor:
    """Executor for k8s sweeps using DynamoGraphDeployment."""

    def __init__(self) -> None:
        self._config: Optional[SweepConfig] = None
        self._template_path: Optional[Path] = None
        self._incluster_endpoint: str = ""  # in-cluster service DNS for aiperf Jobs

    def prepare(self, config: SweepConfig) -> None:
        """Store config and validate k8s setup."""
        self._config = config
        k8s = config.k8s

        if k8s.deploy and not k8s.deploy_template:
            raise ValueError(
                "--deploy requires --deploy-template; otherwise pre-deploy the DGD and omit --deploy"
            )
        if k8s.deploy_template and not k8s.deploy:
            raise ValueError(
                "--deploy-template mutates cluster resources; pass --deploy to allow template application"
            )

        if k8s.deploy_template:
            self._template_path = Path(k8s.deploy_template)
            if not self._template_path.exists():
                raise FileNotFoundError(
                    f"Deploy template not found: {self._template_path}"
                )
            print(f"  Using deploy template: {self._template_path}")

        if k8s.hf_token:
            print(
                f"  Updating HuggingFace token secret: {k8s_template.DEFAULT_HF_TOKEN_SECRET_NAME}"
            )
            apply_secret_literal(
                k8s_template.DEFAULT_HF_TOKEN_SECRET_NAME,
                k8s.namespace,
                "HF_TOKEN",
                k8s.hf_token,
            )

        # Compute the in-cluster endpoint for aiperf Jobs.
        # The user-provided --endpoint may be port-forwarded (e.g. localhost:18000),
        # but aiperf Jobs run inside the cluster and need the service DNS name.
        if k8s.dgd_name:
            self._incluster_endpoint = f"{k8s.dgd_name}-frontend:{k8s.frontend_port}"
        else:
            self._incluster_endpoint = k8s.endpoint
        print(f"  In-cluster endpoint for aiperf: {self._incluster_endpoint}")

        # Wait for model to be ready before starting sweep.
        # Skip when using deploy templates -- the deployment hasn't been applied yet.
        if not self._template_path:
            print("--- Pre-flight: waiting for frontend ---")
            k8s_dgd.wait_model_ready(
                self._incluster_endpoint,
                config.model_name,
                max_wait=300,
                namespace=k8s.namespace,
            )

    def apply_deploy(
        self,
        deploy: DeployDimension,
        prev: Optional[DeployDimension],
    ) -> None:
        """Apply a deployment change -- template-based or DGD patching."""
        if self._config is None:
            raise RuntimeError("prepare() must be called before apply_deploy()")
        config = self._config
        k8s = config.k8s

        if self._template_path:
            # Template-based deployment: render + apply
            k8s_template.apply_rendered_template(self._template_path, deploy, config)
            print("  Waiting for deployment to be ready...")
            k8s_dgd.wait_model_ready(
                self._incluster_endpoint,
                config.model_name,
                namespace=k8s.namespace,
                max_wait=300,
            )
            return

        # Legacy DGD patching
        if not k8s.dgd_name:
            print("  WARNING: no DGD name set for k8s mode; skipping deploy")
            return

        # Check if tokenizer changed from previous run
        if prev is not None and deploy.tokenizer != prev.tokenizer:
            # Tokenizer changed -- need to switch backend
            k8s_dgd.dgd_switch_backend(
                k8s.dgd_name,
                k8s.namespace,
                k8s.endpoint,
                config.model_name,
                deploy.tokenizer,
            )
            return

        # First run or same tokenizer -- apply reset strategy
        # (On first run the DGD is already deployed with the right backend;
        #  we just reset to get a clean baseline for metrics.)
        self._apply_reset_strategy()

    def _apply_reset_strategy(self) -> None:
        """Apply the configured reset strategy."""
        if self._config is None:
            raise RuntimeError(
                "prepare() must be called before _apply_reset_strategy()"
            )
        k8s = self._config.k8s
        strategy = k8s.reset_strategy

        if strategy == "graph":
            if k8s.dgd_name:
                k8s_dgd.dgd_restart_graph(
                    k8s.dgd_name,
                    k8s.namespace,
                    k8s.endpoint,
                    self._config.model_name,
                )
            else:
                print("  WARNING: graph reset requires --dgd-name")
        elif strategy == "frontend":
            if k8s.dgd_name:
                k8s_dgd.dgd_restart_frontend(
                    k8s.dgd_name,
                    k8s.namespace,
                    k8s.endpoint,
                    self._config.model_name,
                )
            else:
                print("  WARNING: frontend reset requires --dgd-name")
        elif strategy == "none":
            # Just wait for readiness
            if k8s.dgd_name:
                k8s_dgd.dgd_wait_all_ready(
                    k8s.dgd_name,
                    k8s.namespace,
                    k8s.endpoint,
                    self._config.model_name,
                    max_wait=60,
                )
            else:
                k8s_dgd.wait_model_ready(
                    k8s.endpoint, self._config.model_name, max_wait=60
                )

    def execute_run(self, run_spec: RunSpec, run_dir: Path) -> RunResult:
        """Execute a single k8s run: metrics capture + aiperf + post-metrics."""
        if self._config is None:
            raise RuntimeError("prepare() must be called before execute_run()")
        config = self._config
        k8s = config.k8s
        aiperf = run_spec.aiperf

        result = RunResult(run_spec=run_spec, run_dir=str(run_dir))
        run_dir.mkdir(parents=True, exist_ok=True)

        # Capture pre-run metrics (use in-cluster endpoint + kubectl exec fallback)
        frontend_label = (
            (
                f"nvidia.com/dynamo-graph-deployment-name={k8s.dgd_name},"
                f"nvidia.com/dynamo-component-type=frontend"
            )
            if k8s.dgd_name
            else None
        )
        capture_metrics(
            self._incluster_endpoint,
            run_dir / "frontend_metrics_pre.txt",
            namespace=k8s.namespace,
            pod_label=frontend_label,
        )

        # Run aiperf as a k8s Job (uses in-cluster service endpoint)
        success = k8s_aiperf.run_aiperf(
            artifact_dir=run_dir / "aiperf",
            endpoint=self._incluster_endpoint,
            model_name=config.model_name,
            concurrency=aiperf.concurrency,
            isl=aiperf.isl,
            namespace=k8s.namespace,
            image=k8s.image,
            run_id=run_spec.run_id,
            osl=aiperf.osl,
            benchmark_duration=aiperf.benchmark_duration,
            num_requests=aiperf.num_requests,
            request_rate=aiperf.request_rate,
            export_level=k8s.export_level,
            image_pull_secret=k8s.image_pull_secret,
            hf_token_secret_name=k8s_template.DEFAULT_HF_TOKEN_SECRET_NAME,
        )

        if success:
            result.status = "ok"
        else:
            result.status = "fail"

        # Capture post-run metrics
        capture_metrics(
            self._incluster_endpoint,
            run_dir / "frontend_metrics_post.txt",
            namespace=k8s.namespace,
            pod_label=frontend_label,
        )

        # Parse aiperf results
        _parse_k8s_aiperf_into_result(result, run_dir)

        return result

    def cleanup(self) -> None:
        """No persistent state to clean up."""
        pass


def _parse_k8s_aiperf_into_result(result: RunResult, run_dir: Path) -> None:
    """Parse aiperf results from k8s run directory."""
    aiperf_json = run_dir / "aiperf" / "profile_export_aiperf.json"
    if not aiperf_json.exists():
        return

    try:
        data = json.loads(aiperf_json.read_text())
        rt = data.get("request_throughput", {})
        result.req_per_sec = rt.get("avg", 0) or 0
        ot = data.get("output_token_throughput", {})
        result.output_tok_per_sec = ot.get("avg", 0) or 0
        ttft = data.get("time_to_first_token", data.get("ttft", {}))
        if isinstance(ttft, dict):
            result.ttft_p50_ms = ttft.get("p50", 0) or 0
            result.ttft_p99_ms = ttft.get("p99", 0) or 0
        itl = data.get("inter_token_latency", data.get("itl", {}))
        if isinstance(itl, dict):
            result.itl_p50_ms = itl.get("p50", 0) or 0
            result.itl_p99_ms = itl.get("p99", 0) or 0
        bd = data.get("benchmark_duration", 0)
        result.duration_sec = bd.get("avg", 0) if isinstance(bd, dict) else (bd or 0)
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
