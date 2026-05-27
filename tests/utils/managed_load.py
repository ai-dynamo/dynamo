# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ManagedLoad - YAML template-based load testing using shared PVC.

This module provides a simplified load testing framework that:
1. Uses a YAML template for the Job spec (instead of generating dynamically)
2. Modifies only the aiperf command as needed
3. Uses shared PVC with ManagedDeployment for storing results
4. Uses PvcExtractor for extracting results from shared PVC
"""

import asyncio
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import kr8s
import yaml
from kubernetes_asyncio import client
from kubernetes_asyncio.client import exceptions


def _get_template_dir() -> str:
    """Get the templates directory path."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


@dataclass
class WorkerPin:
    """Pin every request to a specific (service, replica_index) pair.

    The role+index tuple is resolved at StartLoad.execute() time against
    the live ManagedDeployment — the framework reads the matching pod's
    ``DynamoWorkerMetadata`` CR (nvidia.com/v1alpha1) and injects the
    discovered ``instance_id`` into the request's ``nvext.worker_id``
    block. Either or both phases may be pinned; an unpinned phase is
    chosen by the FE's normal routing.

    Requires ``DYN_DISCOVERY_BACKEND=kubernetes`` on the workers so the
    DynamoWorkerMetadata CRs exist; with the etcd backend they don't.
    """

    decode_service: Optional[str] = None
    decode_replica_index: Optional[int] = None
    prefill_service: Optional[str] = None
    prefill_replica_index: Optional[int] = None


@dataclass
class LoadConfig:
    """Configuration for load test parameters.

    Note: endpoint_path is the API suffix (e.g., "/v1/chat/completions").
    The base URL (host:port) is passed separately to ManagedLoad.
    """

    model_name: str = "Qwen/Qwen3-0.6B"
    tokenizer: Optional[str] = None
    endpoint_path: str = "/v1/chat/completions"  # API endpoint suffix

    # Load parameters
    concurrency: int = 8
    request_count: Optional[int] = None
    duration_minutes: Optional[float] = None

    # Token parameters
    input_tokens_mean: int = 512
    input_tokens_stddev: int = 0
    output_tokens_mean: int = 64
    output_tokens_stddev: int = 0

    # AIPerf bucketed (ISL,OSL):weight distribution. When set, REPLACES
    # the mean/stddev pair above. Format is the literal `--seq-dist`
    # string aiperf expects, e.g.
    # "100,200:5;500,200:15;1000,200:20;1600,200:30;3400,200:20;7000,200:10"
    # (six buckets — each "isl,osl:probability"). Lets a test express a
    # lognormal-shape distribution (p50 1.6k / p99 7k) that prod traffic
    # follows. None = use mean/stddev path.
    seq_dist: Optional[str] = None

    # Prefix-cache shaping. AIPerf builds a pool of `num_prefix_prompts`
    # templates, each truncated to `prefix_prompt_length` tokens, and
    # rotates them on the front of every request. Block-level hit-rate
    # ≈ prefix_prompt_length / mean_isl (clamped to vllm block size 16).
    # Example: ~37% with num_prefix_prompts=15 / prefix_prompt_length=600
    # against ISL p50 ≈ 1600.
    num_prefix_prompts: Optional[int] = None
    prefix_prompt_length: Optional[int] = None
    # Alternative: single shared system prompt of L tokens prepended to
    # every request. Mutually exclusive with the prefix-prompts pool;
    # set whichever fits the production shape.
    shared_system_prompt_length: Optional[int] = None

    # Request parameters
    streaming: bool = True
    request_rate: Optional[float] = None
    request_timeout_seconds: float = 30.0
    warmup_requests: Optional[int] = None
    # Probabilistic mid-flight cancellation. AIPerf sends the request
    # fully, then aborts after `request_cancellation_delay` seconds.
    # `request_cancellation_rate` is the percentage of requests cancelled
    # (0-100). Designed for testing how the server handles client
    # disconnects — exercises the SO_LINGER=0 RST path on the frontend
    # without needing iptables fault injection. Independent of the SLA
    # timeout (`request_timeout_seconds`) which also cancels but only on
    # the slow tail.
    request_cancellation_rate: Optional[float] = None
    request_cancellation_delay: float = 0.0
    # AIPerf arrival shaping. "constant" (default), "poisson", "burst".
    # `arrival_smoothness` is the burstiness coefficient when arrival_pattern=burst
    # (lower = more bursty; ~0.2 produces sharp peaks).
    arrival_pattern: Optional[str] = None
    arrival_smoothness: Optional[float] = None
    # Closed-loop concurrency ramp: linearly grow from
    # `warmup_concurrency` (or 1) up to `concurrency` over
    # `concurrency_ramp_duration` seconds. Mirrors the diurnal climb
    # observed on production disagg decode pods (typical diurnal climb,
    # e.g. 7 → 16 RPS over ~2 h) in miniature.
    concurrency_ramp_duration: Optional[float] = None
    warmup_concurrency: Optional[int] = None
    warmup_duration: Optional[float] = None
    # aiperf goodput SLOs. List of "metric_tag:value" strings in the
    # metric's display unit; aiperf treats a request as "good" only if
    # all SLOs hold. Common tags + units:
    #   request_latency:<ms>                    (e2e)
    #   time_to_first_token:<ms>                (TTFT)
    #   inter_token_latency:<ms>                (ITL)
    #   output_token_throughput_per_user:<tok/s>
    # See https://arxiv.org/pdf/2401.09670 (DistServe) for the definition.
    # When set, aiperf reports a ``goodput`` summary alongside
    # request_throughput. None = goodput not computed.
    goodput: Optional[list[str]] = None
    # aiperf transport connection reuse. 'pooled' (aiperf default) keeps a
    # single long-lived socket per worker; 'never' opens a fresh socket
    # per request and tears it down after. Fault-injection tests should
    # use 'never' so a NetworkPolicy applied mid-load can actually deny
    # new flows — connection-tracked pooled sockets survive policy
    # creation and the partition becomes a no-op.
    connection_reuse_strategy: Optional[str] = None
    # Additional /metrics endpoints for aiperf's --server-metrics. The
    # base URL (Frontend) is scraped automatically; URLs here are
    # appended so the run also collects per-worker Prometheus
    # snapshots (e.g. the dynamo system_port pass-through, which
    # surfaces vllm:* counters like vllm:nixl_num_kv_expired_reqs).
    # StartLoad auto-populates this from worker pod IPs when None is
    # passed; explicit list overrides. Note that pod IPs are valid
    # for the duration of a pod's lifetime — tests that destroy and
    # recreate a worker (DeletePod, etc.) will see scraping stop on
    # recovery; the framework's per-pod _get_pod_metrics snapshot at
    # fault-injection time still captures that point.
    extra_server_metrics_urls: Optional[list[str]] = None

    # Inference parameters
    temperature: float = 0.0
    repetition_penalty: float = 1.0
    ignore_eos: bool = True

    # Extra inference parameters
    extra_inputs: Optional[Dict[str, Any]] = None

    # Pin every request to a specific (service, replica_index) pair via
    # nvext.worker_id. Resolved at StartLoad.execute() time by reading
    # the matching pod's DynamoWorkerMetadata CR. None = no pinning.
    worker_pin: Optional[WorkerPin] = None

    def __post_init__(self):
        if self.warmup_requests is None:
            self.warmup_requests = min(self.concurrency, 10)
        if self.tokenizer is None:
            self.tokenizer = self.model_name


@dataclass
class ManagedLoad:
    """YAML template-based load testing using shared PVC.

    Args:
        namespace: Kubernetes namespace for the load test job
        load_config: Load test configuration (concurrency, tokens, etc.)
        pvc_name: Shared PVC from ManagedDeployment for storing results
        endpoint_url: Base URL of the frontend service (e.g., http://frontend:8000)
    """

    namespace: str
    load_config: LoadConfig
    pvc_name: str  # Shared PVC from ManagedDeployment
    endpoint_url: str  # Base URL (host:port) - passed by caller
    # Per-test sub-path prefix inside the shared PVC. Non-empty only in
    # reuse-PVC mode (--log-pvc). When set, the load Job mounts the PVC
    # at `<pvc_run_id>/aiperf` so each test on a shared PVC writes to
    # its own location, and extract+clear operates on that sub-path
    # only.
    pvc_run_id: str = ""
    template_path: Optional[str] = None
    log_dir: Optional[str] = None
    job_name: Optional[str] = None
    container_results_dir: str = "/tmp/aiperf"

    # Internal state
    _core_api: Optional[client.CoreV1Api] = None
    _batch_api: Optional[client.BatchV1Api] = None
    _logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    _job_created: bool = False
    _terminated: bool = False
    _load_completed: bool = False
    _unique_suffix: str = field(default_factory=lambda: secrets.token_hex(4))

    def __post_init__(self):
        # Generate unique job name if not provided
        if self.job_name is None:
            self.job_name = f"load-test-{self._unique_suffix}"

        # Use default template path if not provided
        if self.template_path is None:
            self.template_path = os.path.join(_get_template_dir(), "load_job.yaml")

        # Set up local output directory. Each ManagedLoad gets its own
        # `load/<job_name>/` so chained StartLoad / rung results don't
        # overlay each other locally. (The previous flat `load/` mirrored
        # only the last rung's files on disk.)
        if self.log_dir:
            self.local_output_dir = os.path.join(
                self.log_dir, "load", self.job_name or "default"
            )
            os.makedirs(self.local_output_dir, exist_ok=True)
            self._logger.info(
                f"Load test results will be saved to: {self.local_output_dir}"
            )
        else:
            self.local_output_dir = None

    async def _init_kubernetes(self):
        """Initialize kubernetes clients."""
        from tests.utils.k8s_helpers import init_kubernetes_clients

        self._core_api, self._batch_api, _, _, _ = await init_kubernetes_clients()

    def _load_template(self) -> dict:
        """Load and parse YAML template."""
        with open(self.template_path, "r") as f:
            return yaml.safe_load(f)

    def _build_aiperf_command(self) -> str:
        """Build aiperf command string from LoadConfig."""
        cfg = self.load_config

        args = [
            "aiperf",
            "profile",
            "--artifact-dir",
            self.container_results_dir,
            "--model",
            cfg.model_name,
            "--tokenizer",
            cfg.tokenizer,
            "--endpoint-type",
            "chat",
            "--endpoint",
            cfg.endpoint_path,
            "--url",
            self.endpoint_url,
        ]

        # ISL/OSL shape: bucketed --seq-dist takes precedence over
        # mean/stddev when set. Both paths flow through aiperf's
        # synthetic dataset generator.
        if cfg.seq_dist:
            args.extend(["--seq-dist", cfg.seq_dist])
        else:
            args.extend(
                [
                    "--synthetic-input-tokens-mean",
                    str(cfg.input_tokens_mean),
                    "--synthetic-input-tokens-stddev",
                    str(cfg.input_tokens_stddev),
                    "--output-tokens-mean",
                    str(cfg.output_tokens_mean),
                    "--output-tokens-stddev",
                    str(cfg.output_tokens_stddev),
                ]
            )

        # Prefix-cache shaping (mutually exclusive paths). When neither
        # is set, aiperf generates random prompts -> ~0% block-level hit.
        if cfg.num_prefix_prompts and cfg.prefix_prompt_length:
            args.extend(
                [
                    "--num-prefix-prompts",
                    str(cfg.num_prefix_prompts),
                    "--prefix-prompt-length",
                    str(cfg.prefix_prompt_length),
                ]
            )
        elif cfg.shared_system_prompt_length:
            args.extend(
                [
                    "--shared-system-prompt-length",
                    str(cfg.shared_system_prompt_length),
                ]
            )

        args.extend(
            [
                "--concurrency",
                str(cfg.concurrency),
                "--request-timeout-seconds",
                str(cfg.request_timeout_seconds),
                "--num-dataset-entries",
                "12800",
                "--random-seed",
                "100",
                "--workers-max",
                "252",
                "--record-processors",
                "32",
                "--ui",
                "simple",
                "--verbose",
            ]
        )

        # Closed-loop concurrency ramp + warmup variants.
        if cfg.concurrency_ramp_duration:
            args.extend(
                ["--concurrency-ramp-duration", str(cfg.concurrency_ramp_duration)]
            )
        if cfg.warmup_concurrency:
            args.extend(["--warmup-concurrency", str(cfg.warmup_concurrency)])
        if cfg.warmup_duration:
            args.extend(["--warmup-duration", str(cfg.warmup_duration)])

        # Arrival pattern (rate-based mode; ignored when concurrency mode
        # is in use, which is our default).
        if cfg.arrival_pattern:
            args.extend(["--arrival-pattern", cfg.arrival_pattern])
        if cfg.arrival_smoothness is not None:
            args.extend(["--arrival-smoothness", str(cfg.arrival_smoothness)])

        # Probabilistic mid-flight cancellation (separate from SLA
        # timeout — both can fire on the same run).
        if cfg.request_cancellation_rate:
            args.extend(
                ["--request-cancellation-rate", str(cfg.request_cancellation_rate)]
            )
            if cfg.request_cancellation_delay:
                args.extend(
                    [
                        "--request-cancellation-delay",
                        str(cfg.request_cancellation_delay),
                    ]
                )

        # Add duration or request count
        if cfg.duration_minutes:
            args.extend(["--benchmark-duration", str(cfg.duration_minutes * 60)])
        elif cfg.request_count:
            args.extend(["--request-count", str(cfg.request_count)])

        # aiperf 0.7.0 rejects --warmup-request-count 0 (requires > 0).
        # Only emit the flag when the caller wants warmup; otherwise let
        # aiperf use its default (no warmup or its own default count).
        if cfg.warmup_requests:
            args.extend(["--warmup-request-count", str(cfg.warmup_requests)])

        if cfg.streaming:
            args.append("--streaming")

        if cfg.request_rate:
            args.extend(["--request-rate", str(cfg.request_rate)])

        if cfg.connection_reuse_strategy:
            args.extend(["--connection-reuse-strategy", cfg.connection_reuse_strategy])

        if cfg.goodput:
            # aiperf accepts a single space-separated 'KEY:VALUE' string
            # (parse_str_as_numeric_dict in input_config.py). Repeated
            # --goodput flags overwrite each other; single-string is
            # required. Shell-quoting handled by shlex.join on the way out.
            args.extend(["--goodput", " ".join(cfg.goodput)])

        if cfg.extra_server_metrics_urls:
            # aiperf accepts: --server-metrics <url1> <url2> ... — the
            # base URL is scraped automatically by default, the URLs
            # passed here are *additional* endpoints.
            # Note: requires aiperf >= 0.7.0 for the flag to take effect;
            # earlier versions silently ignored it.
            args.extend(["--server-metrics", *cfg.extra_server_metrics_urls])
            # Default formats are JSON + CSV — aggregated summaries only.
            # Add JSONL (raw 1Hz time-series, line-delimited) so we can
            # post-process which signal spiked first vs. last. Skip
            # PARQUET: redundant for our minute-scale runs and pulls in
            # pyarrow as a downstream-tool dep with no real benefit at
            # this data volume.
            args.extend(
                [
                    "--server-metrics-formats",
                    "json",
                    "csv",
                    "jsonl",
                ]
            )

        # Build extra inputs.
        #
        # max_tokens / min_tokens semantics in vllm:
        #   - We pass our requested max_tokens to the engine.
        #   - vllm INTERNALLY clamps max_tokens down to
        #     `max_model_len − prompt_token_count` per-request. If a high
        #     ISL sample (e.g. 8112 tokens with max_model_len=8192) leaves
        #     only 80 tokens of budget, vllm uses max_tokens=80 for THAT
        #     request — even though the client asked for 100.
        #   - If the client's `min_tokens` exceeds vllm's clamped
        #     max_tokens, vllm rejects the request with
        #       `ValueError: min_tokens=N must be ≤ max_tokens=K, got K<N`.
        #
        # Default (here): min_tokens=1. vllm's per-request clamp is
        # always allowed to drop the cap; the floor is permissive; no
        # rejection. Tests get the load they ask for.
        #
        # Opt-in failure mode (e.g. test_scenario_bad_input_distribution
        # passes `extra_inputs={"min_tokens": output_tokens_mean}`):
        # rigid min=max forces vllm to reject any request whose ISL
        # leaves <min_tokens of budget. Useful for modeling the "client
        # sends out-of-spec request" failure that climbs error counters.
        extra_inputs: Dict[str, Any] = {
            "min_tokens": 1,
            "temperature": cfg.temperature,
            "repetition_penalty": cfg.repetition_penalty,
        }
        # When seq_dist is in use, AIPerf drives per-request max_tokens
        # from the bucketed OSL — adding a static max_tokens here would
        # clamp every request to the same OSL, defeating the
        # distribution. Only set max_tokens when seq_dist is None.
        if not cfg.seq_dist:
            extra_inputs["max_tokens"] = cfg.output_tokens_mean

        if cfg.ignore_eos:
            extra_inputs["ignore_eos"] = True

        if cfg.extra_inputs:
            extra_inputs.update(cfg.extra_inputs)

        for key, value in extra_inputs.items():
            # Two AIPerf wire formats for --extra-inputs:
            #   - flat scalar: `key:value` (e.g. `max_tokens:200`)
            #   - JSON object: `{"key": <value>}` (e.g.
            #     `{"nvext":{"prefill_worker_id":123,"decode_worker_id":456}}`)
            # The JSON form is required to express nested objects like
            # `nvext.*` — see components/src/dynamo/profiler/utils/aiperf.py:65
            # for the canonical example. Detect dicts and route them to
            # the JSON path so `extra_inputs={"nvext": {...}}` works as
            # the caller expects.
            if isinstance(value, dict):
                args.extend(["--extra-inputs", json.dumps({key: value})])
            else:
                args.extend(["--extra-inputs", f"{key}:{value}"])

        # shlex.join quotes args that contain whitespace (e.g. --goodput's
        # 'request_latency:30000 ttft:5000' string) so they survive the
        # docker exec -> bash chain as a single token to aiperf.
        import shlex as _shlex

        return _shlex.join(args)

    def _apply_config_to_template(self, template: dict) -> dict:
        """Apply LoadConfig to template - set command, PVC, namespace."""
        # Set metadata
        template["metadata"]["name"] = self.job_name
        template["metadata"]["namespace"] = self.namespace

        # Get pod spec
        pod_spec = template["spec"]["template"]["spec"]
        container = pod_spec["containers"][0]

        # Build the aiperf command
        aiperf_cmd = self._build_aiperf_command()

        # Update the container args with the aiperf command
        # The template has AIPERF_CMD placeholder
        original_script = container["args"][0]
        updated_script = original_script.replace("AIPERF_CMD", aiperf_cmd)
        container["args"] = [updated_script]

        # Update environment variables
        for env in container.get("env", []):
            if env["name"] == "ENDPOINT_URL":
                env["value"] = self.endpoint_url
            elif env["name"] == "MODEL_NAME":
                env["value"] = self.load_config.model_name

        # Set PVC reference
        for volume in pod_spec.get("volumes", []):
            if "persistentVolumeClaim" in volume:
                volume["persistentVolumeClaim"]["claimName"] = self.pvc_name

        # Update volume mount path if needed. We prepend `<job_name>` to
        # the subPath so each chained StartLoad / rung writes to its
        # own location on the PVC (aiperf overwrites profile_export*
        # files on each invocation; sharing one sub-path means rung N+1
        # destroys rung N's results before extraction). In reuse-PVC
        # mode we also prepend `<run_id>` so multiple test scenarios
        # sharing the PVC stay isolated from each other.
        rung_subpath = f"{self.job_name}/aiperf"
        if self.pvc_run_id:
            rung_subpath = f"{self.pvc_run_id}/{rung_subpath}"
        for mount in container.get("volumeMounts", []):
            if mount["name"] == "results-volume":
                mount["mountPath"] = self.container_results_dir
                mount["subPath"] = rung_subpath

        return template

    async def _create_job(self):
        """Create the load test job in Kubernetes."""
        template = self._load_template()
        job_spec = self._apply_config_to_template(template)

        self._logger.info(f"Creating load test job: {self.job_name}")

        try:
            assert self._batch_api is not None, "Kubernetes API not initialized"
            await self._batch_api.create_namespaced_job(
                namespace=self.namespace, body=job_spec
            )
            self._job_created = True
            self._logger.info(f"Load test job created: {self.job_name}")
        except exceptions.ApiException as e:
            if e.status == 409:  # Already exists
                self._logger.warning(f"Job {self.job_name} already exists")
                self._job_created = True
            else:
                self._logger.error(f"Failed to create job {self.job_name}: {e}")
                raise

    async def run(self, wait_for_completion: bool = True) -> Dict[str, Any]:
        """Start the load test job."""
        await self._create_job()

        if wait_for_completion:
            success = await self.wait_for_completion()
            return {"success": success, "job_name": self.job_name}
        else:
            self._logger.info(f"Load test job {self.job_name} started (not waiting)")
            return {"success": True, "job_name": self.job_name}

    async def _wait_for_status_marker(
        self, marker_file: str, marker_description: str, timeout: int
    ) -> bool:
        """Wait for a specific status marker file to appear in the pod."""
        start_time = time.time()

        self._logger.info(
            f"Waiting for {marker_description} in job {self.job_name} (timeout: {timeout}s)"
        )

        while (time.time() - start_time) < timeout:
            if self._terminated:
                self._logger.info(f"{marker_description} wait terminated by request")
                return False

            try:
                # Find the pod for this job
                pods = []
                pod_generator = kr8s.get(
                    "pods",
                    namespace=self.namespace,
                    label_selector=f"job-name={self.job_name}",
                )
                for pod in pod_generator:
                    pods.append(pod)

                if pods:
                    pod = pods[0]

                    # Check if the status marker exists
                    try:
                        result = await asyncio.wait_for(
                            asyncio.create_task(
                                asyncio.to_thread(pod.exec, ["test", "-f", marker_file])
                            ),
                            timeout=10.0,
                        )

                        if result.returncode == 0:
                            self._logger.info(
                                f"{marker_description} marker found in job {self.job_name}"
                            )
                            return True

                    except (asyncio.TimeoutError, Exception):
                        pass

                    # Check for job failure
                    assert self._batch_api is not None
                    job = await self._batch_api.read_namespaced_job(
                        name=self.job_name, namespace=self.namespace
                    )

                    if job.status.failed:
                        self._logger.error(
                            f"Load test job {self.job_name} failed while waiting for {marker_description}"
                        )
                        return False

                    # Check if job completed (not running anymore)
                    if job.status.completion_time is not None:
                        self._logger.info(
                            f"Job {self.job_name} completed while waiting for {marker_description}"
                        )
                        return True

            except exceptions.ApiException as e:
                self._logger.warning(f"Error checking {marker_description} status: {e}")

            await asyncio.sleep(5)

        if self._terminated:
            return False
        else:
            raise TimeoutError(f"{marker_description} did not appear within {timeout}s")

    async def wait_for_started(self, timeout: int = 300) -> bool:
        """Wait for the load test to start (5 minute timeout by default)."""
        marker_file = f"{self.container_results_dir}/status/{self.job_name}/started"
        return await self._wait_for_status_marker(
            marker_file, "Load test start", timeout
        )

    async def wait_for_completion(self, timeout: Optional[int] = None) -> bool:
        """Wait for the load test to complete."""
        if timeout is None:
            if self.load_config.duration_minutes:
                timeout = (
                    int(self.load_config.duration_minutes * 60) + 300
                )  # 5 min buffer
            elif self.load_config.request_count:
                timeout = max(self.load_config.request_count * 2 + 60, 300)
            else:
                timeout = 600  # Default 10 minutes

        marker_file = f"{self.container_results_dir}/status/{self.job_name}/completed"
        result = await self._wait_for_status_marker(
            marker_file, "Load test completion", timeout
        )
        if result:
            self._load_completed = True
        return result

    async def terminate(self):
        """Gracefully terminate the running load test.

        Sends SIGINT to aiperf, waits for graceful shutdown, then SIGTERM.
        Container exits after aiperf completes - results are on PVC.

        If load already completed naturally, skips signaling.
        """
        self._logger.info(f"Terminating load test in job {self.job_name}")
        self._terminated = True

        if not self._job_created:
            self._logger.warning("No job created to terminate")
            return

        # Skip signaling if load already completed naturally
        if self._load_completed:
            self._logger.info(
                "Load test already completed, skipping termination signals"
            )
            return

        try:
            # Find and signal the pod
            pods = []
            pod_generator = kr8s.get(
                "pods",
                namespace=self.namespace,
                label_selector=f"job-name={self.job_name}",
            )
            for pod in pod_generator:
                pods.append(pod)

            if pods:
                pod = pods[0]
                self._logger.info(f"Sending termination signal to pod {pod.name}")

                # Send SIGINT for graceful shutdown
                try:
                    result = await asyncio.wait_for(
                        asyncio.create_task(
                            asyncio.to_thread(pod.exec, ["pkill", "-SIGINT", "aiperf"])
                        ),
                        timeout=10.0,
                    )
                    if result.returncode == 0:
                        self._logger.info("SIGINT sent to aiperf process")
                    else:
                        self._logger.warning(
                            "SIGINT failed, process may already be stopped"
                        )
                except Exception as e:
                    self._logger.warning(f"Failed to send SIGINT: {e}")

                # Wait for graceful shutdown
                await asyncio.sleep(5)

                # Send SIGTERM if still running
                try:
                    result = await asyncio.wait_for(
                        asyncio.create_task(
                            asyncio.to_thread(pod.exec, ["pkill", "-SIGTERM", "aiperf"])
                        ),
                        timeout=10.0,
                    )
                    if result.returncode == 0:
                        self._logger.info("SIGTERM sent to aiperf process")
                except Exception as e:
                    self._logger.warning(f"Failed to send SIGTERM: {e}")

                # Wait for pod to complete (container exits after aiperf finishes)
                self._logger.info("Waiting for pod to complete...")
                for attempt in range(60):  # Wait up to 1 minute
                    try:
                        pod.refresh()
                        phase = pod.status.phase
                        if phase in ("Succeeded", "Failed"):
                            self._logger.info(f"Pod completed with phase: {phase}")
                            break
                    except Exception:
                        pass

                    if attempt < 59:
                        await asyncio.sleep(1)
                else:
                    self._logger.warning(
                        "Pod did not complete in time, proceeding anyway"
                    )

                self._logger.info("Load test termination completed")
            else:
                self._logger.warning("No pods found to terminate")

        except Exception as e:
            self._logger.error(f"Error during termination: {e}")

    async def is_running(self) -> bool:
        """Check if the load test is currently running."""
        if not self._job_created or self._terminated:
            return False

        try:
            assert self._batch_api is not None
            job = await self._batch_api.read_namespaced_job(
                name=self.job_name, namespace=self.namespace
            )

            # Job is running if not completed and not failed
            is_active = job.status.completion_time is None and (
                job.status.failed is None or job.status.failed == 0
            )

            if is_active:
                # Double check by looking for running pods
                pods = []
                pod_generator = kr8s.get(
                    "pods",
                    namespace=self.namespace,
                    label_selector=f"job-name={self.job_name}",
                )
                for pod in pod_generator:
                    if pod.status.phase in ["Running", "Pending"]:
                        pods.append(pod)
                return len(pods) > 0

            return False

        except exceptions.ApiException as e:
            if e.status == 404:
                return False
            self._logger.warning(f"Error checking job status: {e}")
            return False

    async def _delete_pod(self) -> None:
        """Delete the load test pod."""
        try:
            pods = list(
                kr8s.get(
                    "pods",
                    namespace=self.namespace,
                    label_selector=f"job-name={self.job_name}",
                )
            )

            if not pods:
                return

            pod = pods[0]
            self._logger.info(f"Deleting pod {pod.name}")
            pod.delete(force=True)

            # Wait for pod to be deleted
            for _ in range(30):
                try:
                    remaining = list(
                        kr8s.get(
                            "pods",
                            namespace=self.namespace,
                            label_selector=f"job-name={self.job_name}",
                        )
                    )
                    if not remaining:
                        self._logger.info("Pod deleted")
                        break
                except Exception:
                    break
                await asyncio.sleep(1)

        except Exception as e:
            self._logger.warning(f"Failed to delete pod: {e}")

    async def _debug_pod_files(self) -> None:
        """Debug helper: List files in the pod's results directory."""
        try:
            pods = list(
                kr8s.get(
                    "pods",
                    namespace=self.namespace,
                    label_selector=f"job-name={self.job_name}",
                )
            )

            if not pods:
                self._logger.info("DEBUG: No pods found")
                return

            pod = pods[0]
            self._logger.info(f"DEBUG: Pod {pod.name} phase: {pod.status.phase}")

            if pod.status.phase != "Running":
                return

            # List files in results directory
            result = await asyncio.wait_for(
                asyncio.create_task(
                    asyncio.to_thread(
                        pod.exec,
                        ["ls", "-la", self.container_results_dir],
                    )
                ),
                timeout=10.0,
            )
            if result.returncode == 0:
                self._logger.info(
                    f"DEBUG: Files in {self.container_results_dir}:\n{result.stdout.decode()}"
                )

        except Exception as e:
            self._logger.warning(f"DEBUG: Failed to list files: {e}")

    async def get_results(self) -> Optional[Dict[str, Any]]:
        """Get parsed results JSON from PVC via PvcExtractor."""
        try:
            # Debug: Show what files are in the pod before deletion
            await self._debug_pod_files()

            # Delete the pod (results are already in PVC)
            await self._delete_pod()

            # Extract from PVC using PvcExtractor
            from tests.utils.pvc_extractor import PvcExtractor

            extractor = PvcExtractor(namespace=self.namespace, logger=self._logger)
            await extractor.init()

            output_dir = self.local_output_dir or "load"
            # Mirror the per-rung subPath layout used in `_apply_config_to_template`
            # so we extract exactly this rung's files (not whatever the last
            # ManagedLoad left under `aiperf/`).
            sub_path = f"{self.job_name}/aiperf"
            if self.pvc_run_id:
                sub_path = f"{self.pvc_run_id}/{sub_path}"
            result = await extractor.extract(
                pvc_name=self.pvc_name,
                sub_path=sub_path,
                container_path=self.container_results_dir,
                # Explicit result-file globs. Avoid `*.json` because
                # aiperf's `inputs.json` (synthetic prompt corpus) can be
                # 100s of MB and saturates the WebSocket `cat` we use to
                # stream the tar out of the pod. The inputs.json isn't
                # needed for analysis.
                file_patterns=[
                    "profile_export*.json",
                    "profile_export*.jsonl",
                    "profile_export*.csv",
                    "server_metrics_export*.json",
                    "server_metrics_export*.jsonl",
                    "server_metrics_export*.csv",
                    "aiperf.log",
                ],
                local_output_dir=output_dir,
                # clear_after_extract defaults to True
            )

            if result.get("success"):
                output_path = Path(result["output_dir"])
                available_files = (
                    list(output_path.iterdir()) if output_path.exists() else []
                )
                self._logger.info(
                    f"Available result files: {[f.name for f in available_files]}"
                )

                json_file = output_path / "profile_export_aiperf.json"
                if json_file.exists():
                    self._logger.info(f"Using aiperf summary: {json_file}")
                    with open(json_file) as f:
                        return json.load(f)

                self._logger.warning(
                    f"No profile_export_aiperf.json found in {output_path}. "
                    f"Available files: {[f.name for f in available_files]}"
                )
            else:
                # Make the failure obvious in the test log — previously this
                # branch silently dropped a successful run's data when the
                # extractor timed out on FSx, leaving no per-rung results
                # while the load itself had succeeded.
                self._logger.error(
                    "Result extraction failed for load '%s' (pvc=%s, sub_path=aiperf): %s. "
                    "Load Job completed but aiperf JSON was not copied off the PVC; "
                    "raise pvc_extractor timeouts or run with --log-pvc to keep the data on the PVC.",
                    self.job_name,
                    self.pvc_name,
                    result.get("error", "<no error key>"),
                )

            return None

        except Exception:
            self._logger.exception("Failed to get results")
            return None

    async def _cleanup(self):
        """Clean up the load test job."""
        if self._job_created and self._batch_api:
            try:
                from kubernetes_asyncio.client.models import V1DeleteOptions

                delete_options = V1DeleteOptions(propagation_policy="Foreground")
                await self._batch_api.delete_namespaced_job(
                    name=self.job_name,
                    namespace=self.namespace,
                    body=delete_options,
                )
                self._logger.info(f"Load test job {self.job_name} deleted")
            except exceptions.ApiException as e:
                if e.status != 404:
                    self._logger.warning(f"Failed to delete job {self.job_name}: {e}")

    async def __aenter__(self) -> "ManagedLoad":
        """Create the load job."""
        await self._init_kubernetes()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup job resources."""
        # Log if we're exiting due to an exception (Ctrl-C, etc.)
        if exc_type is not None:
            self._logger.warning(
                f"Exiting due to exception ({exc_type.__name__}), running cleanup"
            )
        else:
            # Only try to extract results if we're not exiting due to an exception
            if self._job_created and self.local_output_dir:
                try:
                    await self.get_results()
                except Exception:
                    # Log full traceback so silent timeouts / k8s API errors
                    # are debuggable; previously this swallowed the cause and
                    # printed only the exception's str (empty for TimeoutError).
                    self._logger.exception(
                        "Failed to extract results during cleanup " "(job=%s, pvc=%s)",
                        self.job_name,
                        self.pvc_name,
                    )

        # Always run cleanup, catching any cleanup errors
        try:
            await self._cleanup()
        except Exception as cleanup_error:
            self._logger.error(f"Error during cleanup: {cleanup_error}")
