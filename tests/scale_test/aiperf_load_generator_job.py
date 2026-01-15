# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import shlex
import time
import yaml
from dataclasses import dataclass, field
from typing import List, Optional

from kubernetes_asyncio import client
from kubernetes_asyncio.client import exceptions

logger = logging.getLogger(__name__)


# Default to python:3.12-slim - aiperf will be installed at runtime
DEFAULT_AIPERF_IMAGE: str = "python:3.12-slim"

# AIPerf git ref to install at runtime when using base python image
AIPERF_GIT_REF: str = "4d3fa29403c8f75da22a14f1f7b3aeb27db9288f"


@dataclass
class AIPerfConfig:
    """Configuration for AIPerf load generation."""

    # Input sequence length
    isl_mean: int = 512
    isl_stddev: int = 0

    # Output sequence length
    osl_mean: int = 128
    osl_stddev: int = 0

    # Concurrency (number of parallel requests)
    concurrency: int = 10

    # Request count (0 = run for duration)
    request_count: int = 0

    # Warmup requests per concurrency level
    warmup_request_count: int = 2

    # Prefix caching simulation
    prefix_prompt_length: int = 0
    num_prefix_prompts: int = 0

    # Random seed for reproducibility
    random_seed: int = 100

    # Streaming mode
    streaming: bool = True

    # Extra inputs to pass to the model
    ignore_eos: bool = True

    # Worker configuration for high concurrency
    # Constraint: workers_max * http_connection_limit < num_ephemeral_ports
    # With port range 1024-65000, we have ~63976 ports
    # Default: 252 * 252 = 63504 < 63976
    workers_max: int = 252
    http_connection_limit: int = 252


class AIPerfLoadGeneratorJob:
    """Manages a Kubernetes Job that runs AIPerf load generation inside the cluster.

    Note: Requires a container image with AIPerf installed (e.g., the Dynamo container).
    """

    def __init__(
        self,
        namespace: str,
        frontend_url: str,
        model: str,
        duration_sec: int,
        config: Optional[AIPerfConfig] = None,
        job_name: str = "scale-test-aiperf",
        image: Optional[str] = None,
        tokenizer: Optional[str] = None,
    ):
        self.namespace = namespace
        self.frontend_url = frontend_url
        self.model = model
        self.duration_sec = duration_sec
        self.config = config or AIPerfConfig()
        self.job_name = job_name
        self.image = image or DEFAULT_AIPERF_IMAGE
        self.tokenizer = tokenizer or model

        self._batch_api: Optional[client.BatchV1Api] = None
        self._core_api: Optional[client.CoreV1Api] = None

    def _compute_resources(self) -> dict:
        """Compute container resources based on concurrency and request count.

        Memory requirements come from two sources:
        1. Connection state: ~100KB per concurrent connection
        2. Profiling data: varies by ISL/OSL - tokens are stored as strings/lists

        With Python overhead, multiply estimates by 3-4x for safety.
        """
        concurrency = self.config.concurrency
        request_count = self.config.request_count
        isl = self.config.isl_mean
        osl = self.config.osl_mean

        # Base memory for connections (MB)
        connection_mem_mb = (concurrency * 100) // 1024  # 100KB per connection

        # Memory for profiling data (MB)
        # Each request stores: input tokens, output tokens, timing data, metadata
        # Estimate ~4 bytes per token (token IDs) + overhead
        # Plus Python object overhead (~100 bytes per object)
        bytes_per_request = (isl + osl) * 4 + 500  # tokens + metadata
        profiling_mem_mb = (request_count * bytes_per_request) // (1024 * 1024)

        # Total memory with safety margin (4x for Python overhead + GC pressure)
        total_mem_mb = connection_mem_mb + profiling_mem_mb
        mem_gb = max(16, (total_mem_mb // 1024) * 4 + 8)  # 4x safety margin + 8GB base

        # CPU based on concurrency
        if concurrency <= 100:
            cpu_req, cpu_lim = "2", "4"
        elif concurrency <= 1000:
            cpu_req, cpu_lim = "4", "8"
        elif concurrency <= 10000:
            cpu_req, cpu_lim = "8", "16"
        else:
            cpu_req, cpu_lim = "16", "32"

        mem_req = f"{mem_gb}Gi"
        mem_lim = f"{mem_gb * 2}Gi"

        logger.info(
            f"AIPerf resources for concurrency={concurrency}, requests={request_count}: "
            f"CPU {cpu_req}/{cpu_lim}, memory {mem_req}/{mem_lim}"
        )

        return {
            "requests": {"cpu": cpu_req, "memory": mem_req},
            "limits": {"cpu": cpu_lim, "memory": mem_lim},
        }

    def _build_aiperf_command(self) -> List[str]:
        """Build the AIPerf CLI command."""
        cfg = self.config

        # Calculate request count: use explicit value if provided, otherwise estimate from duration.
        # AIPerf does not have a duration-based mode, so we must specify a request count.
        # Heuristic: assume ~1 request/sec per concurrency slot as a baseline, then scale by duration.
        if cfg.request_count > 0:
            request_count = cfg.request_count
        else:
            # Estimate: duration_sec * concurrency / avg_request_time
            # Assume avg request takes ~2-5 seconds for typical ISL/OSL. Use conservative estimate.
            # This gives enough requests to keep the system busy for the full duration.
            requests_per_sec_per_slot = 0.5  # Conservative: each slot completes ~0.5 req/sec
            request_count = max(
                cfg.concurrency * 10,  # Minimum: 10 requests per concurrency slot
                int(self.duration_sec * cfg.concurrency * requests_per_sec_per_slot),
            )
            logger.info(
                f"No explicit request_count provided. "
                f"Calculated {request_count} requests for {self.duration_sec}s duration "
                f"at concurrency {cfg.concurrency}"
            )

        cmd = [
            "aiperf",
            "profile",
            "--ui",
            "simple",
            "--model",
            self.model,
            "--tokenizer",
            self.tokenizer,
            "--endpoint-type",
            "chat",
            "--endpoint",
            "/v1/chat/completions",
            "--url",
            self.frontend_url,
            "--synthetic-input-tokens-mean",
            str(cfg.isl_mean),
            "--synthetic-input-tokens-stddev",
            str(cfg.isl_stddev),
            "--output-tokens-mean",
            str(cfg.osl_mean),
            "--output-tokens-stddev",
            str(cfg.osl_stddev),
            "--concurrency",
            str(cfg.concurrency),
            "--random-seed",
            str(cfg.random_seed),
            "--warmup-request-count",
            str(cfg.warmup_request_count),
            "--artifact-dir",
            "/tmp/aiperf-results",
            "-H",
            "Authorization: Bearer NOT_USED",
            "-H",
            "Accept: text/event-stream",
        ]

        if cfg.streaming:
            cmd.append("--streaming")

        # Always set request count and dataset entries
        cmd.extend(["--request-count", str(request_count)])
        cmd.extend(
            ["--num-dataset-entries", str(request_count + cfg.warmup_request_count)]
        )

        # min/max tokens to force exact output length
        cmd.extend(["--extra-inputs", f"max_tokens:{cfg.osl_mean}"])
        cmd.extend(["--extra-inputs", f"min_tokens:{cfg.osl_mean}"])

        if cfg.ignore_eos:
            cmd.extend(["--extra-inputs", "ignore_eos:true"])
            cmd.extend(["--extra-inputs", '{"nvext":{"ignore_eos":true}}'])

        # Prefix caching simulation
        if cfg.prefix_prompt_length > 0 and cfg.num_prefix_prompts > 0:
            cmd.extend(["--prefix-prompt-length", str(cfg.prefix_prompt_length)])
            cmd.extend(["--num-prefix-prompts", str(cfg.num_prefix_prompts)])

        # Worker configuration for high concurrency
        cmd.extend(["--workers-max", str(cfg.workers_max)])

        return cmd

    def _build_job_manifest(self) -> dict:
        """Build the Kubernetes Job manifest for AIPerf."""
        aiperf_cmd = self._build_aiperf_command()

        # Build shell command that runs aiperf and prints results
        shell_script = f"""
set -e
echo "========================================"
echo "AIPerf Load Generator - Setup"
echo "========================================"

# Install dependencies and aiperf
echo "Installing dependencies..."
apt-get update && apt-get install -y curl jq procps git && apt-get clean
echo "Installing aiperf..."
pip install git+https://github.com/ai-dynamo/aiperf.git@{AIPERF_GIT_REF}
echo "aiperf installation completed"

echo ""
echo "========================================"
echo "AIPerf Load Generator"
echo "========================================"
echo "Target: {self.frontend_url}"
echo "Model: {self.model}"
echo "Duration: {self.duration_sec}s"
echo "Concurrency: {self.config.concurrency}"
echo "ISL: {self.config.isl_mean} (stddev: {self.config.isl_stddev})"
echo "OSL: {self.config.osl_mean} (stddev: {self.config.osl_stddev})"
echo ""
sysctl -w net.ipv4.ip_local_port_range="1024 65000"
cat /proc/sys/net/ipv4/ip_local_port_range

echo "System limits:"
echo "  File descriptors (ulimit -n): $(ulimit -n)"
echo "  Max processes (ulimit -u): $(ulimit -u)"
echo "  Open files (soft): $(ulimit -Sn)"
echo "  Open files (hard): $(ulimit -Hn)"
echo ""
echo "Memory info:"
echo "  Container limit: $(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo 'unknown')"
free -h
echo "========================================"
echo ""
echo "AIPerf command:"
echo "{shlex.join(aiperf_cmd)}"
echo ""

# Start background memory monitor
(
  while true; do
    MEM_USED=$(free -m | awk '/^Mem:/ {{print $3}}')
    MEM_TOTAL=$(free -m | awk '/^Mem:/ {{print $2}}')
    echo "[MEMMON] $(date '+%H:%M:%S') - ${{MEM_USED}}MB / ${{MEM_TOTAL}}MB used"
    sleep 10
  done
) &
MEMMON_PID=$!

echo "========================================"
echo "AIPerf Output"
echo "========================================"

{shlex.join(aiperf_cmd)}
AIPERF_EXIT=$?
if [ $AIPERF_EXIT -ne 0 ]; then
    echo ""
    echo "AIPerf exited with code $AIPERF_EXIT"
fi

echo ""
echo "========================================"
echo "Results Summary"
echo "========================================"

# Find results file - AIPerf outputs to profile_export_aiperf.json
RESULT_FILE=""
if [ -f /tmp/aiperf-results/profile_export_aiperf.json ]; then
    RESULT_FILE="/tmp/aiperf-results/profile_export_aiperf.json"
elif [ -f /tmp/aiperf-results/profile_export.json ]; then
    RESULT_FILE="/tmp/aiperf-results/profile_export.json"
fi

if [ -n "$RESULT_FILE" ]; then
    echo "Results file: $RESULT_FILE"
    echo ""

    # Extract and display key metrics using Python
    # Use quoted heredoc to prevent shell variable expansion, pass file as argument
    python3 - "$RESULT_FILE" << 'PARSE_RESULTS'
import json
import sys

result_file = sys.argv[1]

with open(result_file, 'r') as f:
    data = json.load(f)

def get_stat(d, key, stat='avg'):
    val = d.get(key, {{}})
    if isinstance(val, dict):
        return val.get(stat)
    return None

def fmt(val):
    if val is None:
        return 'N/A'
    return '%.2f' % val

print('=' * 60)
print('KEY METRICS')
print('=' * 60)
print()

# Time to First Token
ttft_avg = get_stat(data, 'time_to_first_token', 'avg')
if ttft_avg is not None:
    print('Time to First Token (TTFT):')
    print('  avg: %s ms' % fmt(ttft_avg))
    print('  p50: %s ms' % fmt(get_stat(data, 'time_to_first_token', 'p50')))
    print('  p99: %s ms' % fmt(get_stat(data, 'time_to_first_token', 'p99')))
    print()

# Inter-token Latency
itl_avg = get_stat(data, 'inter_token_latency', 'avg')
if itl_avg is not None:
    print('Inter-Token Latency (ITL):')
    print('  avg: %s ms' % fmt(itl_avg))
    print('  p50: %s ms' % fmt(get_stat(data, 'inter_token_latency', 'p50')))
    print('  p99: %s ms' % fmt(get_stat(data, 'inter_token_latency', 'p99')))
    print()

# Throughput
req_thpt = get_stat(data, 'request_throughput', 'avg')
tok_thpt = get_stat(data, 'output_token_throughput', 'avg')
if req_thpt is not None or tok_thpt is not None:
    print('Throughput:')
    if req_thpt is not None:
        print('  Requests/sec: %s' % fmt(req_thpt))
    if tok_thpt is not None:
        print('  Output tokens/sec: %s' % fmt(tok_thpt))
    print()

# Request latency
req_lat_avg = get_stat(data, 'request_latency', 'avg')
if req_lat_avg is not None:
    print('Request Latency (end-to-end):')
    print('  avg: %s ms' % fmt(req_lat_avg))
    print('  p50: %s ms' % fmt(get_stat(data, 'request_latency', 'p50')))
    print('  p99: %s ms' % fmt(get_stat(data, 'request_latency', 'p99')))
    print()

# Request counts
if 'request_count' in data:
    print('Requests completed: %s' % data['request_count'])
    print()

print('=' * 60)
print('FULL JSON RESULTS')
print('=' * 60)
print(json.dumps(data, indent=2))
PARSE_RESULTS

else
    echo "No results file found in /tmp/aiperf-results/"
    echo "Contents of artifact directory:"
    ls -la /tmp/aiperf-results/ 2>/dev/null || echo "  (directory does not exist)"
fi

echo ""
echo "========================================"
echo "AIPerf load generation complete."
echo "========================================"
"""

        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": self.job_name,
                "namespace": self.namespace,
                "labels": {"app": "scale-test-aiperf"},
            },
            "spec": {
                "ttlSecondsAfterFinished": 300,
                "backoffLimit": 0,
                "template": {
                    "metadata": {"labels": {"app": "scale-test-aiperf"}},
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [
                            {
                                "name": "aiperf",
                                "image": self.image,
                                "command": ["bash", "-c", shell_script],
                                "env": [
                                    {"name": "PYTHONUNBUFFERED", "value": "1"},
                                    {
                                        "name": "AIPERF_HTTP_CONNECTION_LIMIT",
                                        "value": str(self.config.http_connection_limit),
                                    },
                                ],
                                # "resources": self._compute_resources(),
                                "securityContext": {
                                    "privileged": True,
                                },
                            }
                        ],
                    },
                },
            },
        }

    async def create_and_wait(
        self,
        batch_api: client.BatchV1Api,
        core_api: client.CoreV1Api,
        timeout: int = 600,
    ) -> bool:
        """Create the AIPerf job and wait for completion."""
        self._batch_api = batch_api
        self._core_api = core_api

        logger.info(f"Creating AIPerf job: {self.job_name}")
        job_manifest = self._build_job_manifest()
        with open("job_manifest.yaml", "w") as f:
            yaml.dump(job_manifest, f)

        try:
            await self._batch_api.create_namespaced_job(
                namespace=self.namespace, body=job_manifest
            )
        except exceptions.ApiException as e:
            if e.status == 409:
                logger.warning("Job already exists, recreating...")
                await self._delete_job()
                await asyncio.sleep(2)
                await self._batch_api.create_namespaced_job(
                    namespace=self.namespace, body=job_manifest
                )
            else:
                logger.error(f"Failed to create job: {e}")
                return False

        return await self._wait_for_completion(timeout)

    async def _wait_for_completion(self, timeout: int) -> bool:
        """Wait for the job to complete."""
        start_time = time.time()
        pod_name = None

        logger.info(f"Waiting for AIPerf job to complete (timeout: {timeout}s)...")

        while time.time() - start_time < timeout:
            try:
                job = await self._batch_api.read_namespaced_job_status(
                    name=self.job_name, namespace=self.namespace
                )

                # Get pod name if we don't have it
                if pod_name is None:
                    pods = await self._core_api.list_namespaced_pod(
                        namespace=self.namespace,
                        label_selector=f"job-name={self.job_name}",
                    )
                    if pods.items:
                        pod_name = pods.items[0].metadata.name
                        logger.info(f"AIPerf pod started: {pod_name}")

                if job.status.succeeded:
                    logger.info("AIPerf job completed successfully")
                    if pod_name:
                        await self._print_logs(pod_name)
                    return True
                elif job.status.failed:
                    logger.error(f"AIPerf job failed")
                    if pod_name:
                        await self._print_logs(pod_name)
                    return False

                await asyncio.sleep(5)

            except exceptions.ApiException:
                await asyncio.sleep(2)

        logger.error("Timeout waiting for AIPerf job")
        if pod_name:
            await self._print_logs(pod_name)
        return False

    async def _print_logs(self, pod_name: str) -> None:
        """Print the pod logs and status for debugging."""
        print("\n" + "=" * 70)
        print("AIPERF LOAD GENERATOR LOGS")
        print("=" * 70)

        # Print pod status first for debugging
        try:
            pod = await self._core_api.read_namespaced_pod(
                name=pod_name, namespace=self.namespace
            )
            phase = pod.status.phase
            print(f"Pod status: {phase}")

            # Print container status if available
            if pod.status.container_statuses:
                for cs in pod.status.container_statuses:
                    if cs.state.terminated:
                        t = cs.state.terminated
                        print(
                            f"Container '{cs.name}': terminated "
                            f"(exit_code={t.exit_code}, reason={t.reason})"
                        )
                        if t.message:
                            print(f"  Message: {t.message}")
                    elif cs.state.waiting:
                        print(
                            f"Container '{cs.name}': waiting "
                            f"(reason={cs.state.waiting.reason})"
                        )

            print("-" * 70)
        except exceptions.ApiException as e:
            logger.debug(f"Could not get pod status: {e}")

        # Print logs
        try:
            logs = await self._core_api.read_namespaced_pod_log(
                name=pod_name, namespace=self.namespace
            )
            print(logs)
        except exceptions.ApiException as e:
            logger.error(f"Failed to get logs: {e}")

        print("=" * 70 + "\n")

    async def delete(self) -> None:
        """Delete the job."""
        await self._delete_job()

    async def _delete_job(self) -> None:
        """Delete the Kubernetes job."""
        if self._batch_api is None:
            return
        try:
            await self._batch_api.delete_namespaced_job(
                name=self.job_name,
                namespace=self.namespace,
                propagation_policy="Background",
            )
        except exceptions.ApiException as e:
            if e.status != 404:
                logger.error(f"Failed to delete job: {e}")


class MultiTargetAIPerfJob:
    """Manages multiple AIPerf jobs targeting frontend(s).

    Automatically splits high concurrency across multiple pods for better
    resource utilization and fault isolation.

    Note: Requires a container image with AIPerf installed (e.g., the Dynamo container).
    """

    # Default max concurrency per pod - keeps memory usage reasonable
    DEFAULT_MAX_CONCURRENCY_PER_POD = 4000

    def __init__(
        self,
        namespace: str,
        frontend_urls: List[str],
        model: str,
        duration_sec: int,
        config: Optional[AIPerfConfig] = None,
        job_name_prefix: str = "scale-test-aiperf",
        image: Optional[str] = None,
        tokenizer: Optional[str] = None,
        max_concurrency_per_pod: Optional[int] = None,
    ):

        self.namespace = namespace
        self.frontend_urls = frontend_urls
        self.model = model
        self.duration_sec = duration_sec
        self.config = config or AIPerfConfig()
        self.job_name_prefix = job_name_prefix
        self.image = image or DEFAULT_AIPERF_IMAGE
        self.tokenizer = tokenizer or model
        self.max_concurrency_per_pod = (
            max_concurrency_per_pod or self.DEFAULT_MAX_CONCURRENCY_PER_POD
        )

        self._jobs: List[AIPerfLoadGeneratorJob] = []

    def _create_jobs(self) -> None:
        """Create AIPerf job instances, splitting high concurrency across pods."""
        self._jobs = []

        total_concurrency = self.config.concurrency
        total_requests = self.config.request_count

        # Calculate how many pods we need based on max concurrency per pod
        num_pods = max(1, (total_concurrency + self.max_concurrency_per_pod - 1) // self.max_concurrency_per_pod)

        # Distribute concurrency and requests across pods
        base_concurrency = total_concurrency // num_pods
        concurrency_remainder = total_concurrency % num_pods

        base_requests = total_requests // num_pods if total_requests > 0 else 0
        requests_remainder = total_requests % num_pods if total_requests > 0 else 0

        logger.info(
            f"Splitting concurrency={total_concurrency} across {num_pods} pods "
            f"(max {self.max_concurrency_per_pod} per pod)"
        )

        for i in range(num_pods):
            # Distribute concurrency (give remainder to first pods)
            pod_concurrency = base_concurrency + (1 if i < concurrency_remainder else 0)
            if pod_concurrency == 0:
                continue

            # Distribute requests proportionally
            pod_requests = base_requests + (1 if i < requests_remainder else 0) if total_requests > 0 else 0

            # Round-robin across frontend URLs
            url = self.frontend_urls[i % len(self.frontend_urls)]

            job_config = AIPerfConfig(
                isl_mean=self.config.isl_mean,
                isl_stddev=self.config.isl_stddev,
                osl_mean=self.config.osl_mean,
                osl_stddev=self.config.osl_stddev,
                concurrency=pod_concurrency,
                request_count=pod_requests,
                warmup_request_count=self.config.warmup_request_count,
                prefix_prompt_length=self.config.prefix_prompt_length,
                num_prefix_prompts=self.config.num_prefix_prompts,
                random_seed=self.config.random_seed + i,
                streaming=self.config.streaming,
                ignore_eos=self.config.ignore_eos,
            )

            job = AIPerfLoadGeneratorJob(
                namespace=self.namespace,
                frontend_url=url,
                model=self.model,
                duration_sec=self.duration_sec,
                config=job_config,
                job_name=f"{self.job_name_prefix}-{i}",
                image=self.image,
                tokenizer=self.tokenizer,
            )
            self._jobs.append(job)

        logger.info(f"Created {len(self._jobs)} AIPerf jobs")

    async def create_and_wait(
        self,
        batch_api: client.BatchV1Api,
        core_api: client.CoreV1Api,
        timeout: int = 600,
    ) -> bool:
        """Create all AIPerf jobs and wait for completion."""
        self._create_jobs()

        logger.info(
            f"Creating {len(self._jobs)} AIPerf jobs for {len(self.frontend_urls)} frontends"
        )

        # Create all jobs
        create_tasks = [
            job.create_and_wait(batch_api, core_api, timeout) for job in self._jobs
        ]

        results = await asyncio.gather(*create_tasks, return_exceptions=True)

        # Check results
        success = True
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Job {i} failed with exception: {result}")
                success = False
            elif not result:
                logger.error(f"Job {i} failed")
                success = False

        return success

    async def delete(self) -> None:
        """Delete all jobs."""
        delete_tasks = [job.delete() for job in self._jobs]
        await asyncio.gather(*delete_tasks, return_exceptions=True)

