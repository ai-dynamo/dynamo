# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SLO Comparison Test: Single Deployment vs Global Planner/Scheduler

This test compares two scenarios:
1. Single DynamoGraphDeployment with two concurrent aiperfs sending to the same endpoint
2. Two DynamoGraphDeployments under global planner/scheduler with two aiperfs routing to separate deployments

The test measures and compares SLO metrics:
- Throughput (tokens/sec)
- Time to First Token (TTFT)
- Inter-Token Latency (ITL)
- End-to-end latency
- Resource utilization
"""

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Test configuration constants
TEST_DURATION = 180  # 3 minutes (not used, kept for reference)
REQUEST_RATE_PER_STREAM = 3.0  # req/s per stream
REQUEST_COUNT = 1500  # Total requests per stream (~8 minutes at 3 req/s)
WARMUP_REQUEST_COUNT = 10  # Warmup requests before measurement
# Cooldown phase: extremely low load to trigger scale-down
COOLDOWN_RATE = 0.5  # Very low req/s to trigger planner scale-down
COOLDOWN_DURATION = 60  # 1 minute cooldown
ISL = 4000  # Input sequence length
OSL = 150  # Output sequence length


@dataclass
class LoadTestResult:
    """Results from a single load test."""

    scenario: str
    stream_name: str
    throughput: float
    ttft_mean: float
    ttft_p50: float
    ttft_p99: float
    itl_mean: float
    itl_p50: float
    itl_p99: float
    e2e_latency_mean: float
    e2e_latency_p50: float
    e2e_latency_p99: float
    requested_rps: float
    actual_duration: float
    artifact_dir: str
    success: bool
    goodput_constraints: str = ""


class LoadGenerator:
    """Generate load for SLO testing."""

    def __init__(
        self,
        model: str = "nvidia/Llama-3.1-8B-Instruct-FP8",
        isl: int = ISL,
        osl: int = OSL,
    ):
        self.model = model
        self.isl = isl
        self.osl = osl

    async def generate_load_direct(
        self,
        base_url: str,
        endpoint: str,
        req_per_sec: float,
        duration_sec: int,
        artifact_dir: str,
        stream_name: str = "stream",
        goodput: str = "",
        request_count: Optional[int] = None,
    ) -> LoadTestResult:
        """
        Generate load directly to a frontend endpoint.

        Args:
            base_url: Base URL (e.g., "localhost:8000")
            endpoint: Endpoint path (e.g., "/v1/chat/completions")
            req_per_sec: Target requests per second
            duration_sec: Duration to generate load (used for timeout, but actual duration determined by request_count)
            artifact_dir: Directory to store aiperf artifacts
            stream_name: Name for logging
            goodput: Goodput SLO constraints (e.g., "time_to_first_token:100 inter_token_latency:3.0")
            request_count: Optional override for request count (uses REQUEST_COUNT constant if None)

        Returns:
            LoadTestResult with metrics
        """
        # Use provided request_count or default to global constant
        actual_request_count = request_count if request_count is not None else REQUEST_COUNT
        
        logger.info(
            f"[{stream_name}] Generating load: {req_per_sec} req/s, {actual_request_count} requests (+ {WARMUP_REQUEST_COUNT} warmup) to {base_url}{endpoint}"
        )
        if goodput:
            logger.info(f"[{stream_name}] Goodput constraints: {goodput}")

        cmd = [
            "aiperf",
            "profile",
            "--model",
            self.model,
            "--tokenizer",
            self.model,
            "--endpoint-type",
            "chat",
            "--url",
            base_url,
            "--endpoint",
            endpoint,
            "--streaming",
            "--synthetic-input-tokens-mean",
            str(self.isl),
            "--output-tokens-mean",
            str(self.osl),
            "--request-rate",
            str(req_per_sec),
            "--request-count",
            str(actual_request_count),
            "--warmup-request-count",
            str(WARMUP_REQUEST_COUNT),
            "--num-dataset-entries",
            str(max(20, actual_request_count + WARMUP_REQUEST_COUNT)),
            "--artifact-dir",
            artifact_dir,
        ]
        
        # Add goodput constraints if provided
        if goodput:
            cmd.extend(["--goodput", goodput])
        
        cmd.append("-v")

        logger.info(f"[{stream_name}] Command: {' '.join(cmd)}")

        start_time = time.time()
        # Calculate expected duration based on request count and rate
        expected_duration = (actual_request_count + WARMUP_REQUEST_COUNT) / req_per_sec
        # Set timeout to 2x expected duration + 5 minutes buffer
        timeout = int(expected_duration * 2 + 300)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                logger.error(f"[{stream_name}] aiperf timed out")
                return LoadTestResult(
                    scenario="",
                    stream_name=stream_name,
                    throughput=0,
                    ttft_mean=0,
                    ttft_p50=0,
                    ttft_p99=0,
                    itl_mean=0,
                    itl_p50=0,
                    itl_p99=0,
                    e2e_latency_mean=0,
                    e2e_latency_p50=0,
                    e2e_latency_p99=0,
                    requested_rps=req_per_sec,
                    actual_duration=time.time() - start_time,
                    artifact_dir=artifact_dir,
                    success=False,
                    goodput_constraints=goodput,
                )

            end_time = time.time()
            actual_duration = end_time - start_time

            # Persist logs
            import os

            os.makedirs(artifact_dir, exist_ok=True)
            with open(os.path.join(artifact_dir, "aiperf.stdout.log"), "wb") as f:
                f.write(stdout or b"")
            with open(os.path.join(artifact_dir, "aiperf.stderr.log"), "wb") as f:
                f.write(stderr or b"")

            if proc.returncode == 0:
                logger.info(f"[{stream_name}] Load generation completed")
                metrics = self._parse_aiperf_results(artifact_dir, stream_name)
                return LoadTestResult(
                    scenario="",
                    stream_name=stream_name,
                    throughput=metrics.get("throughput", 0),
                    ttft_mean=metrics.get("ttft_mean", 0),
                    ttft_p50=metrics.get("ttft_p50", 0),
                    ttft_p99=metrics.get("ttft_p99", 0),
                    itl_mean=metrics.get("itl_mean", 0),
                    itl_p50=metrics.get("itl_p50", 0),
                    itl_p99=metrics.get("itl_p99", 0),
                    e2e_latency_mean=metrics.get("e2e_latency_mean", 0),
                    e2e_latency_p50=metrics.get("e2e_latency_p50", 0),
                    e2e_latency_p99=metrics.get("e2e_latency_p99", 0),
                    requested_rps=req_per_sec,
                    actual_duration=actual_duration,
                    artifact_dir=artifact_dir,
                    success=True,
                    goodput_constraints=goodput,
                )
            else:
                logger.error(f"[{stream_name}] aiperf failed: {proc.returncode}")
                # Log stderr for debugging
                if stderr:
                    stderr_text = stderr.decode('utf-8', errors='ignore')
                    logger.error(f"[{stream_name}] aiperf stderr:\n{stderr_text}")
                if stdout:
                    stdout_text = stdout.decode('utf-8', errors='ignore')
                    logger.error(f"[{stream_name}] aiperf stdout:\n{stdout_text}")
                return LoadTestResult(
                    scenario="",
                    stream_name=stream_name,
                    throughput=0,
                    ttft_mean=0,
                    ttft_p50=0,
                    ttft_p99=0,
                    itl_mean=0,
                    itl_p50=0,
                    itl_p99=0,
                    e2e_latency_mean=0,
                    e2e_latency_p50=0,
                    e2e_latency_p99=0,
                    requested_rps=req_per_sec,
                    actual_duration=actual_duration,
                    artifact_dir=artifact_dir,
                    success=False,
                    goodput_constraints=goodput,
                )

        except Exception as e:
            logger.error(f"[{stream_name}] aiperf execution error: {e}")
            return LoadTestResult(
                scenario="",
                stream_name=stream_name,
                throughput=0,
                ttft_mean=0,
                ttft_p50=0,
                ttft_p99=0,
                itl_mean=0,
                itl_p50=0,
                itl_p99=0,
                e2e_latency_mean=0,
                e2e_latency_p50=0,
                e2e_latency_p99=0,
                requested_rps=req_per_sec,
                actual_duration=time.time() - start_time,
                artifact_dir=artifact_dir,
                success=False,
                goodput_constraints=goodput,
            )

    def _parse_aiperf_results(
        self, artifact_dir: str, stream_name: str
    ) -> Dict[str, Any]:
        """Parse aiperf results from artifact directory."""
        import os

        try:
            json_files = [f for f in os.listdir(artifact_dir) if f.endswith(".json")]
            if not json_files:
                logger.warning(f"[{stream_name}] No JSON results found")
                return {}

            results_file = None
            for json_file in json_files:
                if "profile_export" in json_file or "aiperf" in json_file:
                    results_file = os.path.join(artifact_dir, json_file)
                    break

            if not results_file:
                results_file = os.path.join(artifact_dir, json_files[0])

            with open(results_file, "r") as f:
                data = json.load(f)

            metrics = {
                "throughput": data.get("output_token_throughput", {}).get("avg", 0),
                "ttft_mean": data.get("time_to_first_token", {}).get("avg", 0),
                "ttft_p50": data.get("time_to_first_token", {}).get("p50", 0),
                "ttft_p99": data.get("time_to_first_token", {}).get("p99", 0),
                "itl_mean": data.get("inter_token_latency", {}).get("avg", 0),
                "itl_p50": data.get("inter_token_latency", {}).get("p50", 0),
                "itl_p99": data.get("inter_token_latency", {}).get("p99", 0),
                "e2e_latency_mean": data.get("request_latency", {}).get("avg", 0),
                "e2e_latency_p50": data.get("request_latency", {}).get("p50", 0),
                "e2e_latency_p99": data.get("request_latency", {}).get("p99", 0),
            }
            return metrics

        except Exception as e:
            logger.warning(f"[{stream_name}] Failed to parse results: {e}")
            return {}


class SLOComparisonTest:
    """Compare SLO metrics between single deployment and global scheduler scenarios."""

    def __init__(
        self,
        namespace: str = "default",
        single_deployment_name: str = "llama-single",
        deployment_a_name: str = "llama-deployment-a",
        deployment_b_name: str = "llama-deployment-b",
        scheduler_url: str = "http://dynamo-scheduler",
        save_results: bool = False,
        stream1_goodput: str = "time_to_first_token:100 inter_token_latency:3.0",
        stream2_goodput: str = "time_to_first_token:150 inter_token_latency:4.0",
    ):
        self.namespace = namespace
        self.single_deployment_name = single_deployment_name
        self.deployment_a_name = deployment_a_name
        self.deployment_b_name = deployment_b_name
        self.scheduler_url = scheduler_url
        self.save_results = save_results
        self.stream1_goodput = stream1_goodput
        self.stream2_goodput = stream2_goodput
        self.load_generator = LoadGenerator()

    async def _discover_frontend_url(self, deployment_name: str) -> str:
        """Discover frontend service URL for a deployment."""
        # Use Kubernetes service DNS
        service_name = f"{deployment_name}-frontend"
        return f"http://{service_name}.{self.namespace}.svc.cluster.local:8000"
    
    async def _wait_for_frontend_ready(self, frontend_url: str, timeout: int = 300):
        """Wait for frontend to be ready and responding to health checks."""
        import httpx
        
        logger.info(f"Waiting for frontend to be ready: {frontend_url}")
        start_time = time.time()
        last_error = None
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            while time.time() - start_time < timeout:
                try:
                    # Try to hit the health endpoint or models endpoint
                    response = await client.get(f"{frontend_url}/health")
                    if response.status_code == 200:
                        logger.info(f"Frontend is ready: {frontend_url}")
                        return
                    last_error = f"Health check returned {response.status_code}"
                except Exception as e:
                    last_error = str(e)
                    logger.debug(f"Health check failed: {e}, retrying...")
                
                await asyncio.sleep(5)
        
        error_msg = f"Frontend not ready after {timeout}s. Last error: {last_error}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    async def _discover_deployment_indices(
        self, target_deployments: List[str]
    ) -> Dict[str, int]:
        """Discover which frontend index corresponds to which deployment."""
        import httpx

        deployment_to_index = {}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.scheduler_url}/frontends")
                frontends_data = response.json()
                total_frontends = frontends_data.get("total", 0)

                logger.info(f"Scheduler has {total_frontends} frontends")

                for index in range(total_frontends):
                    try:
                        response = await client.get(
                            f"{self.scheduler_url}/frontends/{index}"
                        )
                        frontend_info = response.json()
                        deployment_name = frontend_info.get("deployment_name", "")

                        logger.info(
                            f"  Index {index}: {deployment_name} ({frontend_info.get('url', 'N/A')})"
                        )

                        if deployment_name in target_deployments:
                            deployment_to_index[deployment_name] = index

                    except Exception as e:
                        logger.warning(f"Failed to query frontend {index}: {e}")

                missing = set(target_deployments) - set(deployment_to_index.keys())
                if missing:
                    raise ValueError(f"Missing deployments: {missing}")

                return deployment_to_index

        except Exception as e:
            logger.error(f"Failed to discover deployment indices: {e}")
            raise

    async def run_scenario_1_single_deployment(
        self, test_dir: str
    ) -> Dict[str, Any]:
        """
        Scenario 1: Single DynamoGraphDeployment with two concurrent aiperfs.

        Both streams send to the same frontend endpoint.
        """
        logger.info("=" * 80)
        logger.info("SCENARIO 1: Single Deployment with Concurrent Requests")
        logger.info("=" * 80)

        frontend_url = await self._discover_frontend_url(self.single_deployment_name)
        logger.info(f"Frontend URL: {frontend_url}")
        
        # Wait for frontend to be ready
        await self._wait_for_frontend_ready(frontend_url)

        # Remove http:// prefix for aiperf
        base_url = frontend_url.replace("http://", "")

        # Create artifact directories
        import os

        scenario_dir = os.path.join(test_dir, "scenario_1_single_deployment")
        os.makedirs(scenario_dir, exist_ok=True)

        stream1_dir = os.path.join(scenario_dir, "stream1")
        stream2_dir = os.path.join(scenario_dir, "stream2")

        # Run both load tests concurrently to the same endpoint
        logger.info(
            f"Running two concurrent streams at {REQUEST_RATE_PER_STREAM} req/s each"
        )

        start_time = time.time()

        results = await asyncio.gather(
            self.load_generator.generate_load_direct(
                base_url=base_url,
                endpoint="/v1/chat/completions",
                req_per_sec=REQUEST_RATE_PER_STREAM,
                duration_sec=TEST_DURATION,
                artifact_dir=stream1_dir,
                stream_name="Stream-1",
                goodput=self.stream1_goodput,
            ),
            self.load_generator.generate_load_direct(
                base_url=base_url,
                endpoint="/v1/chat/completions",
                req_per_sec=REQUEST_RATE_PER_STREAM,
                duration_sec=TEST_DURATION,
                artifact_dir=stream2_dir,
                stream_name="Stream-2",
                goodput=self.stream2_goodput,
            ),
        )

        total_time = time.time() - start_time

        result1, result2 = results
        result1.scenario = "single_deployment"
        result2.scenario = "single_deployment"

        # Cooldown phase: run extremely low load to trigger scale-down
        logger.info("=" * 80)
        logger.info("COOLDOWN: Running low load to trigger scale-down")
        logger.info(f"Rate: {COOLDOWN_RATE} req/s for {COOLDOWN_DURATION}s")
        logger.info("=" * 80)
        
        cooldown_dir = os.path.join(scenario_dir, "cooldown")
        cooldown_request_count = int(COOLDOWN_RATE * COOLDOWN_DURATION)
        await self.load_generator.generate_load_direct(
            base_url=base_url,
            endpoint="/v1/chat/completions",
            req_per_sec=COOLDOWN_RATE,
            duration_sec=COOLDOWN_DURATION,
            artifact_dir=cooldown_dir,
            stream_name="Cooldown",
            goodput="",  # No goodput constraints for cooldown
            request_count=cooldown_request_count,
        )
        
        logger.info("Cooldown complete - planner should scale down to 1P1D")

        return {
            "scenario": "single_deployment",
            "description": "Two concurrent streams to single deployment",
            "stream1": result1.__dict__,
            "stream2": result2.__dict__,
            "total_duration": total_time,
            "combined_throughput": result1.throughput + result2.throughput,
        }

    async def run_scenario_2_global_scheduler(self, test_dir: str) -> Dict[str, Any]:
        """
        Scenario 2: Two DynamoGraphDeployments with global scheduler.

        Each stream routes to a different deployment through scheduler.
        """
        logger.info("=" * 80)
        logger.info("SCENARIO 2: Global Scheduler with Separate Deployments")
        logger.info("=" * 80)

        # Discover deployment indices
        deployment_indices = await self._discover_deployment_indices(
            [self.deployment_a_name, self.deployment_b_name]
        )

        index_a = deployment_indices[self.deployment_a_name]
        index_b = deployment_indices[self.deployment_b_name]

        logger.info(f"Deployment A ({self.deployment_a_name}): index {index_a}")
        logger.info(f"Deployment B ({self.deployment_b_name}): index {index_b}")

        # Remove http:// prefix for aiperf
        scheduler_base_url = self.scheduler_url.replace("http://", "")

        # Create artifact directories
        import os

        scenario_dir = os.path.join(test_dir, "scenario_2_global_scheduler")
        os.makedirs(scenario_dir, exist_ok=True)

        stream1_dir = os.path.join(scenario_dir, "stream1_deployment_a")
        stream2_dir = os.path.join(scenario_dir, "stream2_deployment_b")

        # Run both load tests concurrently through scheduler routing
        logger.info(
            f"Running two streams at {REQUEST_RATE_PER_STREAM} req/s routed to separate deployments"
        )

        start_time = time.time()

        results = await asyncio.gather(
            self.load_generator.generate_load_direct(
                base_url=scheduler_base_url,
                endpoint=f"/route/{index_a}/v1/chat/completions",
                req_per_sec=REQUEST_RATE_PER_STREAM,
                duration_sec=TEST_DURATION,
                artifact_dir=stream1_dir,
                stream_name=f"Stream-1-to-{self.deployment_a_name}",
                goodput=self.stream1_goodput,
            ),
            self.load_generator.generate_load_direct(
                base_url=scheduler_base_url,
                endpoint=f"/route/{index_b}/v1/chat/completions",
                req_per_sec=REQUEST_RATE_PER_STREAM,
                duration_sec=TEST_DURATION,
                artifact_dir=stream2_dir,
                stream_name=f"Stream-2-to-{self.deployment_b_name}",
                goodput=self.stream2_goodput,
            ),
        )

        total_time = time.time() - start_time

        result1, result2 = results
        result1.scenario = "global_scheduler"
        result2.scenario = "global_scheduler"

        # Cooldown phase: run extremely low load to trigger scale-down on both deployments
        logger.info("=" * 80)
        logger.info("COOLDOWN: Running low load to trigger scale-down on both deployments")
        logger.info(f"Rate: {COOLDOWN_RATE} req/s per deployment for {COOLDOWN_DURATION}s")
        logger.info("=" * 80)
        
        cooldown_a_dir = os.path.join(scenario_dir, "cooldown_deployment_a")
        cooldown_b_dir = os.path.join(scenario_dir, "cooldown_deployment_b")
        cooldown_request_count = int(COOLDOWN_RATE * COOLDOWN_DURATION)
        
        # Run cooldown on both deployments concurrently
        await asyncio.gather(
            self.load_generator.generate_load_direct(
                base_url=scheduler_base_url,
                endpoint=f"/route/{index_a}/v1/chat/completions",
                req_per_sec=COOLDOWN_RATE,
                duration_sec=COOLDOWN_DURATION,
                artifact_dir=cooldown_a_dir,
                stream_name=f"Cooldown-{self.deployment_a_name}",
                goodput="",  # No goodput constraints for cooldown
                request_count=cooldown_request_count,
            ),
            self.load_generator.generate_load_direct(
                base_url=scheduler_base_url,
                endpoint=f"/route/{index_b}/v1/chat/completions",
                req_per_sec=COOLDOWN_RATE,
                duration_sec=COOLDOWN_DURATION,
                artifact_dir=cooldown_b_dir,
                stream_name=f"Cooldown-{self.deployment_b_name}",
                goodput="",  # No goodput constraints for cooldown
                request_count=cooldown_request_count,
            ),
        )
        
        logger.info("Cooldown complete - both planners should scale down to 1P1D")

        return {
            "scenario": "global_scheduler",
            "description": "Two streams routed to separate deployments",
            "stream1": result1.__dict__,
            "stream2": result2.__dict__,
            "total_duration": total_time,
            "combined_throughput": result1.throughput + result2.throughput,
        }

    def compare_results(
        self, scenario1_results: Dict, scenario2_results: Dict
    ) -> Dict[str, Any]:
        """Compare metrics between the two scenarios."""
        logger.info("=" * 80)
        logger.info("COMPARISON ANALYSIS")
        logger.info("=" * 80)

        s1_stream1 = scenario1_results["stream1"]
        s1_stream2 = scenario1_results["stream2"]
        s2_stream1 = scenario2_results["stream1"]
        s2_stream2 = scenario2_results["stream2"]

        # Calculate averages for each scenario
        s1_avg_throughput = (s1_stream1["throughput"] + s1_stream2["throughput"]) / 2
        s2_avg_throughput = (s2_stream1["throughput"] + s2_stream2["throughput"]) / 2

        s1_avg_ttft = (s1_stream1["ttft_mean"] + s1_stream2["ttft_mean"]) / 2
        s2_avg_ttft = (s2_stream1["ttft_mean"] + s2_stream2["ttft_mean"]) / 2

        s1_avg_itl = (s1_stream1["itl_mean"] + s1_stream2["itl_mean"]) / 2
        s2_avg_itl = (s2_stream1["itl_mean"] + s2_stream2["itl_mean"]) / 2

        s1_avg_e2e = (
            s1_stream1["e2e_latency_mean"] + s1_stream2["e2e_latency_mean"]
        ) / 2
        s2_avg_e2e = (
            s2_stream1["e2e_latency_mean"] + s2_stream2["e2e_latency_mean"]
        ) / 2

        comparison = {
            "throughput": {
                "scenario1_avg": s1_avg_throughput,
                "scenario2_avg": s2_avg_throughput,
                "scenario1_combined": scenario1_results["combined_throughput"],
                "scenario2_combined": scenario2_results["combined_throughput"],
                "improvement_pct": (
                    (s2_avg_throughput - s1_avg_throughput) / s1_avg_throughput * 100
                    if s1_avg_throughput > 0
                    else 0
                ),
            },
            "ttft": {
                "scenario1_avg": s1_avg_ttft,
                "scenario2_avg": s2_avg_ttft,
                "improvement_pct": (
                    (s1_avg_ttft - s2_avg_ttft) / s1_avg_ttft * 100
                    if s1_avg_ttft > 0
                    else 0
                ),
            },
            "itl": {
                "scenario1_avg": s1_avg_itl,
                "scenario2_avg": s2_avg_itl,
                "improvement_pct": (
                    (s1_avg_itl - s2_avg_itl) / s1_avg_itl * 100
                    if s1_avg_itl > 0
                    else 0
                ),
            },
            "e2e_latency": {
                "scenario1_avg": s1_avg_e2e,
                "scenario2_avg": s2_avg_e2e,
                "improvement_pct": (
                    (s1_avg_e2e - s2_avg_e2e) / s1_avg_e2e * 100
                    if s1_avg_e2e > 0
                    else 0
                ),
            },
        }

        # Print comparison
        logger.info("\nThroughput (tokens/sec):")
        logger.info(f"  Scenario 1 (Single): {s1_avg_throughput:.2f}")
        logger.info(f"  Scenario 2 (Global):  {s2_avg_throughput:.2f}")
        logger.info(
            f"  Improvement: {comparison['throughput']['improvement_pct']:+.2f}%"
        )

        logger.info("\nTime to First Token (ms):")
        logger.info(f"  Scenario 1 (Single): {s1_avg_ttft:.2f}")
        logger.info(f"  Scenario 2 (Global):  {s2_avg_ttft:.2f}")
        logger.info(f"  Improvement: {comparison['ttft']['improvement_pct']:+.2f}%")

        logger.info("\nInter-Token Latency (ms):")
        logger.info(f"  Scenario 1 (Single): {s1_avg_itl:.2f}")
        logger.info(f"  Scenario 2 (Global):  {s2_avg_itl:.2f}")
        logger.info(f"  Improvement: {comparison['itl']['improvement_pct']:+.2f}%")

        logger.info("\nEnd-to-End Latency (ms):")
        logger.info(f"  Scenario 1 (Single): {s1_avg_e2e:.2f}")
        logger.info(f"  Scenario 2 (Global):  {s2_avg_e2e:.2f}")
        logger.info(
            f"  Improvement: {comparison['e2e_latency']['improvement_pct']:+.2f}%"
        )

        return comparison

    async def run_full_comparison_test(self, scenario: str = "both") -> Dict[str, Any]:
        """Run specified scenario(s) and compare results if both are run.
        
        Args:
            scenario: Which scenario to run - "1", "2", or "both"
        """
        logger.info("Starting SLO Comparison Test")
        logger.info(f"Test Configuration:")
        logger.info(f"  Scenario: {scenario}")
        logger.info(f"  Duration: {TEST_DURATION}s")
        logger.info(f"  Request Rate (per stream): {REQUEST_RATE_PER_STREAM} req/s")
        logger.info(f"  Input Sequence Length: {ISL}")
        logger.info(f"  Output Sequence Length: {OSL}")
        logger.info("")

        # Create test directory
        import os
        import tempfile

        timestamp = int(time.time())
        if self.save_results:
            base_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "test_results",
                f"slo_comparison_{timestamp}",
            )
        else:
            base_dir = os.path.join(tempfile.gettempdir(), f"slo_comparison_{timestamp}")

        os.makedirs(base_dir, exist_ok=True)
        logger.info(f"Results directory: {base_dir}")

        scenario1_results = None
        scenario2_results = None
        comparison = None

        # Run Scenario 1 if requested
        if scenario in ["1", "both"]:
            scenario1_results = await self.run_scenario_1_single_deployment(base_dir)
            
            # Wait between scenarios if running both
            if scenario == "both":
                logger.info("\nWaiting 30s between scenarios...")
                await asyncio.sleep(30)

        # Run Scenario 2 if requested
        if scenario in ["2", "both"]:
            scenario2_results = await self.run_scenario_2_global_scheduler(base_dir)

        # Compare results only if both scenarios were run
        if scenario == "both" and scenario1_results and scenario2_results:
            comparison = self.compare_results(scenario1_results, scenario2_results)

        # Compile final results
        final_results = {
            "test_timestamp": timestamp,
            "test_config": {
                "duration": TEST_DURATION,
                "request_rate_per_stream": REQUEST_RATE_PER_STREAM,
                "isl": ISL,
                "osl": OSL,
            },
            "scenario1": scenario1_results,
            "scenario2": scenario2_results,
            "comparison": comparison,
            "results_dir": base_dir,
        }

        return final_results


async def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="SLO Comparison Test")
    parser.add_argument("--namespace", default="default", help="Kubernetes namespace")
    parser.add_argument(
        "--single-deployment",
        default="llama-single",
        help="Name of single deployment (scenario 1)",
    )
    parser.add_argument(
        "--deployment-a",
        default="llama-deployment-a",
        help="Name of first deployment (scenario 2)",
    )
    parser.add_argument(
        "--deployment-b",
        default="llama-deployment-b",
        help="Name of second deployment (scenario 2)",
    )
    parser.add_argument(
        "--scheduler-url",
        default="http://dynamo-scheduler",
        help="Scheduler URL",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save results to test_results/ directory",
    )
    parser.add_argument(
        "--stream1-goodput",
        default="time_to_first_token:100 inter_token_latency:3.0",
        help="Goodput constraints for stream 1 (stricter SLO)",
    )
    parser.add_argument(
        "--stream2-goodput",
        default="time_to_first_token:150 inter_token_latency:4.0",
        help="Goodput constraints for stream 2 (relaxed SLO)",
    )
    parser.add_argument(
        "--scenario",
        default="both",
        choices=["1", "2", "both"],
        help="Which scenario to run: 1 (single deployment), 2 (global scheduler), or both",
    )

    args = parser.parse_args()

    test = SLOComparisonTest(
        namespace=args.namespace,
        single_deployment_name=args.single_deployment,
        deployment_a_name=args.deployment_a,
        deployment_b_name=args.deployment_b,
        scheduler_url=args.scheduler_url,
        save_results=args.save_results,
        stream1_goodput=args.stream1_goodput,
        stream2_goodput=args.stream2_goodput,
    )

    try:
        results = await test.run_full_comparison_test(scenario=args.scenario)

        # Save results
        import os

        results_file = os.path.join(
            results["results_dir"], "slo_comparison_results.json"
        )
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info("=" * 80)
        logger.info("TEST COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {results_file}")

        return 0

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))

