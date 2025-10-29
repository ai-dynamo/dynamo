# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end test for differential scaling across multiple DynamoGraphDeployments.

This test assumes the global planner/scheduler deployment is running with two
DynamoGraphDeployments (llama-deployment-a and llama-deployment-b). It tests that:
1. The scheduler can route requests to specific deployments using route-to-index
2. Heavy load on one deployment triggers scaling while light load doesn't
3. The global planner correctly manages resources across both deployments

Test scenario:
- Deployment A (index 0): Heavy load (18 req/s) -> should scale from 1P1D to 2P1D
- Deployment B (index 1): Light load (4 req/s) -> should remain at 1P1D
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
HEALTH_CHECK_TIMEOUT = 10
PORT_FORWARD_SETUP_DELAY = 3
FINAL_STABILIZATION_DELAY = 90
MONITORING_INTERVAL = 15
BUFFER_DURATION = 90

# Scheduler configuration
SCHEDULER_PORT = 8080
SCHEDULER_SERVICE_NAME = "dynamo-scheduler"


@dataclass
class PodCounts:
    """Track pod counts at a specific time for a deployment."""

    timestamp: float
    deployment_name: str
    prefill_pods: int
    decode_pods: int
    total_pods: int

    def __str__(self):
        return f"{self.deployment_name}: P={self.prefill_pods}, D={self.decode_pods}, Total={self.total_pods}"


class KubernetesMonitor:
    """Monitor Kubernetes deployments and pod scaling."""

    def __init__(
        self,
        namespace: str = "default",
        deployment_names: List[str] = None,
    ):
        self.namespace = namespace
        self.deployment_names = deployment_names or []
        self.pod_history: Dict[str, List[PodCounts]] = {
            name: [] for name in self.deployment_names
        }

    def _run_kubectl(self, cmd: List[str]) -> Tuple[bool, str]:
        """Run kubectl command and return success status and output."""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0, result.stdout.strip()
        except subprocess.TimeoutExpired:
            logger.error(f"kubectl command timed out: {' '.join(cmd)}")
            return False, ""
        except Exception as e:
            logger.error(f"kubectl command failed: {e}")
            return False, ""

    def get_pod_counts(self, deployment_name: str) -> Optional[PodCounts]:
        """Get current pod counts for prefill and decode workers of a specific deployment."""
        cmd = [
            "kubectl",
            "get",
            "pods",
            "-n",
            self.namespace,
            "--selector",
            f"nvidia.com/dynamo-namespace={deployment_name}",
            "-o",
            "json",
        ]

        success, output = self._run_kubectl(cmd)
        if not success:
            logger.warning(f"Failed to get pod counts for {deployment_name}")
            return None

        try:
            data = json.loads(output)
            prefill_pods = 0
            decode_pods = 0
            total_pods = 0

            for pod in data.get("items", []):
                pod_phase = pod.get("status", {}).get("phase", "")
                pod_name = pod.get("metadata", {}).get("name", "")
                pod_labels = pod.get("metadata", {}).get("labels", {})
                
                # Check if it's a worker pod using selector label or name
                selector = pod_labels.get("nvidia.com/selector", "")
                
                # Only count Running worker pods
                if pod_phase == "Running":
                    # Identify prefill vs decode by selector label or pod name
                    if "prefill" in selector.lower() or "prefill" in pod_name.lower():
                        prefill_pods += 1
                        total_pods += 1
                    elif "decode" in selector.lower() or "decode" in pod_name.lower():
                        decode_pods += 1
                        total_pods += 1
                    # Skip frontend, planner, etc.

            counts = PodCounts(
                timestamp=time.time(),
                deployment_name=deployment_name,
                prefill_pods=prefill_pods,
                decode_pods=decode_pods,
                total_pods=total_pods,
            )

            self.pod_history[deployment_name].append(counts)
            return counts

        except Exception as e:
            logger.error(f"Failed to parse pod counts for {deployment_name}: {e}")
            return None

    def get_all_pod_counts(self) -> Dict[str, Optional[PodCounts]]:
        """Get pod counts for all monitored deployments."""
        counts = {}
        for deployment_name in self.deployment_names:
            counts[deployment_name] = self.get_pod_counts(deployment_name)
        return counts

    async def monitor_scaling(
        self, duration: int, interval: int = 10
    ) -> Dict[str, List[PodCounts]]:
        """Monitor pod scaling for all deployments for a given duration."""
        logger.info(
            f"Monitoring pod scaling for {len(self.deployment_names)} deployments "
            f"for {duration}s (interval: {interval}s)"
        )

        start_time = time.time()

        while time.time() - start_time < duration:
            all_counts = self.get_all_pod_counts()
            for deployment_name, counts in all_counts.items():
                if counts:
                    logger.info(f"Pod counts: {counts}")

            await asyncio.sleep(interval)

        return self.pod_history


class DifferentialLoadGenerator:
    """Generate differential load across multiple deployment endpoints."""

    def __init__(
        self,
        scheduler_base_url: str = "http://localhost:8080",
        model: str = "nvidia/Llama-3.1-8B-Instruct-FP8",
        isl: int = 4000,
        osl: int = 150,
        save_results: bool = False,
    ):
        self.scheduler_base_url = scheduler_base_url
        self.model = model
        self.isl = isl
        self.osl = osl
        self.save_results = save_results

    async def _make_request(self, method: str, url: str, **kwargs):
        """Make an HTTP request.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Full URL to request
            **kwargs: Additional arguments to pass to httpx
            
        Returns:
            httpx.Response object
        """
        import httpx
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            if method == "GET":
                return await client.get(url, **kwargs)
            elif method == "POST":
                return await client.post(url, **kwargs)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

    async def generate_load_to_endpoint(
        self,
        frontend_index: int,
        req_per_sec: float,
        duration_sec: int,
        artifact_dir: str,
        endpoint_name: str = "endpoint",
    ) -> Dict[str, Any]:
        """
        Generate load to a specific frontend index through the scheduler.

        Args:
            frontend_index: Index of the frontend to route to (0, 1, etc.)
            req_per_sec: Target requests per second
            duration_sec: Duration to generate load (seconds)
            artifact_dir: Directory to store aiperf artifacts
            endpoint_name: Name for logging/identification

        Returns:
            Dictionary with load test results
        """
        logger.info(
            f"Generating load to {endpoint_name} (index {frontend_index}): "
            f"{req_per_sec} req/s for {duration_sec}s"
        )

        # Calculate request count
        request_count = max(1, int(req_per_sec * duration_sec))

        # Build scheduler URL with route-to-index
        # The scheduler expects: /route/{frontend_index}/{path}
        # So we route to: /route/{frontend_index}/v1/chat/completions
        scheduler_url = self.scheduler_base_url.replace("http://", "")

        # Build aiperf command
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
            scheduler_url,
            "--endpoint",
            f"/route/{frontend_index}/v1/chat/completions",
            "--streaming",
            "--synthetic-input-tokens-mean",
            str(self.isl),
            "--output-tokens-mean",
            str(self.osl),
            "--request-rate",
            str(req_per_sec),
            "--request-count",
            str(request_count),
            "--num-dataset-entries",
            str(max(20, int(req_per_sec * 10))),
            "--artifact-dir",
            artifact_dir,
            "-v",
        ]

        logger.info(f"[{endpoint_name}] Running command: {' '.join(cmd)}")
        logger.info(
            f"[{endpoint_name}] Expected duration: {duration_sec}s, "
            f"timeout: {max(duration_sec * 2 + 120, int(duration_sec * 2.5))}s"
        )

        # Run aiperf
        start_time = time.time()
        timeout = max(duration_sec * 2 + 120, int(duration_sec * 2.5))
        
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
                logger.error(f"[{endpoint_name}] aiperf timed out")
                raise RuntimeError(f"Load generation timed out for {endpoint_name}")

            end_time = time.time()
            actual_duration = end_time - start_time

            # Persist logs
            import os
            try:
                os.makedirs(artifact_dir, exist_ok=True)
                with open(os.path.join(artifact_dir, "aiperf.stdout.log"), "wb") as f:
                    f.write(stdout or b"")
                with open(os.path.join(artifact_dir, "aiperf.stderr.log"), "wb") as f:
                    f.write(stderr or b"")
            except Exception:
                pass

            if proc.returncode == 0:
                logger.info(f"[{endpoint_name}] Load generation completed successfully")
                logger.info(f"[{endpoint_name}] Actual duration: {actual_duration:.2f}s")
                results = self._parse_aiperf_results(artifact_dir, endpoint_name)
                results.update(
                    {
                        "endpoint_name": endpoint_name,
                        "frontend_index": frontend_index,
                        "requested_req_per_sec": req_per_sec,
                        "actual_duration": actual_duration,
                        "target_duration": duration_sec,
                        "artifact_dir": artifact_dir,
                        "success": True,
                    }
                )
                return results
            else:
                logger.error(
                    f"[{endpoint_name}] aiperf failed with return code {proc.returncode}"
                )
                raise RuntimeError(f"aiperf failed for {endpoint_name}")
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"[{endpoint_name}] aiperf execution error: {e}")
            raise

    def _parse_aiperf_results(
        self, artifact_dir: str, endpoint_name: str
    ) -> Dict[str, Any]:
        """Parse aiperf results from artifact directory."""
        import os
        try:
            # Look for JSON results
            json_files = [f for f in os.listdir(artifact_dir) if f.endswith(".json")]
            if not json_files:
                logger.warning(
                    f"[{endpoint_name}] No JSON results found in artifact directory"
                )
                return {}

            # Find main results file
            results_file = None
            for json_file in json_files:
                if "profile_export" in json_file or "aiperf" in json_file:
                    results_file = os.path.join(artifact_dir, json_file)
                    break

            if not results_file:
                results_file = os.path.join(artifact_dir, json_files[0])

            logger.info(f"[{endpoint_name}] Parsing results from: {results_file}")

            with open(results_file, "r") as f:
                metrics = json.load(f)

            results = {
                "throughput": metrics.get("output_token_throughput", {}).get("avg", 0),
                "ttft_mean": metrics.get("time_to_first_token", {}).get("avg", 0),
                "itl_mean": metrics.get("inter_token_latency", {}).get("avg", 0),
                "end_to_end_latency_mean": metrics.get("request_latency", {}).get(
                    "avg", 0
                ),
            }
            logger.info(f"[{endpoint_name}] Parsed results: {results}")
            return results

        except Exception as e:
            logger.warning(f"[{endpoint_name}] Failed to parse aiperf results: {e}")
            return {}

    async def discover_deployment_indices(
        self, target_deployments: List[str]
    ) -> Dict[str, int]:
        """
        Discover which frontend index corresponds to which deployment.
        
        Args:
            target_deployments: List of deployment names to find
            
        Returns:
            Dictionary mapping deployment name to frontend index
        """
        deployment_to_index = {}
        
        try:
            # Query scheduler to find out how many frontends exist
            frontends_response = await self._make_request(
                "GET", f"{self.scheduler_base_url}/frontends"
            )
            frontends_data = frontends_response.json()
            total_frontends = frontends_data.get("total", 0)
            
            logger.info(f"Scheduler has {total_frontends} frontends available")
            
            # Query each frontend index to find its deployment
            for index in range(total_frontends):
                try:
                    response = await self._make_request(
                        "GET", f"{self.scheduler_base_url}/frontends/{index}"
                    )
                    frontend_info = response.json()
                    deployment_name = frontend_info.get("deployment_name", "unknown")
                    
                    logger.info(f"  Index {index}: {deployment_name} ({frontend_info.get('url', 'N/A')})")
                    
                    if deployment_name in target_deployments:
                        deployment_to_index[deployment_name] = index
                        
                except Exception as e:
                    logger.warning(f"Failed to query frontend index {index}: {e}")
                    
            # Verify we found all target deployments
            missing = set(target_deployments) - set(deployment_to_index.keys())
            if missing:
                logger.error(f"Could not find frontend indices for deployments: {missing}")
                raise ValueError(f"Missing deployments: {missing}")
                
            return deployment_to_index
            
        except Exception as e:
            logger.error(f"Failed to discover deployment indices: {e}")
            raise

    async def run_differential_load_test(
        self, deployment_a_rps: float, deployment_b_rps: float, duration_sec: int,
        deployment_a_name: str = None, deployment_b_name: str = None
    ) -> Dict[str, Any]:
        """
        Run concurrent load tests against two deployments with different rates.

        Args:
            deployment_a_rps: Request rate for deployment A
            deployment_b_rps: Request rate for deployment B
            duration_sec: Duration of the load test
            deployment_a_name: Name of deployment A (defaults to first in list)
            deployment_b_name: Name of deployment B (defaults to second in list)

        Returns:
            Dictionary with results from both deployments
        """
        logger.info("Starting differential load test")
        
        # Discover which frontend index corresponds to which deployment
        deployment_indices = await self.discover_deployment_indices(
            [deployment_a_name or "llama-deployment-a", deployment_b_name or "llama-deployment-b"]
        )
        
        index_a = deployment_indices.get(deployment_a_name or "llama-deployment-a")
        index_b = deployment_indices.get(deployment_b_name or "llama-deployment-b")
        
        logger.info(f"  Deployment A ({deployment_a_name or 'llama-deployment-a'}, index {index_a}): {deployment_a_rps} req/s")
        logger.info(f"  Deployment B ({deployment_b_name or 'llama-deployment-b'}, index {index_b}): {deployment_b_rps} req/s")
        logger.info(f"  Duration: {duration_sec}s")

        # Create artifact directories
        import os
        import tempfile
        
        timestamp = int(time.time())
        if self.save_results:
            base_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "test_results",
                f"differential_scaling_{timestamp}",
            )
        else:
            base_dir = os.path.join(
                tempfile.gettempdir(), f"differential_scaling_{timestamp}"
            )

        os.makedirs(base_dir, exist_ok=True)
        logger.info(f"Saving results to: {base_dir}")

        deployment_a_dir = os.path.join(base_dir, "deployment_a")
        deployment_b_dir = os.path.join(base_dir, "deployment_b")

        # Run both load tests concurrently
        try:
            results_a, results_b = await asyncio.gather(
                self.generate_load_to_endpoint(
                    frontend_index=index_a,
                    req_per_sec=deployment_a_rps,
                    duration_sec=duration_sec,
                    artifact_dir=deployment_a_dir,
                    endpoint_name=f"Deployment-A-{deployment_a_name or 'llama-deployment-a'}",
                ),
                self.generate_load_to_endpoint(
                    frontend_index=index_b,
                    req_per_sec=deployment_b_rps,
                    duration_sec=duration_sec,
                    artifact_dir=deployment_b_dir,
                    endpoint_name=f"Deployment-B-{deployment_b_name or 'llama-deployment-b'}",
                ),
            )

            return {
                "test_timestamp": timestamp,
                "base_dir": base_dir,
                "deployment_a": results_a,
                "deployment_b": results_b,
                "config": {
                    "deployment_a_rps": deployment_a_rps,
                    "deployment_b_rps": deployment_b_rps,
                    "duration": duration_sec,
                },
            }

        except Exception as e:
            logger.error(f"Differential load test failed: {e}")
            raise


class DifferentialScalingTest:
    """End-to-end test for differential scaling across multiple deployments."""

    def __init__(
        self,
        namespace: str = "default",
        scheduler_url: str = "http://localhost:8080",
        deployment_names: List[str] = None,
        save_results: bool = False,
    ):
        self.namespace = namespace
        self.scheduler_url = scheduler_url
        self.deployment_names = deployment_names or [
            "llama-deployment-a",
            "llama-deployment-b",
        ]
        self.save_results = save_results

        self.k8s_monitor = KubernetesMonitor(
            namespace=namespace, deployment_names=self.deployment_names
        )
        self.load_generator = DifferentialLoadGenerator(
            scheduler_base_url=scheduler_url, save_results=save_results
        )

        self.test_results: Dict[str, Any] = {}

    async def run_differential_scaling_test(self) -> Dict:
        """
        Run the complete differential scaling test.

        Test scenario:
        - Deployment A: 18 req/s (heavy load) -> should scale to 2P1D
        - Deployment B: 4 req/s (light load) -> should remain at 1P1D
        - Duration: 180 seconds to allow time for scaling
        """
        logger.info("Starting differential scaling test")
        logger.info(f"Monitoring deployments: {self.deployment_names}")

        test_start_time = time.time()

        # Record initial state for all deployments
        initial_counts = self.k8s_monitor.get_all_pod_counts()
        for deployment_name, counts in initial_counts.items():
            logger.info(f"Initial state for {deployment_name}: {counts}")

        # Test parameters
        deployment_a_rps = 18.0  # Heavy load - should trigger scaling
        deployment_b_rps = 4.0  # Light load - should not trigger scaling
        load_duration = 180  # 3 minutes of load

        # Calculate total monitoring duration
        total_test_duration = load_duration + BUFFER_DURATION
        monitoring_task = asyncio.create_task(
            self.k8s_monitor.monitor_scaling(
                total_test_duration, interval=MONITORING_INTERVAL
            )
        )

        # Initialize results
        load_results = {}

        try:
            # Run differential load test
            logger.info(
                f"Running differential load: A={deployment_a_rps} req/s, B={deployment_b_rps} req/s"
            )
            load_results = await self.load_generator.run_differential_load_test(
                deployment_a_rps=deployment_a_rps,
                deployment_b_rps=deployment_b_rps,
                duration_sec=load_duration,
            )

            # Check final pod counts
            final_counts = self.k8s_monitor.get_all_pod_counts()
            for deployment_name, counts in final_counts.items():
                logger.info(f"Final state for {deployment_name}: {counts}")

            # Wait for stabilization
            logger.info(f"Waiting {FINAL_STABILIZATION_DELAY}s for stabilization...")
            await asyncio.sleep(FINAL_STABILIZATION_DELAY)

            # Get final final counts
            final_final_counts = self.k8s_monitor.get_all_pod_counts()
            for deployment_name, counts in final_final_counts.items():
                logger.info(f"Final final state for {deployment_name}: {counts}")

        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            raise
        finally:
            # Stop monitoring
            monitoring_task.cancel()
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass

        # Compile results
        test_results: Dict[str, Any] = {
            "test_duration": time.time() - test_start_time,
            "config": {
                "deployment_a_rps": deployment_a_rps,
                "deployment_b_rps": deployment_b_rps,
                "load_duration": load_duration,
                "deployments": self.deployment_names,
            },
            "initial_pod_counts": {
                name: counts.__dict__ if counts else None
                for name, counts in initial_counts.items()
            },
            "load_results": load_results,
            "final_pod_counts": {
                name: counts.__dict__ if counts else None
                for name, counts in final_counts.items()
            },
            "final_final_pod_counts": {
                name: counts.__dict__ if counts else None
                for name, counts in final_final_counts.items()
            },
            "pod_history": {
                name: [counts.__dict__ for counts in history]
                for name, history in self.k8s_monitor.pod_history.items()
            },
            "scaling_analysis": self.analyze_differential_scaling(),
        }

        return test_results

    def analyze_differential_scaling(self) -> Dict:
        """Analyze the differential scaling behavior from pod history."""
        analysis = {}

        for deployment_name in self.deployment_names:
            history = self.k8s_monitor.pod_history.get(deployment_name, [])

            if len(history) < 2:
                analysis[deployment_name] = {"error": "Insufficient data for analysis"}
                continue

            # Find scaling events
            scaling_events = []
            for i in range(1, len(history)):
                prev = history[i - 1]
                curr = history[i]

                if (
                    curr.prefill_pods != prev.prefill_pods
                    or curr.decode_pods != prev.decode_pods
                ):
                    scaling_events.append(
                        {
                            "timestamp": curr.timestamp,
                            "from": f"P={prev.prefill_pods}, D={prev.decode_pods}",
                            "to": f"P={curr.prefill_pods}, D={curr.decode_pods}",
                            "change": {
                                "prefill": curr.prefill_pods - prev.prefill_pods,
                                "decode": curr.decode_pods - prev.decode_pods,
                            },
                        }
                    )

            # Check expected behavior
            initial = history[0]
            final = history[-1]

            analysis[deployment_name] = {
                "scaling_events": scaling_events,
                "initial_state": f"P={initial.prefill_pods}, D={initial.decode_pods}",
                "final_state": f"P={final.prefill_pods}, D={final.decode_pods}",
                "total_scaling_events": len(scaling_events),
                "initial_pods": {
                    "prefill": initial.prefill_pods,
                    "decode": initial.decode_pods,
                },
                "final_pods": {
                    "prefill": final.prefill_pods,
                    "decode": final.decode_pods,
                },
            }

        return analysis

    def validate_test_results(self, results: Dict) -> Dict:
        """
        Validate that the test achieved expected differential scaling behavior.

        Expected:
        - Deployment A (heavy load): 1P1D -> 2P1D
        - Deployment B (light load): 1P1D -> 1P1D (no scaling)
        """
        validation: Dict[str, Any] = {
            "test_passed": False,
            "issues": [],
            "summary": "",
            "deployment_results": {},
        }

        analysis = results.get("scaling_analysis", {})

        # Expected behavior
        expected_behavior = {
            "llama-deployment-a": {"initial_prefill": 1, "final_prefill": 2},
            "llama-deployment-b": {"initial_prefill": 1, "final_prefill": 1},
        }

        all_passed = True

        for deployment_name, expected in expected_behavior.items():
            deployment_analysis = analysis.get(deployment_name, {})
            initial_pods = deployment_analysis.get("initial_pods", {})
            final_pods = deployment_analysis.get("final_pods", {})

            initial_prefill = initial_pods.get("prefill", 0)
            final_prefill = final_pods.get("prefill", 0)

            passed = (
                initial_prefill == expected["initial_prefill"]
                and final_prefill == expected["final_prefill"]
            )

            validation["deployment_results"][deployment_name] = {
                "passed": passed,
                "expected_initial": expected["initial_prefill"],
                "actual_initial": initial_prefill,
                "expected_final": expected["final_prefill"],
                "actual_final": final_prefill,
                "scaling_events": deployment_analysis.get("total_scaling_events", 0),
            }

            if not passed:
                all_passed = False
                if initial_prefill != expected["initial_prefill"]:
                    validation["issues"].append(
                        f"{deployment_name}: Started with {initial_prefill}P "
                        f"instead of {expected['initial_prefill']}P"
                    )
                if final_prefill != expected["final_prefill"]:
                    validation["issues"].append(
                        f"{deployment_name}: Ended with {final_prefill}P "
                        f"instead of {expected['final_prefill']}P"
                    )

        validation["test_passed"] = all_passed

        if all_passed:
            validation["summary"] = (
                "✅ Test PASSED: Differential scaling worked as expected\n"
                "  - Deployment A (heavy load): scaled 1P1D -> 2P1D\n"
                "  - Deployment B (light load): remained at 1P1D"
            )
        else:
            validation["summary"] = (
                "❌ Test FAILED: Differential scaling did not work as expected"
            )

        # Add performance metrics if available
        load_results = results.get("load_results", {})
        deployment_a = load_results.get("deployment_a", {})
        deployment_b = load_results.get("deployment_b", {})

        if deployment_a.get("throughput", 0) > 0:
            validation["deployment_a_throughput"] = (
                f"{deployment_a['throughput']:.2f} tokens/s"
            )
        if deployment_b.get("throughput", 0) > 0:
            validation["deployment_b_throughput"] = (
                f"{deployment_b['throughput']:.2f} tokens/s"
            )

        return validation


async def main():
    """Main function for running the differential scaling test."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Differential Scaling E2E Test for Global Planner/Scheduler"
    )
    parser.add_argument("--namespace", default="default", help="Kubernetes namespace")
    parser.add_argument(
        "--scheduler-url",
        default="http://localhost:8080",
        help="Scheduler service URL",
    )
    parser.add_argument(
        "--deployment-a",
        default="llama-deployment-a",
        help="Name of first deployment",
    )
    parser.add_argument(
        "--deployment-b",
        default="llama-deployment-b",
        help="Name of second deployment",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save results to test_results/ directory instead of /tmp",
    )

    args = parser.parse_args()

    test = DifferentialScalingTest(
        namespace=args.namespace,
        scheduler_url=args.scheduler_url,
        deployment_names=[args.deployment_a, args.deployment_b],
        save_results=args.save_results,
    )

    try:
        logger.info(f"Checking scheduler availability at {args.scheduler_url}...")

        # Run the differential scaling test
        logger.info("Running differential scaling test...")
        results = await test.run_differential_scaling_test()

        # Validate results
        validation = test.validate_test_results(results)

        # Save results
        import os
        timestamp = int(time.time())
        if args.save_results:
            results_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "test_results"
            )
            os.makedirs(results_dir, exist_ok=True)
            results_file = os.path.join(
                results_dir, f"differential_scaling_results_{timestamp}.json"
            )
        else:
            results_file = f"/tmp/differential_scaling_results_{timestamp}.json"

        with open(results_file, "w") as f:
            json.dump({"results": results, "validation": validation}, f, indent=2)

        # Print summary
        logger.info("=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)
        logger.info(validation["summary"])

        if validation["issues"]:
            logger.info("\nIssues found:")
            for issue in validation["issues"]:
                logger.info(f"  - {issue}")

        logger.info("\nDeployment Results:")
        for deployment_name, result in validation["deployment_results"].items():
            status = "✅ PASSED" if result["passed"] else "❌ FAILED"
            logger.info(f"  {deployment_name}: {status}")
            logger.info(
                f"    Initial: {result['actual_initial']}P (expected {result['expected_initial']}P)"
            )
            logger.info(
                f"    Final: {result['actual_final']}P (expected {result['expected_final']}P)"
            )
            logger.info(f"    Scaling events: {result['scaling_events']}")

        if any(k.endswith("_throughput") for k in validation.keys()):
            logger.info("\nPerformance:")
            if "deployment_a_throughput" in validation:
                logger.info(
                    f"  Deployment A: {validation['deployment_a_throughput']}"
                )
            if "deployment_b_throughput" in validation:
                logger.info(
                    f"  Deployment B: {validation['deployment_b_throughput']}"
                )

        logger.info(f"\nDetailed results saved to: {results_file}")
        logger.info("=" * 80)

        return 0 if validation["test_passed"] else 1

    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))

