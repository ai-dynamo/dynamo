# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end test for SLA planner scaling behavior.

This test assumes a disaggregated planner deployment is already running
and accessible at localhost:8000. It monitors pod scaling and validates
that the planner correctly scales from 1P1D to 2P1D when load increases
through graduated phases: 8 req/s (baseline) → 15 req/s (moderate) → 25 req/s (prefill scaling trigger).
"""

import asyncio
import json
import logging
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from utils.load_generator import LoadGenerator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Test configuration constants
HEALTH_CHECK_TIMEOUT = 10
PORT_FORWARD_SETUP_DELAY = 3
FINAL_STABILIZATION_DELAY = 60
MONITORING_INTERVAL = 15
BUFFER_DURATION = 90


@dataclass
class PodCounts:
    """Track pod counts at a specific time."""

    timestamp: float
    prefill_pods: int
    decode_pods: int
    total_pods: int

    def __str__(self):
        return f"P={self.prefill_pods}, D={self.decode_pods}, Total={self.total_pods}"


class KubernetesMonitor:
    """Monitor Kubernetes deployment and pod scaling."""

    def __init__(
        self, namespace: str = "default", deployment_name: str = "vllm-disagg-planner"
    ):
        self.namespace = namespace
        self.deployment_name = deployment_name
        self.pod_history: List[PodCounts] = []

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

    def get_pod_counts(self) -> Optional[PodCounts]:
        """Get current pod counts for prefill and decode workers."""
        cmd = [
            "kubectl",
            "get",
            "pods",
            "-n",
            self.namespace,
            "--selector",
            f"nvidia.com/dynamo-namespace={self.deployment_name}",
            "-o",
            "json",
        ]

        success, output = self._run_kubectl(cmd)
        if not success:
            logger.warning("Failed to get pod counts")
            return None

        try:
            data = json.loads(output)
            prefill_pods = 0
            decode_pods = 0
            total_pods = 0

            for pod in data.get("items", []):
                pod_phase = pod.get("status", {}).get("phase", "")
                pod_labels = pod.get("metadata", {}).get("labels", {})
                component = pod_labels.get("nvidia.com/dynamo-component", "")

                # Only count Running pods
                if pod_phase == "Running":
                    if component == "VllmPrefillWorker":
                        prefill_pods += 1
                    elif component == "VllmDecodeWorker":
                        decode_pods += 1
                    else:
                        continue
                    total_pods += 1

            counts = PodCounts(
                timestamp=time.time(),
                prefill_pods=prefill_pods,
                decode_pods=decode_pods,
                total_pods=total_pods,
            )

            self.pod_history.append(counts)
            return counts

        except Exception as e:
            logger.error(f"Failed to parse pod counts: {e}")
            return None

    async def monitor_scaling(
        self, duration: int, interval: int = 10
    ) -> List[PodCounts]:
        """Monitor pod scaling for a given duration."""
        logger.info(f"Monitoring pod scaling for {duration}s (interval: {interval}s)")

        start_time = time.time()
        monitoring_data = []

        while time.time() - start_time < duration:
            counts = self.get_pod_counts()
            if counts:
                monitoring_data.append(counts)
                logger.info(f"Pod counts: {counts}")

            await asyncio.sleep(interval)

        return monitoring_data

    def wait_for_deployment_ready(self, timeout: int = 300) -> bool:
        """Wait for deployment to be ready."""
        logger.info(f"Waiting for deployment {self.deployment_name} to be ready...")

        cmd = [
            "kubectl",
            "wait",
            "--for=condition=available",
            f"deployment/{self.deployment_name}",
            "-n",
            self.namespace,
            f"--timeout={timeout}s",
        ]

        success, output = self._run_kubectl(cmd)
        if success:
            logger.info("Deployment is ready")
            return True
        else:
            logger.error(f"Deployment failed to become ready: {output}")
            return False

    def apply_deployment(self, yaml_file: str) -> bool:
        """Apply Kubernetes deployment from YAML file."""
        logger.info(f"Applying deployment from {yaml_file}")

        cmd = ["kubectl", "apply", "-f", yaml_file, "-n", self.namespace]
        success, output = self._run_kubectl(cmd)

        if success:
            logger.info("Deployment applied successfully")
            return True
        else:
            logger.error(f"Failed to apply deployment: {output}")
            return False

    def delete_deployment(self, yaml_file: str) -> bool:
        """Delete Kubernetes deployment."""
        logger.info(f"Deleting deployment from {yaml_file}")

        cmd = [
            "kubectl",
            "delete",
            "-f",
            yaml_file,
            "-n",
            self.namespace,
            "--ignore-not-found",
        ]
        success, output = self._run_kubectl(cmd)

        if success:
            logger.info("Deployment deleted successfully")
        else:
            logger.warning(f"Failed to delete deployment: {output}")

        return success

    def check_service_health(
        self, service_name: str | None = None, port: int = 8000
    ) -> bool:
        """Check if the frontend service is healthy."""
        if service_name is None:
            service_name = f"{self.deployment_name}-frontend"

        # Port forward to check health
        cmd = [
            "kubectl",
            "port-forward",
            f"service/{service_name}",
            f"{port}:{port}",
            "-n",
            self.namespace,
        ]

        proc = None
        try:
            # Start port forwarding in background
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Give it a moment to establish connection
            time.sleep(PORT_FORWARD_SETUP_DELAY)

            # Try to check health endpoint
            try:
                response = urllib.request.urlopen(
                    f"http://localhost:{port}/health", timeout=HEALTH_CHECK_TIMEOUT
                )
                healthy = response.status == 200
                logger.info(f"Service health check: {'OK' if healthy else 'FAILED'}")
            except Exception as e:
                logger.warning(f"Health check failed: {e}")
                healthy = False

            return healthy

        except Exception as e:
            logger.error(f"Failed to check service health: {e}")
            return False
        finally:
            # Ensure port forwarding is terminated
            if proc is not None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()


class ScalingE2ETest:
    """End-to-end test for SLA planner scaling behavior."""

    def __init__(
        self,
        namespace: str = "default",
        base_url: str = "http://localhost:8000",
        save_results: bool = False,
    ):
        self.namespace = namespace
        self.base_url = base_url
        self.save_results = save_results

        self.k8s_monitor = KubernetesMonitor(namespace)
        self.load_generator = LoadGenerator(
            base_url=base_url, save_results=save_results
        )

        self.test_results: Dict[str, Any] = {}

    async def run_scaling_test(self) -> Dict:
        """
        Run the complete scaling test.

        Hardcoded scenario:
        - Phase 1 (12 req/s): Should maintain 1P1D
        - Phase 2 (24 req/s): Should scale to 2P1D
        """
        logger.info("Starting scaling integration test")

        test_start_time = time.time()

        # Record initial state
        initial_counts = self.k8s_monitor.get_pod_counts()
        logger.info(f"Test starting with: {initial_counts}")

        # Start background monitoring
        # Calculate based on actual phases from load generator
        # Phase durations: baseline(90s) + transition(30s) + moderate(120s) + transition(30s) + trigger(180s) + buffer
        total_test_duration = 90 + 30 + 120 + 30 + 180 + BUFFER_DURATION
        monitoring_task = asyncio.create_task(
            self.k8s_monitor.monitor_scaling(
                total_test_duration, interval=MONITORING_INTERVAL
            )
        )

        # Initialize results in case of exception
        baseline_results = {}
        moderate_results = {}
        trigger_results = {}

        try:
            # Use the load generator's built-in scaling test
            logger.info("Running scaling scenario (8 req/s -> 15 req/s -> 25 req/s)")
            load_results = await self.load_generator.run_scaling_test()

            # Extract load results for analysis (3-phase structure)
            phase_results = load_results.get("phase_results", {})
            baseline_results = phase_results.get("phase1_baseline", {})
            moderate_results = phase_results.get("phase2_moderate", {})
            trigger_results = phase_results.get("phase3_prefill_scaling_trigger", {})

            # Check final pod counts
            final_counts = self.k8s_monitor.get_pod_counts()
            logger.info(f"Final pod counts: {final_counts}")

            # Wait a bit more to capture any delayed scaling
            logger.info("Waiting for potential delayed scaling...")
            await asyncio.sleep(FINAL_STABILIZATION_DELAY)

            # Get final final counts
            final_final_counts = self.k8s_monitor.get_pod_counts()
            logger.info(f"Final final pod counts: {final_final_counts}")

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
                # Document actual test configuration
                "baseline_rps": 8.0,
                "moderate_rps": 15.0,
                "trigger_rps": 25.0,
                "phase_durations": {"baseline": 90, "moderate": 120, "trigger": 180},
                "transition_delay": 30,
            },
            "initial_pod_counts": initial_counts.__dict__ if initial_counts else None,
            "baseline_results": baseline_results,
            "moderate_results": moderate_results,
            "trigger_results": trigger_results,
            "final_pod_counts": final_counts.__dict__ if final_counts else None,
            "final_final_pod_counts": final_final_counts.__dict__
            if final_final_counts
            else None,
            "pod_history": [counts.__dict__ for counts in self.k8s_monitor.pod_history],
            "scaling_analysis": self.analyze_scaling_behavior(),
        }

        return test_results

    def analyze_scaling_behavior(self) -> Dict:
        """Analyze the scaling behavior from pod history."""
        if len(self.k8s_monitor.pod_history) < 2:
            return {"error": "Insufficient data for analysis"}

        history = self.k8s_monitor.pod_history

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

        # Check if expected scaling occurred
        initial = history[0]
        final = history[-1]

        expected_scaling = {
            "initial_1p1d": initial.prefill_pods == 1 and initial.decode_pods == 1,
            "final_2p1d": final.prefill_pods == 2 and final.decode_pods == 1,
            "scaling_occurred": len(scaling_events) > 0,
            "correct_scaling": (
                final.prefill_pods == 2
                and final.decode_pods == 1
                and initial.prefill_pods == 1
                and initial.decode_pods == 1
            ),
        }

        return {
            "scaling_events": scaling_events,
            "initial_state": f"P={initial.prefill_pods}, D={initial.decode_pods}",
            "final_state": f"P={final.prefill_pods}, D={final.decode_pods}",
            "expected_scaling": expected_scaling,
            "total_scaling_events": len(scaling_events),
        }

    def validate_test_results(self, results: Dict) -> Dict:
        """Validate that the test achieved expected scaling behavior."""
        validation: Dict[str, Any] = {"test_passed": False, "issues": [], "summary": ""}

        # Check if we have the expected data
        if not results.get("scaling_analysis"):
            validation["issues"].append("No scaling analysis data")
            return validation

        analysis = results["scaling_analysis"]
        expected = analysis.get("expected_scaling", {})

        # Validate initial state
        if not expected.get("initial_1p1d"):
            validation["issues"].append("Test did not start with 1P1D configuration")

        # Validate final state
        if not expected.get("final_2p1d"):
            validation["issues"].append(
                "Test did not end with expected 2P1D configuration"
            )

        # Validate scaling occurred
        if not expected.get("scaling_occurred"):
            validation["issues"].append("No scaling events detected")

        # Check if correct scaling occurred
        if expected.get("correct_scaling"):
            validation["test_passed"] = True
            validation[
                "summary"
            ] = "✅ Test PASSED: Successfully scaled from 1P1D to 2P1D"
        else:
            validation[
                "summary"
            ] = "❌ Test FAILED: Did not achieve expected 1P1D -> 2P1D scaling"

        # Add performance validation across all phases
        baseline = results.get("baseline_results", {})
        moderate = results.get("moderate_results", {})
        trigger = results.get("trigger_results", {})

        if baseline.get("throughput", 0) > 0:
            validation["baseline_throughput"] = f"{baseline['throughput']:.2f} req/s"
        if moderate.get("throughput", 0) > 0:
            validation["moderate_throughput"] = f"{moderate['throughput']:.2f} req/s"
        if trigger.get("throughput", 0) > 0:
            validation["trigger_throughput"] = f"{trigger['throughput']:.2f} req/s"

        return validation


async def main():
    """Main function for running the e2e test."""
    import argparse

    parser = argparse.ArgumentParser(description="SLA Planner Scaling E2E Test")
    parser.add_argument("--namespace", default="default", help="Kubernetes namespace")
    parser.add_argument(
        "--base-url", default="http://localhost:8000", help="Service URL"
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save results to tests/planner/e2e_scaling_results instead of /tmp",
    )
    # No additional arguments needed - test is hardcoded

    args = parser.parse_args()

    test = ScalingE2ETest(
        namespace=args.namespace, base_url=args.base_url, save_results=args.save_results
    )

    try:
        # Check that service is accessible
        logger.info(f"Checking service availability at {args.base_url}...")

        # Run the scaling test
        logger.info("Running scaling test...")
        results = await test.run_scaling_test()

        # Validate results
        validation = test.validate_test_results(results)

        # Save results
        timestamp = int(time.time())
        results_file = f"/tmp/scaling_test_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump({"results": results, "validation": validation}, f, indent=2)

        # Print summary
        logger.info("=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(validation["summary"])

        if validation["issues"]:
            logger.info("\nIssues found:")
            for issue in validation["issues"]:
                logger.info(f"  - {issue}")

        if any(k.endswith("_throughput") for k in validation.keys()):
            logger.info("\nPerformance:")
            if "baseline_throughput" in validation:
                logger.info(
                    f"  Baseline (8 req/s): {validation['baseline_throughput']}"
                )
            if "moderate_throughput" in validation:
                logger.info(
                    f"  Moderate (15 req/s): {validation['moderate_throughput']}"
                )
            if "trigger_throughput" in validation:
                logger.info(f"  Trigger (25 req/s): {validation['trigger_throughput']}")

        logger.info(f"\nDetailed results saved to: {results_file}")
        logger.info("=" * 60)

        return 0 if validation["test_passed"] else 1

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))
