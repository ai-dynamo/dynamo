# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Manual end-to-end scaling check for the SLA planner.

This script intentionally lives outside the automated test tree so it can be kept in the
planner image without being collected by pytest.
"""

import asyncio
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from dynamo.planner.tests.unit.load_generator import LoadGenerator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

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
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0, result.stdout.strip()
        except subprocess.TimeoutExpired:
            logger.error("kubectl command timed out: %s", " ".join(cmd))
            return False, ""
        except OSError as exc:
            logger.error("kubectl command failed: %s", exc)
            return False, ""

    def get_pod_counts(self) -> Optional[PodCounts]:
        cmd = [
            "kubectl",
            "get",
            "pods",
            "-n",
            self.namespace,
            "--selector",
            f"nvidia.com/dynamo-namespace={self.namespace}-{self.deployment_name}",
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
                sub_component = pod_labels.get(
                    "nvidia.com/dynamo-sub-component-type", ""
                )

                if pod_phase == "Running":
                    if sub_component == "prefill":
                        prefill_pods += 1
                    elif sub_component == "decode":
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
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse pod counts: %s", exc)
            return None

    async def monitor_scaling(
        self, duration: int, interval: int = 10
    ) -> List[PodCounts]:
        logger.info(
            "Monitoring pod scaling for %ss (interval: %ss)", duration, interval
        )

        start_time = time.time()
        monitoring_data = []

        while time.time() - start_time < duration:
            counts = self.get_pod_counts()
            if counts:
                monitoring_data.append(counts)
                logger.info("Pod counts: %s", counts)
            await asyncio.sleep(interval)

        return monitoring_data


class ScalingE2ETest:
    """Manual end-to-end scaling validation for the SLA planner."""

    def __init__(
        self,
        namespace: str = "default",
        base_url: str = "http://localhost:8000",
        save_results: bool = False,
        mode: str = "throughput",
    ):
        self.namespace = namespace
        self.base_url = base_url
        self.save_results = save_results
        self.mode = mode
        self.k8s_monitor = KubernetesMonitor(namespace)
        self.load_generator = LoadGenerator(
            base_url=base_url, save_results=save_results
        )

    async def run_scaling_test(self) -> Dict[str, Any]:
        logger.info("Starting manual scaling integration test (mode=%s)", self.mode)

        test_start_time = time.time()
        initial_counts = self.k8s_monitor.get_pod_counts()
        logger.info("Test starting with: %s", initial_counts)

        total_test_duration = (
            120 + 30 + 120 + BUFFER_DURATION
            if self.mode == "load"
            else 90 + 30 + 120 + BUFFER_DURATION
        )
        monitoring_task = asyncio.create_task(
            self.k8s_monitor.monitor_scaling(
                total_test_duration, interval=MONITORING_INTERVAL
            )
        )

        baseline_results: Dict[str, Any] = {}
        trigger_results: Dict[str, Any] = {}

        try:
            load_results = await self.load_generator.run_scaling_test(mode=self.mode)
            phase_results = load_results.get("phase_results", {})
            baseline_results = phase_results.get("phase1_baseline", {})
            trigger_results = phase_results.get("phase2_prefill_scaling_trigger", {})

            final_counts = self.k8s_monitor.get_pod_counts()
            logger.info("Final pod counts: %s", final_counts)

            logger.info("Waiting for potential delayed scaling...")
            await asyncio.sleep(FINAL_STABILIZATION_DELAY)

            final_final_counts = self.k8s_monitor.get_pod_counts()
            logger.info("Final final pod counts: %s", final_final_counts)
        finally:
            monitoring_task.cancel()
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass

        return {
            "test_duration": time.time() - test_start_time,
            "config": {
                "baseline_rps": 8.0,
                "trigger_rps": 18.0,
                "phase_durations": {"baseline": 90, "trigger": 120},
                "transition_delay": 30,
            },
            "initial_pod_counts": initial_counts.__dict__ if initial_counts else None,
            "baseline_results": baseline_results,
            "trigger_results": trigger_results,
            "final_pod_counts": final_counts.__dict__ if final_counts else None,
            "final_final_pod_counts": final_final_counts.__dict__
            if final_final_counts
            else None,
            "pod_history": [counts.__dict__ for counts in self.k8s_monitor.pod_history],
            "scaling_analysis": self.analyze_scaling_behavior(),
        }

    def analyze_scaling_behavior(self) -> Dict[str, Any]:
        if len(self.k8s_monitor.pod_history) < 2:
            return {"error": "Insufficient data for analysis"}

        history = self.k8s_monitor.pod_history
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

    def validate_test_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        validation: Dict[str, Any] = {"test_passed": False, "issues": [], "summary": ""}
        analysis = results.get("scaling_analysis")
        if not analysis:
            validation["issues"].append("No scaling analysis data")
            return validation

        expected = analysis.get("expected_scaling", {})
        if not expected.get("initial_1p1d"):
            validation["issues"].append("Test did not start with 1P1D configuration")
        if not expected.get("final_2p1d"):
            validation["issues"].append(
                "Test did not end with expected 2P1D configuration"
            )
        if not expected.get("scaling_occurred"):
            validation["issues"].append("No scaling events detected")

        if expected.get("correct_scaling"):
            validation["test_passed"] = True
            validation["summary"] = "PASS: Successfully scaled from 1P1D to 2P1D"
        else:
            validation[
                "summary"
            ] = "FAIL: Did not achieve expected 1P1D -> 2P1D scaling"

        baseline = results.get("baseline_results", {})
        trigger = results.get("trigger_results", {})
        if baseline.get("throughput", 0) > 0:
            validation["baseline_throughput"] = f"{baseline['throughput']:.2f} req/s"
        if trigger.get("throughput", 0) > 0:
            validation["trigger_throughput"] = f"{trigger['throughput']:.2f} req/s"
        return validation


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="SLA Planner Scaling E2E Test")
    parser.add_argument("--namespace", default="default", help="Kubernetes namespace")
    parser.add_argument(
        "--base-url", default="http://localhost:8000", help="Service URL"
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help=(
            "Save results to components/src/dynamo/planner/tests/e2e_scaling_results "
            "instead of /tmp"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["throughput", "load"],
        default="throughput",
        help="Scaling mode to test: throughput (default) or load",
    )

    args = parser.parse_args()
    test = ScalingE2ETest(
        namespace=args.namespace,
        base_url=args.base_url,
        save_results=args.save_results,
        mode=args.mode,
    )

    try:
        logger.info("Running scaling test...")
        results = await test.run_scaling_test()
        validation = test.validate_test_results(results)

        timestamp = int(time.time())
        results_file = f"/tmp/scaling_test_results_{timestamp}.json"
        with open(results_file, "w") as handle:
            json.dump({"results": results, "validation": validation}, handle, indent=2)

        logger.info("=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(validation["summary"])
        for issue in validation["issues"]:
            logger.info("Issue: %s", issue)
        logger.info("Detailed results saved to: %s", results_file)
        logger.info("=" * 60)
        return 0 if validation["test_passed"] else 1
    except Exception:
        logger.exception("Test failed unexpectedly")
        raise


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
