# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Load generation script for SLA planner scaling tests.

This script uses genai-perf to generate load at specific request rates
to test the planner's scaling behavior.
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
from typing import Any, Dict, Optional

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LoadGenerator:
    """Generate load using genai-perf to test planner scaling."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = "nvidia/Llama-3.1-8B-Instruct-FP8",
        isl: int = 3000,
        osl: int = 150,
    ):
        self.base_url = base_url
        self.model = model
        self.isl = isl
        self.osl = osl

    def _calculate_genai_perf_params(
        self,
        req_per_sec: float,
        duration_sec: int,
        estimated_request_duration: float = 3.0,
    ) -> Dict[str, Any]:
        """
        Calculate genai-perf parameters to approximate desired request rate.

        Args:
            req_per_sec: Desired requests per second
            duration_sec: Test duration in seconds
            estimated_request_duration: Estimated average request duration in seconds

        Returns:
            Dictionary with concurrency and request_rate parameters
        """
        # Use request rate for planner testing as suggested
        return {
            "request_rate": req_per_sec,
        }

    async def generate_load(
        self, req_per_sec: float, duration_sec: int, artifact_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate load at specified request rate for given duration.

        Args:
            req_per_sec: Target requests per second
            duration_sec: Duration to generate load (seconds)
            artifact_dir: Directory to store genai-perf artifacts

        Returns:
            Dictionary with load test results
        """
        logger.info(f"Generating load: {req_per_sec} req/s for {duration_sec}s")

        # Calculate genai-perf parameters
        params = self._calculate_genai_perf_params(req_per_sec, duration_sec)
        logger.info(
            f"Using request_rate={params['request_rate']} req/s for {duration_sec}s"
        )

        # Create artifact directory if not provided
        if artifact_dir is None:
            # Store artifacts in tests/planner/artifacts for easier access
            base_artifacts_dir = "/home/hannahz/dev/ai-dynamo/tests/planner/artifacts"
            os.makedirs(base_artifacts_dir, exist_ok=True)
            artifact_dir = tempfile.mkdtemp(
                prefix="scaling_test_", dir=base_artifacts_dir
            )

        os.makedirs(artifact_dir, exist_ok=True)

        # Build genai-perf command (with streaming for planner, but no problematic extra-inputs)
        cmd = [
            "genai-perf",
            "profile",
            "--model",
            self.model,
            "--endpoint-type",
            "chat",
            "--endpoint",
            "/v1/chat/completions",
            "--streaming",
            "--url",
            self.base_url,
            "--synthetic-input-tokens-mean",
            str(self.isl),
            "--output-tokens-mean",
            str(self.osl),
            "--request-rate",
            str(params["request_rate"]),
            "--measurement-interval",
            str(min(30000, duration_sec * 1000)),  # Cap at 30s to avoid timeouts
            "--num-dataset-entries",
            str(
                max(100, int(req_per_sec * min(30, duration_sec)))
            ),  # Match measurement interval
            "--artifact-dir",
            artifact_dir,
        ]

        logger.info(f"Running command: {' '.join(cmd)}")

        # Run genai-perf
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=duration_sec
                + 60,  # Add reasonable buffer for genai-perf overhead
            )

            end_time = time.time()
            actual_duration = end_time - start_time

            if result.returncode == 0:
                logger.info("Load generation completed successfully")
                logger.info(f"Actual duration: {actual_duration:.2f}s")

                # Parse results
                results = self._parse_genai_perf_results(artifact_dir)
                results.update(
                    {
                        "requested_req_per_sec": req_per_sec,
                        "actual_duration": actual_duration,
                        "target_duration": duration_sec,
                        "genai_perf_params": params,
                    }
                )

                return results
            else:
                logger.error(f"genai-perf failed with return code {result.returncode}")
                logger.error(f"stdout: {result.stdout}")
                logger.error(f"stderr: {result.stderr}")
                raise RuntimeError(f"genai-perf failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            logger.error("genai-perf timed out")
            raise RuntimeError("Load generation timed out")

    def _parse_genai_perf_results(self, artifact_dir: str) -> Dict[str, Any]:
        """Parse genai-perf results from artifact directory."""
        try:
            # Look for the profile_export_genai_perf.json file
            json_files = [f for f in os.listdir(artifact_dir) if f.endswith(".json")]
            if not json_files:
                logger.warning("No JSON results found in artifact directory")
                return {}

            # Try to find the main results file
            results_file = None
            for json_file in json_files:
                if "profile_export" in json_file or "genai_perf" in json_file:
                    results_file = os.path.join(artifact_dir, json_file)
                    break

            if not results_file:
                results_file = os.path.join(artifact_dir, json_files[0])

            logger.info(f"Parsing results from: {results_file}")

            with open(results_file, "r") as f:
                data = json.load(f)

            # Extract key metrics
            results = {}

            # Try to extract metrics from different possible formats
            if "experiments" in data and data["experiments"]:
                exp = data["experiments"][0]
                if "perf_metrics" in exp:
                    metrics = exp["perf_metrics"]
                    results.update(
                        {
                            "throughput": metrics.get("throughput", {}).get("avg", 0),
                            "ttft_mean": metrics.get("ttft", {}).get("avg", 0),
                            "itl_mean": metrics.get("inter_token_latency", {}).get(
                                "avg", 0
                            ),
                            "end_to_end_latency_mean": metrics.get(
                                "request_latency", {}
                            ).get("avg", 0),
                        }
                    )

            # If we couldn't find metrics in experiments, try other formats
            if not results and "profile_export_genai_perf" in data:
                summary = data.get("summary", {})
                results.update(
                    {
                        "throughput": summary.get("throughput", 0),
                        "ttft_mean": summary.get("time_to_first_token_ms", 0),
                        "itl_mean": summary.get("inter_token_latency_ms", 0),
                    }
                )

            logger.info(f"Parsed results: {results}")
            return results

        except Exception as e:
            logger.warning(f"Failed to parse genai-perf results: {e}")
            return {}

    async def run_scaling_test(self) -> Dict[str, Any]:
        """
        Run the complete scaling test scenario.

        Hardcoded scenario:
        - Phase 1: 10 req/s for 180s (should maintain 1P1D)
        - Phase 2: 20 req/s for 180s (should trigger 1P1D -> 2P1D)

        Returns:
            Dictionary with complete test results
        """
        # Hardcoded test parameters
        phase1_req_per_sec = 10.0
        phase2_req_per_sec = 20.0
        phase_duration = 180
        transition_delay = 30

        logger.info("Starting scaling test scenario")
        logger.info(f"Phase 1: {phase1_req_per_sec} req/s for {phase_duration}s")
        logger.info(f"Transition delay: {transition_delay}s")
        logger.info(f"Phase 2: {phase2_req_per_sec} req/s for {phase_duration}s")

        # Create directories for artifacts
        timestamp = int(time.time())
        base_dir = f"/tmp/scaling_test_{timestamp}"
        phase1_dir = os.path.join(base_dir, "phase1")
        phase2_dir = os.path.join(base_dir, "phase2")

        os.makedirs(phase1_dir, exist_ok=True)
        os.makedirs(phase2_dir, exist_ok=True)

        results = {
            "test_timestamp": timestamp,
            "config": {
                "phase1_req_per_sec": phase1_req_per_sec,
                "phase2_req_per_sec": phase2_req_per_sec,
                "phase_duration": phase_duration,
                "transition_delay": transition_delay,
                "isl": self.isl,
                "osl": self.osl,
                "model": self.model,
            },
        }

        try:
            # Phase 1: Lower load
            logger.info("Starting Phase 1 - Lower load")
            phase1_results = await self.generate_load(
                req_per_sec=phase1_req_per_sec,
                duration_sec=phase_duration,
                artifact_dir=phase1_dir,
            )
            results["phase1"] = phase1_results

            # Transition delay
            logger.info(f"Waiting {transition_delay}s for metrics to stabilize...")
            await asyncio.sleep(transition_delay)

            # Phase 2: Higher load
            logger.info("Starting Phase 2 - Higher load")
            phase2_results = await self.generate_load(
                req_per_sec=phase2_req_per_sec,
                duration_sec=phase_duration,
                artifact_dir=phase2_dir,
            )
            results["phase2"] = phase2_results

            logger.info("Scaling test completed successfully")

        except Exception as e:
            logger.error(f"Scaling test failed: {e}")
            results["error"] = str(e)
            raise

        # Save combined results
        results_file = os.path.join(base_dir, "scaling_test_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Test results saved to: {results_file}")
        return results


async def main():
    """Main function for standalone testing."""
    parser = argparse.ArgumentParser(
        description="Load generator for SLA planner scaling tests"
    )
    parser.add_argument(
        "--base-url", default="http://localhost:8000", help="Base URL for the service"
    )
    parser.add_argument(
        "--model", default="nvidia/Llama-3.1-8B-Instruct-FP8", help="Model name"
    )
    parser.add_argument("--isl", type=int, default=3000, help="Input sequence length")
    parser.add_argument("--osl", type=int, default=150, help="Output sequence length")
    parser.add_argument(
        "--req-per-sec", type=float, default=10.0, help="Requests per second"
    )
    parser.add_argument(
        "--duration", type=int, default=120, help="Test duration in seconds"
    )
    parser.add_argument(
        "--scaling-test", action="store_true", help="Run the full scaling test scenario"
    )
    parser.add_argument(
        "--phase1-rps", type=float, default=10.0, help="Phase 1 requests per second"
    )
    parser.add_argument(
        "--phase2-rps", type=float, default=20.0, help="Phase 2 requests per second"
    )
    parser.add_argument(
        "--phase-duration", type=int, default=180, help="Duration of each phase"
    )

    args = parser.parse_args()

    generator = LoadGenerator(
        base_url=args.base_url, model=args.model, isl=args.isl, osl=args.osl
    )

    if args.scaling_test:
        results = await generator.run_scaling_test()
        print("Scaling test completed!")
        print(
            f"Phase 1 throughput: {results.get('phase1', {}).get('throughput', 'N/A')} req/s"
        )
        print(
            f"Phase 2 throughput: {results.get('phase2', {}).get('throughput', 'N/A')} req/s"
        )
    else:
        results = await generator.generate_load(
            req_per_sec=args.req_per_sec, duration_sec=args.duration
        )
        print("Load generation completed!")
        print(f"Achieved throughput: {results.get('throughput', 'N/A')} req/s")
        print(f"TTFT: {results.get('ttft_mean', 'N/A')} ms")
        print(f"ITL: {results.get('itl_mean', 'N/A')} ms")


if __name__ == "__main__":
    asyncio.run(main())
