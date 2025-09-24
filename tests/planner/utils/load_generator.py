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
import tempfile
import time
from typing import Any, Dict, List, Optional

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
        isl: int = 4000,
        osl: int = 150,
        save_results: bool = False,
    ):
        self.base_url = base_url
        self.model = model
        self.isl = isl
        self.osl = osl
        self.save_results = save_results

    def _calculate_genai_perf_params(
        self,
        req_per_sec: float,
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
        concurrency = max(1, int(req_per_sec * 3))

        return {
            "concurrency": concurrency,
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
        params = self._calculate_genai_perf_params(req_per_sec)
        logger.info(f"Using request_rate={params['request_rate']} req/s")

        # Create artifact directory if not provided
        if artifact_dir is None:
            artifact_dir = tempfile.mkdtemp(prefix="scaling_test_")

        os.makedirs(artifact_dir, exist_ok=True)

        # Drive test length by caller-provided duration
        request_count = max(1, int(params["request_rate"] * duration_sec))

        logger.info(
            f"Adjusted parameters: duration={duration_sec}s, request_count={request_count}"
        )

        # Build genai-perf command based on coworker's successful approach
        cmd = [
            "genai-perf",
            "profile",
            "--model",
            self.model,
            "--tokenizer",
            self.model,
            "--endpoint-type",
            "chat",
            "--url",
            self.base_url.replace("http://", ""),
            "--streaming",
            "--synthetic-input-tokens-mean",
            str(self.isl),
            "--output-tokens-mean",
            str(self.osl),
            "--request-rate",
            str(params["request_rate"]),
            "--request-count",
            str(request_count),  # Use request count to limit test duration
            "--stability-percentage",
            "50",
            "--num-dataset-entries",
            str(
                max(20, int(params["request_rate"] * 10))
            ),  # Generate reasonable dataset size
            "--artifact-dir",
            artifact_dir,
            "--",
            "-v",
            "-max-threads",
            "64",
        ]

        logger.info(f"Running command: {' '.join(cmd)}")
        logger.info(
            f"Expected duration: {duration_sec}s, timeout: {max(duration_sec * 2 + 120, int(duration_sec * 2.5))}s"
        )

        # Run genai-perf (async)
        start_time = time.time()
        # More generous timeout for high-load tests - allow 2x duration + 2 minutes buffer
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
                logger.error("genai-perf timed out")
                raise RuntimeError("Load generation timed out")

            end_time = time.time()
            actual_duration = end_time - start_time

            # Persist logs for debugging
            try:
                with open(
                    os.path.join(artifact_dir, "genai_perf.stdout.log"), "wb"
                ) as f:
                    f.write(stdout or b"")
                with open(
                    os.path.join(artifact_dir, "genai_perf.stderr.log"), "wb"
                ) as f:
                    f.write(stderr or b"")
            except Exception:
                pass

            if proc.returncode == 0:
                logger.info("Load generation completed successfully")
                logger.info(f"Actual duration: {actual_duration:.2f}s")
                results = self._parse_genai_perf_results(artifact_dir)
                results.update(
                    {
                        "requested_req_per_sec": req_per_sec,
                        "actual_duration": actual_duration,
                        "target_duration": duration_sec,
                        "genai_perf_params": params,
                        "artifact_dir": artifact_dir,
                        "success": True,
                    }
                )
                return results
            else:
                logger.error(f"genai-perf failed with return code {proc.returncode}")
                raise RuntimeError("genai-perf failed; see logs in artifact dir")
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"genai-perf execution error: {e}")
            raise

    def _parse_genai_perf_results(self, artifact_dir: str) -> Dict[str, Any]:
        """Parse genai-perf results from artifact directory."""
        try:
            # Look for the profile_export_genai_perf.json file
            json_files = [f for f in os.listdir(artifact_dir) if f.endswith(".json")]
            if not json_files:
                logger.warning("No JSON results found in artifact directory")
                return {}

            # Main results file
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

            results = {}
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
        Run a graduated scaling test for prefill scaling.

        Uses a conservative graduated approach:
        - Phase 1: 8 req/s (baseline, should maintain 1P1D)
        - Phase 2: 18 req/s (should trigger prefill scaling to 2P1D)

        Returns:
            Dictionary with complete test results
        """
        logger.info(
            "Starting graduated prefill scaling test scenario (targeting 1P1D -> 2P1D)"
        )
        logger.info("Using conservative graduated approach with metric generation")

        # Graduated test parameters (optimized for prefill scaling)
        phases: List[Dict[str, Any]] = [
            {"rate": 8.0, "duration": 90, "name": "baseline"},
            {"rate": 18.0, "duration": 120, "name": "prefill_scaling_trigger"},
        ]
        transition_delay = 30

        # Create artifact directory
        timestamp = int(time.time())
        if self.save_results:
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            base_dir = os.path.join(
                script_dir, "e2e_scaling_results", f"scaling_test_{timestamp}"
            )
        else:
            base_dir = f"/tmp/scaling_test_{timestamp}"

        os.makedirs(base_dir, exist_ok=True)
        logger.info(f"Saving results to: {base_dir}")

        results = {
            "test_timestamp": timestamp,
            "config": {
                "approach": "graduated_scaling",
                "phases": phases,
                "transition_delay": transition_delay,
                "isl": self.isl,
                "osl": self.osl,
                "model": self.model,
            },
        }

        try:
            phase_results = {}

            for i, phase in enumerate(phases):
                phase_name = f"phase{i+1}_{phase['name']}"
                logger.info(
                    f"Starting {phase_name}: {phase['rate']} req/s for {phase['duration']}s"
                )

                phase_dir = os.path.join(base_dir, phase_name)
                phase_result = await self.generate_load(
                    req_per_sec=phase["rate"],
                    duration_sec=phase["duration"],
                    artifact_dir=phase_dir,
                )
                phase_results[phase_name] = phase_result

                # Add transition delay except after last phase
                if i < len(phases) - 1:
                    logger.info(f"Transition delay: {transition_delay}s")
                    await asyncio.sleep(transition_delay)

            results["phase_results"] = phase_results
            logger.info("Graduated scaling test completed successfully")

        except Exception as e:
            logger.error(f"Scaling test failed: {e}")
            results["error"] = str(e)
            raise

        # Save results
        results_file = os.path.join(base_dir, "scaling_test_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Test results saved to: {results_file}")
        return results


async def main():
    """Main function for scaling test execution."""
    parser = argparse.ArgumentParser(
        description="SLA Planner Graduated Scaling Test - Optimized for 2P1D prefill scaling"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Service URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--model",
        default="nvidia/Llama-3.1-8B-Instruct-FP8",
        help="Model name (default: nvidia/Llama-3.1-8B-Instruct-FP8)",
    )
    parser.add_argument(
        "--isl",
        type=int,
        default=4000,
        help="Input sequence length - optimized for prefill scaling (default: 4000)",
    )
    parser.add_argument(
        "--osl",
        type=int,
        default=150,
        help="Output sequence length - optimized for prefill scaling (default: 150)",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save results to tests/planner/e2e_scaling_results instead of /tmp",
    )

    args = parser.parse_args()

    generator = LoadGenerator(
        base_url=args.base_url,
        model=args.model,
        isl=args.isl,
        osl=args.osl,
        save_results=args.save_results,
    )

    print("Starting SLA Planner Graduated Scaling Test...")
    print(f"Parameters: ISL={args.isl}, OSL={args.osl}")
    print(
        "Test phases: 8 -> 15 -> 25 req/s (optimized for 1P1D -> 2P1D prefill scaling)"
    )

    results = await generator.run_scaling_test()

    print("\n" + "=" * 60)
    print("SCALING TEST COMPLETED")
    print("=" * 60)

    # Print results summary
    phase_results = results.get("phase_results", {})
    for phase_name, phase_data in phase_results.items():
        ok = isinstance(phase_data, dict) and phase_data.get(
            "success", bool(phase_data)
        )
        if ok:
            duration = phase_data.get("actual_duration")
            if isinstance(duration, (int, float)):
                print(f"{phase_name}: {duration:.1f}s duration - SUCCESS")
            else:
                print(f"{phase_name}: SUCCESS")
        else:
            print(f"{phase_name}: FAILED")
    print("\nResults saved to scaling test directory")


if __name__ == "__main__":
    asyncio.run(main())
