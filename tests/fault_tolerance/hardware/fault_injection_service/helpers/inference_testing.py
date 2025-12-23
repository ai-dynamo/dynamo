# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
"""
Inference load testing utilities for fault tolerance tests.

Provides continuous load generation and statistics tracking for
validating inference availability during fault injection scenarios.

Supports both local (port-forwarded) and in-cluster execution.
"""

import os
import threading
import time
from typing import Dict, List, Optional

import requests


def get_inference_endpoint(
    deployment_name: str, namespace: str, local_port: int = 8000
) -> str:
    """
    Get inference endpoint URL based on environment.

    Args:
        deployment_name: Name of the deployment
        namespace: Kubernetes namespace
        local_port: Port for local port-forwarding (default: 8000)

    Returns:
        Inference endpoint URL
    """
    in_cluster = os.getenv("KUBERNETES_SERVICE_HOST") is not None

    if in_cluster:
        # Use cluster-internal service DNS
        return (
            f"http://{deployment_name}.{namespace}.svc.cluster.local:80/v1/completions"
        )
    else:
        # Use port-forwarded localhost
        return f"http://localhost:{local_port}/v1/completions"


class InferenceLoadTester:
    """Continuous inference load generator for fault tolerance testing."""

    def __init__(self, endpoint: str, model_name: str, timeout: int = 30):
        """
        Initialize the inference load tester.

        Args:
            endpoint: Inference endpoint URL (e.g., "http://localhost:8000/v1/completions")
            model_name: Model name to use in requests
            timeout: Request timeout in seconds (default: 30)
        """
        self.endpoint = endpoint
        self.model_name = model_name
        self.timeout = timeout
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.results: List[Dict] = []
        self.lock = threading.Lock()
        self.checkpoint_index = 0  # Track checkpoint for per-phase stats

    def send_inference_request(self, prompt: str = "Hello, world!") -> Dict:
        """
        Send a single inference request and return result.

        Args:
            prompt: Text prompt for inference

        Returns:
            Dict with keys: success, status_code, latency, timestamp, error
        """
        try:
            start_time = time.time()
            response = requests.post(
                self.endpoint,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "max_tokens": 50,
                    "temperature": 0.7,
                },
                timeout=self.timeout,
            )
            latency = time.time() - start_time

            return {
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "latency": latency,
                "timestamp": time.time(),
                "error": None if response.status_code == 200 else response.text[:200],
            }
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "status_code": None,
                "latency": self.timeout,
                "timestamp": time.time(),
                "error": "Request timeout",
            }
        except Exception as e:
            return {
                "success": False,
                "status_code": None,
                "latency": time.time() - start_time if "start_time" in locals() else 0,
                "timestamp": time.time(),
                "error": str(e)[:200],
            }

    def _load_loop(self, interval: float = 2.0):
        """Background loop sending requests at specified interval."""
        while self.running:
            result = self.send_inference_request()
            with self.lock:
                self.results.append(result)
            time.sleep(interval)

    def start(self, interval: float = 2.0):
        """
        Start sending inference requests in background.

        Args:
            interval: Seconds between requests (default: 2.0)
        """
        if self.running:
            return

        self.running = True
        self.results = []
        self.thread = threading.Thread(
            target=self._load_loop, args=(interval,), daemon=True
        )
        self.thread.start()

    def stop(self) -> List[Dict]:
        """
        Stop sending requests and return results.

        Returns:
            List of all request results
        """
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)

        with self.lock:
            return self.results.copy()

    def checkpoint(self):
        """Mark current point for per-phase stats. Call before each test phase."""
        with self.lock:
            self.checkpoint_index = len(self.results)

    def get_stats(self, since_checkpoint: bool = False) -> Dict:
        """
        Get statistics for results including latency percentiles.

        Args:
            since_checkpoint: If True, only return stats since last checkpoint.
                            If False, return cumulative stats (default).

        Returns:
            Dict with keys: total, success, failed, success_rate,
                          avg_latency, p50_latency, p95_latency, p99_latency,
                          min_latency, max_latency, errors
        """
        with self.lock:
            # Get results based on whether we want per-phase or cumulative
            if since_checkpoint:
                results = self.results[self.checkpoint_index :]
            else:
                results = self.results

            if not results:
                return {
                    "total": 0,
                    "success": 0,
                    "failed": 0,
                    "success_rate": 0.0,
                    "avg_latency": 0.0,
                    "p50_latency": 0.0,
                    "p95_latency": 0.0,
                    "p99_latency": 0.0,
                    "min_latency": 0.0,
                    "max_latency": 0.0,
                    "errors": [],
                }

            total = len(results)
            success = sum(1 for r in results if r["success"])
            failed = total - success

            # Calculate latency stats for successful requests only
            success_latencies = sorted([r["latency"] for r in results if r["success"]])

            if success_latencies:
                avg_latency = sum(success_latencies) / len(success_latencies)
                min_latency = min(success_latencies)
                max_latency = max(success_latencies)

                # Calculate percentiles
                def percentile(data, p):
                    """Calculate percentile (0-100)"""
                    if not data:
                        return 0.0
                    k = (len(data) - 1) * (p / 100.0)
                    f = int(k)
                    c = f + 1 if (f + 1) < len(data) else f
                    if f == c:
                        return data[f]
                    return data[f] * (c - k) + data[c] * (k - f)

                p50 = percentile(success_latencies, 50)
                p95 = percentile(success_latencies, 95)
                p99 = percentile(success_latencies, 99)
            else:
                avg_latency = min_latency = max_latency = 0.0
                p50 = p95 = p99 = 0.0

            return {
                "total": total,
                "success": success,
                "failed": failed,
                "success_rate": (success / total) * 100,
                "avg_latency": avg_latency,
                "p50_latency": p50,
                "p95_latency": p95,
                "p99_latency": p99,
                "min_latency": min_latency,
                "max_latency": max_latency,
                "errors": [r["error"] for r in results if r["error"]][:5],
            }
