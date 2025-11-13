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

    def get_stats(self) -> Dict:
        """
        Get statistics for current results.

        Returns:
            Dict with keys: total, success, failed, success_rate, avg_latency, errors
        """
        with self.lock:
            if not self.results:
                return {
                    "total": 0,
                    "success": 0,
                    "failed": 0,
                    "success_rate": 0.0,
                    "avg_latency": 0.0,
                    "errors": [],
                }

            total = len(self.results)
            success = sum(1 for r in self.results if r["success"])
            failed = total - success
            avg_latency = sum(r["latency"] for r in self.results if r["success"]) / max(
                success, 1
            )

            return {
                "total": total,
                "success": success,
                "failed": failed,
                "success_rate": (success / total) * 100,
                "avg_latency": avg_latency,
                "errors": [r["error"] for r in self.results if r["error"]][:5],
            }
