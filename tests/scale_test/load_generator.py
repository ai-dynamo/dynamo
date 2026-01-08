# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Basic load generation for scale testing.

This module provides async load generation capabilities to send requests
to multiple frontend endpoints and collect metrics.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


@dataclass
class RequestResult:
    """Result of a single request."""

    url: str
    latency: float
    success: bool
    error: Optional[str] = None
    tokens_generated: int = 0


@dataclass
class LoadGeneratorStats:
    """Statistics for a load generation run."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    latencies: List[float] = field(default_factory=list)

    @property
    def avg_latency(self) -> float:
        """Average latency in milliseconds."""
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies) * 1000

    @property
    def min_latency(self) -> float:
        """Minimum latency in milliseconds."""
        if not self.latencies:
            return 0.0
        return min(self.latencies) * 1000

    @property
    def max_latency(self) -> float:
        """Maximum latency in milliseconds."""
        if not self.latencies:
            return 0.0
        return max(self.latencies) * 1000

    @property
    def p50_latency(self) -> float:
        """50th percentile latency in milliseconds."""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = len(sorted_latencies) // 2
        return sorted_latencies[idx] * 1000

    @property
    def p99_latency(self) -> float:
        """99th percentile latency in milliseconds."""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)] * 1000

    @property
    def error_rate(self) -> float:
        """Error rate as a percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.failed_requests / self.total_requests) * 100


class LoadGenerator:
    """
    Generate load across multiple frontend endpoints.

    Uses AsyncOpenAI to send chat completion requests to frontends and
    tracks latency, throughput, and errors.
    """

    def __init__(
        self,
        frontend_urls: List[str],
        model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_tokens: int = 30,
        timeout: float = 60.0,
    ):
        """
        Initialize the load generator.

        Args:
            frontend_urls: List of frontend base URLs (e.g., ['http://localhost:8001'])
            model: Model name to use in requests
            max_tokens: Maximum tokens to generate per request
            timeout: Request timeout in seconds
        """
        self.frontend_urls = frontend_urls
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout

        # Per-URL statistics
        self.results: Dict[str, LoadGeneratorStats] = {
            url: LoadGeneratorStats() for url in frontend_urls
        }

        # Create OpenAI clients for each frontend
        self.clients: Dict[str, AsyncOpenAI] = {}
        for url in frontend_urls:
            self.clients[url] = AsyncOpenAI(
                base_url=f"{url}/v1",
                api_key="not-needed",
                timeout=timeout,
            )

    async def send_request(self, url: str) -> RequestResult:
        """
        Send a single chat completion request.

        Args:
            url: The frontend URL to send the request to

        Returns:
            RequestResult with latency and success status
        """
        client = self.clients[url]
        start_time = time.time()

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": "Hello, how are you today? Please respond briefly.",
                    }
                ],
                max_tokens=self.max_tokens,
                stream=False,
            )

            latency = time.time() - start_time
            tokens = response.usage.completion_tokens if response.usage else 0

            return RequestResult(
                url=url,
                latency=latency,
                success=True,
                tokens_generated=tokens,
            )

        except Exception as e:
            latency = time.time() - start_time
            return RequestResult(
                url=url,
                latency=latency,
                success=False,
                error=str(e),
            )

    def _record_result(self, result: RequestResult) -> None:
        """Record a request result in the statistics."""
        stats = self.results[result.url]
        stats.total_requests += 1

        if result.success:
            stats.successful_requests += 1
            stats.latencies.append(result.latency)
            stats.total_tokens += result.tokens_generated
        else:
            stats.failed_requests += 1
            logger.warning(f"Request to {result.url} failed: {result.error}")

    async def generate_load(
        self,
        duration_sec: int,
        qps: float,
    ) -> Dict[str, LoadGeneratorStats]:
        """
        Generate load across all URLs for the specified duration.

        Distributes requests evenly across all frontends using round-robin.
        Uses absolute time scheduling to maintain accurate QPS without drift.

        Args:
            duration_sec: Duration to generate load in seconds
            qps: Queries per second (total across all frontends)

        Returns:
            Dictionary mapping URL to LoadGeneratorStats
        """
        logger.info(
            f"Generating load for {duration_sec}s at {qps} QPS across "
            f"{len(self.frontend_urls)} frontends..."
        )

        interval = 1.0 / qps if qps > 0 else 1.0
        start_time = time.time()
        request_count = 0
        pending_tasks: List[asyncio.Task] = []

        while time.time() - start_time < duration_sec:
            # Round-robin across frontends
            url = self.frontend_urls[request_count % len(self.frontend_urls)]

            # Create task for this request
            task = asyncio.create_task(self._send_and_record(url))
            pending_tasks.append(task)
            request_count += 1

            # Calculate when the NEXT request should be sent (absolute time scheduling)
            # This prevents timing drift by accounting for loop overhead
            next_request_time = start_time + (request_count * interval)
            sleep_duration = next_request_time - time.time()

            # Only sleep if we're ahead of schedule
            if sleep_duration > 0:
                await asyncio.sleep(sleep_duration)
            # If sleep_duration <= 0, we're behind schedule, so send next request immediately

            # Clean up completed tasks periodically
            if len(pending_tasks) > 100:
                done = [t for t in pending_tasks if t.done()]
                for t in done:
                    pending_tasks.remove(t)

        # Wait for all remaining requests to complete
        if pending_tasks:
            logger.info(f"Waiting for {len(pending_tasks)} pending requests...")
            await asyncio.gather(*pending_tasks, return_exceptions=True)

        actual_duration = time.time() - start_time
        actual_qps = request_count / actual_duration if actual_duration > 0 else 0
        logger.info(
            f"Load generation complete. Sent {request_count} requests in "
            f"{actual_duration:.1f}s (actual QPS: {actual_qps:.1f})"
        )

        return self.results

    async def _send_and_record(self, url: str) -> None:
        """Send a request and record the result."""
        result = await self.send_request(url)
        self._record_result(result)

    def print_summary(self) -> None:
        """Print a summary of the load generation results."""
        print("\n" + "=" * 70)
        print("LOAD GENERATION RESULTS")
        print("=" * 70)

        total_requests = 0
        total_errors = 0

        for url, stats in self.results.items():
            total_requests += stats.total_requests
            total_errors += stats.failed_requests

            if stats.total_requests > 0:
                print(f"\nFrontend {url}:")
                print(f"  Requests: {stats.total_requests}")
                print(f"  Successful: {stats.successful_requests}")
                print(f"  Errors: {stats.failed_requests} ({stats.error_rate:.1f}%)")
                print(f"  Avg latency: {stats.avg_latency:.1f}ms")
                print(f"  Min latency: {stats.min_latency:.1f}ms")
                print(f"  Max latency: {stats.max_latency:.1f}ms")
                print(f"  P50 latency: {stats.p50_latency:.1f}ms")
                print(f"  P99 latency: {stats.p99_latency:.1f}ms")
                print(f"  Total tokens: {stats.total_tokens}")

        print("\n" + "-" * 70)
        print(f"Total requests: {total_requests}")
        print(f"Total errors: {total_errors}")
        if total_requests > 0:
            print(f"Overall error rate: {(total_errors / total_requests) * 100:.1f}%")
        print("=" * 70 + "\n")

    def reset_stats(self) -> None:
        """Reset all statistics for a fresh run."""
        self.results = {url: LoadGeneratorStats() for url in self.frontend_urls}


async def run_load_test(
    frontend_urls: List[str],
    model: str,
    duration_sec: int,
    qps: float,
    max_tokens: int = 30,
) -> Dict[str, LoadGeneratorStats]:
    """
    Convenience function to run a load test.

    Args:
        frontend_urls: List of frontend URLs to test
        model: Model name for requests
        duration_sec: Test duration in seconds
        qps: Queries per second
        max_tokens: Maximum tokens per request

    Returns:
        Dictionary of results per frontend
    """
    generator = LoadGenerator(
        frontend_urls=frontend_urls,
        model=model,
        max_tokens=max_tokens,
    )

    results = await generator.generate_load(duration_sec=duration_sec, qps=qps)
    generator.print_summary()

    return results
