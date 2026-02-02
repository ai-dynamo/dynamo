#!/usr/bin/env python3
"""
EPD Benchmark Script
Sends concurrent requests to measure TTFT (Time To First Token) performance.
"""

import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass
from typing import List

import aiohttp

IMAGE_URLS = [
    "https://huggingface.co/datasets/Sayali9141/traffic_signal_images/resolve/main/61.jpg",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png",
]


@dataclass
class RequestResult:
    request_id: int
    image_url: str
    ttft: float  # Time to first token (seconds)
    total_time: float  # Total request time (seconds)
    success: bool
    error: str = ""
    response_text: str = ""  # Captured response for quality check


def build_request_payload(
    image_url: str, model: str, max_tokens: int, stream: bool
) -> dict:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe image in detail."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "stream": stream,
    }


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    request_id: int,
    image_url: str,
) -> RequestResult:
    """Send a single request and measure TTFT."""
    start_time = time.perf_counter()
    ttft = None
    response_text = ""

    try:
        if payload.get("stream", False):
            # Streaming mode - measure time to first chunk
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return RequestResult(
                        request_id=request_id,
                        image_url=image_url,
                        ttft=0,
                        total_time=time.perf_counter() - start_time,
                        success=False,
                        error=f"HTTP {response.status}: {error_text[:200]}",
                    )

                chunks = []
                async for line in response.content:
                    if ttft is None:
                        ttft = time.perf_counter() - start_time
                    # Continue reading to complete the request
                    decoded = line.decode("utf-8").strip()
                    if decoded.startswith("data: [DONE]"):
                        break
                    # Parse SSE data to extract content
                    if decoded.startswith("data: "):
                        try:
                            data = json.loads(decoded[6:])
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    chunks.append(content)
                        except json.JSONDecodeError:
                            pass

                response_text = "".join(chunks)
                total_time = time.perf_counter() - start_time
                return RequestResult(
                    request_id=request_id,
                    image_url=image_url,
                    ttft=ttft if ttft else total_time,
                    total_time=total_time,
                    success=True,
                    response_text=response_text,
                )
        else:
            # Non-streaming mode - TTFT is essentially total time
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return RequestResult(
                        request_id=request_id,
                        image_url=image_url,
                        ttft=0,
                        total_time=time.perf_counter() - start_time,
                        success=False,
                        error=f"HTTP {response.status}: {error_text[:200]}",
                    )

                data = await response.json()
                # Extract response text from non-streaming response
                if "choices" in data and len(data["choices"]) > 0:
                    response_text = (
                        data["choices"][0].get("message", {}).get("content", "")
                    )

                total_time = time.perf_counter() - start_time
                return RequestResult(
                    request_id=request_id,
                    image_url=image_url,
                    ttft=total_time,  # For non-streaming, TTFT = total time
                    total_time=total_time,
                    success=True,
                    response_text=response_text,
                )

    except asyncio.TimeoutError:
        return RequestResult(
            request_id=request_id,
            image_url=image_url,
            ttft=0,
            total_time=time.perf_counter() - start_time,
            success=False,
            error="Request timed out",
        )
    except Exception as e:
        return RequestResult(
            request_id=request_id,
            image_url=image_url,
            ttft=0,
            total_time=time.perf_counter() - start_time,
            success=False,
            error=str(e),
        )


async def run_benchmark(
    endpoint: str,
    model: str,
    concurrency: int,
    num_requests: int,
    max_tokens: int,
    stream: bool,
    timeout: int,
) -> List[RequestResult]:
    """Run the benchmark with specified concurrency."""

    connector = aiohttp.TCPConnector(limit=concurrency)
    timeout_config = aiohttp.ClientTimeout(total=timeout)

    async with aiohttp.ClientSession(
        connector=connector, timeout=timeout_config
    ) as session:
        tasks = []
        for i in range(num_requests):
            # Rotate through image URLs
            image_url = IMAGE_URLS[i % len(IMAGE_URLS)]
            payload = build_request_payload(image_url, model, max_tokens, stream)
            task = send_request(session, endpoint, payload, i, image_url)
            tasks.append(task)

        # Run with concurrency limit using semaphore
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_request(task_coro):
            async with semaphore:
                return await task_coro

        bounded_tasks = [bounded_request(task) for task in tasks]
        results = await asyncio.gather(*bounded_tasks)

    return results


def calculate_percentile(data: List[float], percentile: float) -> float:
    """Calculate the given percentile of a list of values."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = (len(sorted_data) - 1) * percentile / 100
    lower = int(index)
    upper = lower + 1
    if upper >= len(sorted_data):
        return sorted_data[-1]
    weight = index - lower
    return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight


def print_sample_responses(results: List[RequestResult], num_samples: int = 3):
    """Print random sample responses for quality verification."""
    successful = [r for r in results if r.success and r.response_text]

    if not successful:
        print("\nNo successful responses with text to display.")
        return

    # Randomly select samples (or all if fewer than num_samples)
    samples = random.sample(successful, min(num_samples, len(successful)))

    print("\n" + "=" * 60)
    print(f"SAMPLE RESPONSES ({len(samples)} random samples)")
    print("=" * 60)

    for i, result in enumerate(samples, 1):
        image_name = result.image_url.split("/")[-1]
        print(f"\n--- Sample {i} (Request #{result.request_id}) ---")
        print(f"Image: {image_name}")
        print(
            f"TTFT: {result.ttft * 1000:.0f}ms | Total: {result.total_time * 1000:.0f}ms"
        )
        print(f"Response ({len(result.response_text)} chars):")
        # Truncate long responses for readability
        response_preview = result.response_text[:500]
        if len(result.response_text) > 500:
            response_preview += "... [truncated]"
        print(f"  {response_preview}")

    print("\n" + "-" * 60)


def print_results(results: List[RequestResult], stream: bool, total_elapsed: float):
    """Print benchmark results with statistics."""
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    print(f"\nTotal Requests:     {len(results)}")
    print(f"Successful:         {len(successful)}")
    print(f"Failed:             {len(failed)}")

    if failed:
        print("\nFailure Summary:")
        error_counts = {}
        for r in failed:
            error_counts[r.error] = error_counts.get(r.error, 0) + 1
        for error, count in error_counts.items():
            print(f"  - {error}: {count}")

    if successful:
        ttft_values = [r.ttft * 1000 for r in successful]  # Convert to ms
        total_times = [r.total_time * 1000 for r in successful]  # Convert to ms

        metric_name = "TTFT" if stream else "Latency"

        print(f"\n{metric_name} Statistics (ms):")
        print("-" * 40)
        print(f"  Min:              {min(ttft_values):>10.2f}")
        print(f"  Max:              {max(ttft_values):>10.2f}")
        print(f"  Mean:             {statistics.mean(ttft_values):>10.2f}")
        print(f"  Median (p50):     {calculate_percentile(ttft_values, 50):>10.2f}")
        print(f"  p90:              {calculate_percentile(ttft_values, 90):>10.2f}")
        print(f"  p95:              {calculate_percentile(ttft_values, 95):>10.2f}")
        print(f"  p99:              {calculate_percentile(ttft_values, 99):>10.2f}")

        if stream:
            print("\nTotal Response Time Statistics (ms):")
            print("-" * 40)
            print(f"  Min:              {min(total_times):>10.2f}")
            print(f"  Max:              {max(total_times):>10.2f}")
            print(f"  Mean:             {statistics.mean(total_times):>10.2f}")
            print(f"  p95:              {calculate_percentile(total_times, 95):>10.2f}")

        if len(successful) > 1:
            print(f"\n  Std Dev:          {statistics.stdev(ttft_values):>10.2f}")

        # Throughput (using actual wall-clock time)
        throughput = len(successful) / total_elapsed if total_elapsed > 0 else 0
        print(f"\nThroughput:         {throughput:>10.2f} req/s")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="EPD Benchmark - Measure TTFT for concurrent requests",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--endpoint",
        "-e",
        default="http://localhost:8000/v1/chat/completions",
        help="API endpoint URL",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="llava-v1.6-mistral-7b-hf",
        help="Model name to use",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=4,
        help="Number of concurrent requests",
    )
    parser.add_argument(
        "--num-requests",
        "-n",
        type=int,
        default=10,
        help="Total number of requests to send",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        default=True,
        help="Use streaming mode (default: True)",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming mode",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=300,
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--show-responses",
        "-r",
        type=int,
        default=3,
        help="Number of random sample responses to display (0 to disable)",
    )

    args = parser.parse_args()

    # Handle stream flag
    stream = args.stream and not args.no_stream

    print("=" * 60)
    print("EPD BENCHMARK")
    print("=" * 60)
    print(f"Endpoint:           {args.endpoint}")
    print(f"Model:              {args.model}")
    print(f"Concurrency:        {args.concurrency}")
    print(f"Total Requests:     {args.num_requests}")
    print(f"Max Tokens:         {args.max_tokens}")
    print(f"Streaming:          {stream}")
    print(f"Timeout:            {args.timeout}s")
    print(f"Image URLs:         {len(IMAGE_URLS)} (rotating)")
    print("=" * 60)
    print("\nRunning benchmark...")

    start = time.perf_counter()
    results = asyncio.run(
        run_benchmark(
            endpoint=args.endpoint,
            model=args.model,
            concurrency=args.concurrency,
            num_requests=args.num_requests,
            max_tokens=args.max_tokens,
            stream=stream,
            timeout=args.timeout,
        )
    )
    elapsed = time.perf_counter() - start

    print(f"Benchmark completed in {elapsed:.2f}s")
    print_results(results, stream, elapsed)

    # Show sample responses for quality verification
    if args.show_responses > 0:
        print_sample_responses(results, args.show_responses)


if __name__ == "__main__":
    main()
