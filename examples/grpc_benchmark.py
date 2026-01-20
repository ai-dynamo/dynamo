#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
gRPC Benchmark for Dynamo KServe server with STREAMING support.
Measures TTFT (Time To First Token) and TPOT (Time Per Output Token).

Similar to llmperf but for KServe gRPC protocol.

Usage:
    python benchmarks/grpc_benchmark.py --model Qwen/Qwen3-0.6B --concurrency 64 128
    python benchmarks/grpc_benchmark.py --model Qwen/Qwen3-0.6B --concurrency 1 8 16 32 64 --requests 50
"""

import argparse
import json
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from statistics import mean, median, stdev
from typing import List, Optional

import numpy as np


@dataclass
class RequestMetrics:
    """Metrics for a single request."""

    ttft_s: float  # Time to first token (seconds)
    end_to_end_latency_s: float  # Total request time (seconds)
    num_output_tokens: int  # Number of output tokens
    inter_token_latency_s: float  # Average time per output token (seconds) - similar to TPOT
    success: bool = True
    error: Optional[str] = None


def run_streaming_request(
    url: str, model: str, prompt: str, timeout: int = 300, verbose: bool = False
) -> RequestMetrics:
    """Run a single STREAMING inference request and measure latency."""
    import tritonclient.grpc as triton_grpc

    try:
        client = triton_grpc.InferenceServerClient(url=url, verbose=verbose)

        # Build text input
        text_input = triton_grpc.InferInput("text_input", [1], "BYTES")
        text_input.set_data_from_numpy(np.array([prompt.encode("utf-8")], dtype=object))

        # CRITICAL: Add "stream" input tensor to enable server-side streaming
        # The KServe server expects a separate BOOL tensor named "stream" or "streaming"
        stream_input = triton_grpc.InferInput("stream", [1], "BOOL")
        stream_input.set_data_from_numpy(np.array([True], dtype=bool))

        # For streaming, we use a callback-based approach
        result_queue = queue.Queue()
        first_token_time = None
        start_time = time.perf_counter()
        token_times = []

        def callback(result, error):
            nonlocal first_token_time
            now = time.perf_counter()
            if error:
                result_queue.put(("error", str(error)))
            else:
                if first_token_time is None:
                    first_token_time = now
                token_times.append(now)
                result_queue.put(("token", result))

        # Start streaming inference with both text_input and stream inputs
        client.start_stream(callback=callback)
        client.async_stream_infer(
            model_name=model,
            inputs=[text_input, stream_input],  # Include stream=True input
            request_id=str(time.time_ns()),
        )

        # Wait for completion with timeout
        num_tokens = 0
        deadline = time.time() + timeout

        while True:
            try:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError("Request timed out")

                msg_type, data = result_queue.get(timeout=min(remaining, 1.0))

                if msg_type == "error":
                    if "EOF" in str(data) or "cancelled" in str(data).lower():
                        # Stream ended normally
                        break
                    raise RuntimeError(data)
                elif msg_type == "token":
                    num_tokens += 1
                    # Check if this is the final response
                    try:
                        output = data.as_numpy("text_output")
                        if output is not None:
                            # Got final output, stream is done
                            break
                    except Exception:
                        pass

            except queue.Empty:
                # Check if stream is still alive
                continue

        client.stop_stream()
        end_time = time.perf_counter()

        # Calculate metrics
        total_time = end_time - start_time
        ttft = (first_token_time - start_time) if first_token_time else total_time

        # TPOT (Time Per Output Token) = (E2E - TTFT) / (output_tokens - 1)
        # This is the standard definition used by llmperf and other tools
        if num_tokens > 1:
            # Standard TPOT calculation from actual streamed tokens
            tpot = (total_time - ttft) / (num_tokens - 1)
        else:
            # Server sent single response - NOT truly streaming
            # TPOT cannot be measured, return -1
            print(
                f"  [WARNING] KServe server is NOT truly streaming (received {num_tokens} callback). TPOT cannot be measured."
            )
            tpot = -1.0

        # Use actual inter-token times if we have streaming data
        if len(token_times) > 1:
            inter_token_times = [
                token_times[i] - token_times[i - 1] for i in range(1, len(token_times))
            ]
            avg_inter_token = mean(inter_token_times) if inter_token_times else tpot
        else:
            avg_inter_token = tpot  # Will be -1 if not truly streaming

        return RequestMetrics(
            ttft_s=ttft,
            end_to_end_latency_s=total_time,
            num_output_tokens=max(num_tokens, 1),
            inter_token_latency_s=avg_inter_token,
            success=True,
        )

    except Exception as e:
        return RequestMetrics(
            ttft_s=0,
            end_to_end_latency_s=0,
            num_output_tokens=0,
            inter_token_latency_s=0,
            success=False,
            error=str(e),
        )


def run_nonstreaming_request(
    url: str, model: str, prompt: str, timeout: int = 300
) -> RequestMetrics:
    """Run a single NON-STREAMING inference request (fallback)."""
    import tritonclient.grpc as triton_grpc

    try:
        client = triton_grpc.InferenceServerClient(url=url, verbose=False)

        text_input = triton_grpc.InferInput("text_input", [1], "BYTES")
        text_input.set_data_from_numpy(np.array([prompt.encode("utf-8")], dtype=object))

        start_time = time.perf_counter()
        response = client.infer(model, inputs=[text_input], client_timeout=timeout)
        end_time = time.perf_counter()

        total_time = end_time - start_time

        # Estimate tokens from output
        num_tokens = 1
        try:
            output = response.as_numpy("text_output")
            if output is not None:
                decoded = (
                    output[0].decode("utf-8")
                    if isinstance(output[0], bytes)
                    else str(output[0])
                )
                # Rough token estimate: ~4 chars per token
                num_tokens = max(len(decoded) // 4, 1)
        except Exception:
            pass

        # For non-streaming:
        # - TTFT ≈ total_time (we get all tokens at once)
        # - TPOT cannot be accurately measured, but we estimate as:
        #   TPOT = (total_time - TTFT) / (num_tokens - 1)
        #   Since TTFT ≈ total_time for non-streaming, TPOT is ~0
        #
        # NOTE: For accurate TPOT, use --streaming mode
        tpot = 0.0  # Cannot measure TPOT accurately in non-streaming mode

        return RequestMetrics(
            ttft_s=total_time,
            end_to_end_latency_s=total_time,
            num_output_tokens=num_tokens,
            inter_token_latency_s=tpot,
            success=True,
        )

    except Exception as e:
        return RequestMetrics(
            ttft_s=0,
            end_to_end_latency_s=0,
            num_output_tokens=0,
            inter_token_latency_s=0,
            success=False,
            error=str(e),
        )


def run_benchmark(
    url: str,
    model: str,
    concurrency_levels: List[int],
    num_requests: int = 50,
    input_tokens_mean: int = 550,
    output_tokens_mean: int = 150,
    streaming: bool = False,
    timeout: int = 300,
):
    """Run benchmark at different concurrency levels (similar to llmperf)."""

    print(f"\n{'='*70}")
    print("gRPC Benchmark (llmperf-style)")
    print(f"{'='*70}")
    print(f"Model: {model}")
    print(f"Server: {url}")
    print(f"Streaming: {streaming}")
    print(f"Requests per level: {num_requests}")
    print(f"Input tokens (approx): {input_tokens_mean}")
    print(f"Output tokens target: {output_tokens_mean}")
    print(f"{'='*70}\n")

    # Generate prompts with approximate token count
    # ~4 chars per token, so input_tokens_mean * 4 chars
    base_text = "Please provide a detailed explanation of the following topic. " * (
        input_tokens_mean // 12
    )
    prompts = [
        f"{base_text} Request {i}. Generate approximately {output_tokens_mean} tokens."
        for i in range(num_requests)
    ]

    all_results = {}

    request_func = run_streaming_request if streaming else run_nonstreaming_request

    for concurrency in concurrency_levels:
        print(f"Running concurrency={concurrency}...")

        results = []
        start_time = time.perf_counter()

        # Use thread pool for concurrent requests
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(request_func, url, model, prompt, timeout)
                for prompt in prompts
            ]
            for future in futures:
                try:
                    result = future.result(timeout=timeout + 60)
                    results.append(result)
                except Exception as e:
                    results.append(
                        RequestMetrics(
                            ttft_s=0,
                            end_to_end_latency_s=0,
                            num_output_tokens=0,
                            inter_token_latency_s=0,
                            success=False,
                            error=str(e),
                        )
                    )

        elapsed = time.perf_counter() - start_time

        # Calculate statistics
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        if successful:
            ttfts = [r.ttft_s for r in successful]
            e2e_latencies = [r.end_to_end_latency_s for r in successful]
            itls = [r.inter_token_latency_s for r in successful]
            output_tokens = [r.num_output_tokens for r in successful]

            summary = {
                "concurrency": concurrency,
                "num_completed": len(successful),
                "num_failed": len(failed),
                # TTFT stats
                "ttft_s_mean": mean(ttfts),
                "ttft_s_median": median(ttfts),
                "ttft_s_p99": np.percentile(ttfts, 99) if len(ttfts) > 1 else ttfts[0],
                "ttft_s_std": stdev(ttfts) if len(ttfts) > 1 else 0,
                # E2E latency stats
                "e2e_latency_s_mean": mean(e2e_latencies),
                "e2e_latency_s_median": median(e2e_latencies),
                "e2e_latency_s_p99": np.percentile(e2e_latencies, 99)
                if len(e2e_latencies) > 1
                else e2e_latencies[0],
                # Inter-token latency (TPOT)
                "inter_token_latency_s_mean": mean(itls),
                "inter_token_latency_s_median": median(itls),
                # Throughput
                "request_throughput": len(successful) / elapsed,
                "output_token_throughput": sum(output_tokens) / elapsed,
                # Token counts
                "output_tokens_mean": mean(output_tokens),
            }
            all_results[concurrency] = summary

            print(f"  Completed: {len(successful)}/{num_requests}")
            print(f"  TTFT mean: {summary['ttft_s_mean']*1000:.2f} ms")
            print(f"  TTFT p99:  {summary['ttft_s_p99']*1000:.2f} ms")
            print(f"  E2E latency mean: {summary['e2e_latency_s_mean']*1000:.2f} ms")
            itl = summary["inter_token_latency_s_mean"]
            itl_str = "N/A (server not streaming)" if itl < 0 else f"{itl*1000:.2f} ms"
            print(f"  Inter-token latency (ITL/TPOT): {itl_str}")
            print(f"  Request throughput: {summary['request_throughput']:.2f} req/s")
            print(
                f"  Output token throughput: {summary['output_token_throughput']:.2f} tok/s"
            )
        else:
            print(f"  All {num_requests} requests failed!")
            if failed:
                print(f"  First error: {failed[0].error}")
        print()

    # Print summary table (llmperf style)
    print(f"\n{'='*90}")
    print("SUMMARY (llmperf-compatible metrics)")
    print(f"{'='*90}")
    print(
        f"{'Concurrency':>12} {'TTFT(ms)':>12} {'TTFT_p99':>12} {'ITL(ms)':>12} {'E2E(ms)':>12} {'Throughput':>12}"
    )
    print(f"{'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

    for conc, s in all_results.items():
        itl = s["inter_token_latency_s_mean"]
        itl_str = "N/A" if itl < 0 else f"{itl*1000:.2f}"
        print(
            f"{conc:>12} {s['ttft_s_mean']*1000:>12.2f} {s['ttft_s_p99']*1000:>12.2f} "
            f"{itl_str:>12} {s['e2e_latency_s_mean']*1000:>12.2f} "
            f"{s['request_throughput']:>12.2f}"
        )

    # Save results to JSON (like llmperf)
    output_file = f"benchmark_results_{model.replace('/', '_')}_{int(time.time())}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="gRPC Benchmark for Dynamo KServe server (llmperf-style)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic benchmark
  python benchmarks/grpc_benchmark.py --model Qwen/Qwen3-0.6B

  # Test multiple concurrency levels
  python benchmarks/grpc_benchmark.py --model Qwen/Qwen3-0.6B --concurrency 1 8 16 32 64 128

  # With streaming (if supported)
  python benchmarks/grpc_benchmark.py --model Qwen/Qwen3-0.6B --streaming

  # Custom input/output tokens
  python benchmarks/grpc_benchmark.py --model Qwen/Qwen3-0.6B --input-tokens 1000 --output-tokens 500
        """,
    )
    parser.add_argument("--url", default="127.0.0.1:8000", help="gRPC server URL")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Model name")
    parser.add_argument(
        "--concurrency",
        nargs="+",
        type=int,
        default=[64, 128],
        help="Concurrency levels to test",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=50,
        help="Number of requests per concurrency level",
    )
    parser.add_argument(
        "--input-tokens", type=int, default=550, help="Mean input tokens"
    )
    parser.add_argument(
        "--output-tokens", type=int, default=150, help="Mean output tokens"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming inference (for proper TTFT)",
    )
    parser.add_argument(
        "--timeout", type=int, default=300, help="Request timeout in seconds"
    )

    args = parser.parse_args()

    run_benchmark(
        url=args.url,
        model=args.model,
        concurrency_levels=args.concurrency,
        num_requests=args.requests,
        input_tokens_mean=args.input_tokens,
        output_tokens_mean=args.output_tokens,
        streaming=args.streaming,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
