#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AFD (Attention-FFN Disaggregation) Benchmark Suite.

This script provides benchmarking tools for evaluating AFD performance
against baseline (aggregated) serving.

Usage:
    python afd_benchmark.py --mode afd --model Qwen/Qwen3-0.6B
    python afd_benchmark.py --mode baseline --model Qwen/Qwen3-0.6B
    python afd_benchmark.py --mode sweep --model Qwen/Qwen3-0.6B --ratios 1,2,4,8
"""

import argparse
import asyncio
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for AFD benchmark."""
    
    # Model settings
    model_path: str = "Qwen/Qwen3-0.6B"
    hidden_dim: int = 896
    ffn_expansion: int = 4
    num_layers: int = 24
    
    # Workload settings
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    sequence_lengths: List[int] = field(default_factory=lambda: [128, 512, 1024, 2048])
    num_iterations: int = 100
    warmup_iterations: int = 10
    
    # AFD settings
    attention_ratios: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    microbatch_size: int = 256
    
    # Hardware settings
    memory_bandwidth_gbps: float = 2039.0  # H100 SXM
    compute_tflops: float = 989.0  # H100 FP16


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    
    mode: str  # "afd" or "baseline"
    batch_size: int
    sequence_length: int
    attention_ratio: Optional[int] = None
    
    # Latency metrics (ms)
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    
    # Throughput metrics
    tokens_per_second: float = 0.0
    requests_per_second: float = 0.0
    
    # Compute breakdown (ms)
    attention_time_ms: float = 0.0
    ffn_time_ms: float = 0.0
    transfer_time_ms: float = 0.0
    
    # Memory metrics (GB)
    attention_memory_gb: float = 0.0
    ffn_memory_gb: float = 0.0
    
    # Utilization
    attention_utilization: float = 0.0
    ffn_utilization: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "attention_ratio": self.attention_ratio,
            "latency": {
                "avg_ms": self.avg_latency_ms,
                "p50_ms": self.p50_latency_ms,
                "p99_ms": self.p99_latency_ms,
                "max_ms": self.max_latency_ms,
            },
            "throughput": {
                "tokens_per_second": self.tokens_per_second,
                "requests_per_second": self.requests_per_second,
            },
            "compute_breakdown": {
                "attention_ms": self.attention_time_ms,
                "ffn_ms": self.ffn_time_ms,
                "transfer_ms": self.transfer_time_ms,
            },
            "memory": {
                "attention_gb": self.attention_memory_gb,
                "ffn_gb": self.ffn_memory_gb,
            },
            "utilization": {
                "attention": self.attention_utilization,
                "ffn": self.ffn_utilization,
            },
        }


class AFDSimulator:
    """Simulates AFD performance based on analytical model.
    
    This uses the formulas from the AFD paper to estimate performance
    without requiring actual model execution.
    """
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
    
    def estimate_attention_time(
        self,
        batch_size: int,
        sequence_length: int,
    ) -> float:
        """Estimate attention computation time.
        
        Attention is memory-bound due to KV cache reads.
        Time ≈ (2 * batch * seq * hidden * num_layers) / memory_bandwidth
        
        Args:
            batch_size: Batch size
            sequence_length: Sequence length
            
        Returns:
            Estimated time in milliseconds
        """
        # Total memory transferred for attention
        # Each attention layer reads KV cache: 2 * batch * seq * hidden * dtype_size
        kv_cache_bytes = (
            2 * batch_size * sequence_length * 
            self.config.hidden_dim * 2 *  # float16
            self.config.num_layers
        )
        
        # Time = bytes / bandwidth
        time_seconds = kv_cache_bytes / (self.config.memory_bandwidth_gbps * 1e9)
        
        # Add compute time (small relative to memory for attention)
        attention_flops = (
            2 * batch_size * sequence_length * sequence_length *
            self.config.hidden_dim * self.config.num_layers
        )
        compute_time = attention_flops / (self.config.compute_tflops * 1e12 * 0.3)  # 30% efficiency
        
        return (time_seconds + compute_time) * 1000  # ms
    
    def estimate_ffn_time(
        self,
        batch_size: int,
        sequence_length: int,
    ) -> float:
        """Estimate FFN computation time.
        
        FFN is compute-bound with sufficient batch size.
        Time ≈ (8 * batch * seq * hidden * ffn_expansion * num_layers) / compute_throughput
        
        Args:
            batch_size: Batch size
            sequence_length: Sequence length
            
        Returns:
            Estimated time in milliseconds
        """
        # FFN FLOPs: 2 * batch * seq * hidden * (hidden * ffn_expansion)
        # With activation: ~8 ops per element
        ffn_flops = (
            8 * batch_size * sequence_length * 
            self.config.hidden_dim * self.config.ffn_expansion *
            self.config.num_layers
        )
        
        # Time = FLOPs / throughput
        # Use 60% efficiency for compute-bound
        time_seconds = ffn_flops / (self.config.compute_tflops * 1e12 * 0.6)
        
        return time_seconds * 1000  # ms
    
    def estimate_transfer_time(
        self,
        batch_size: int,
        sequence_length: int,
        attention_ratio: int,
    ) -> float:
        """Estimate activation transfer time.
        
        Uses RDMA (NIXL) with ~100GB/s bandwidth.
        
        Args:
            batch_size: Batch size
            sequence_length: Sequence length
            attention_ratio: r in r:1 topology
            
        Returns:
            Estimated time in milliseconds
        """
        # Activation size: batch * seq * hidden * dtype_size
        activation_bytes = batch_size * sequence_length * self.config.hidden_dim * 2  # float16
        
        # RDMA bandwidth (NIXL): ~100GB/s
        rdma_bandwidth_gbps = 100.0
        
        # Transfer time
        time_seconds = activation_bytes / (rdma_bandwidth_gbps * 1e9)
        
        # Pipeline efficiency: overlap with computation
        # With microbatch pipelining, transfer overhead is reduced
        pipeline_efficiency = 0.7  # 70% overlap
        
        return time_seconds * 1000 * (1 - pipeline_efficiency)
    
    def estimate_memory(
        self,
        batch_size: int,
        sequence_length: int,
        attention_ratio: int = 1,
    ) -> Tuple[float, float]:
        """Estimate memory usage for attention and FFN workers.
        
        Args:
            batch_size: Batch size
            sequence_length: Sequence length
            attention_ratio: r in r:1 topology
            
        Returns:
            Tuple of (attention_memory_gb, ffn_memory_gb)
        """
        # KV cache size per token (2 layers: K and V)
        kv_cache_per_token = 2 * self.config.hidden_dim * 2  # float16
        kv_cache_bytes = batch_size * sequence_length * kv_cache_per_token * self.config.num_layers
        
        # Attention worker memory (KV cache + activations)
        attention_memory = kv_cache_bytes + batch_size * sequence_length * self.config.hidden_dim * 2
        
        # FFN worker memory (activations only, stateless)
        # Shared by r attention workers
        ffn_memory = batch_size * sequence_length * self.config.hidden_dim * 2 * attention_ratio
        
        return (
            attention_memory / 1e9,  # GB
            ffn_memory / 1e9,  # GB
        )
    
    def simulate_benchmark(
        self,
        batch_size: int,
        sequence_length: int,
        attention_ratio: int,
    ) -> BenchmarkResult:
        """Simulate AFD benchmark run.
        
        Args:
            batch_size: Batch size
            sequence_length: Sequence length
            attention_ratio: r in r:1 topology
            
        Returns:
            BenchmarkResult with simulated metrics
        """
        # Estimate compute times
        attention_time = self.estimate_attention_time(batch_size, sequence_length)
        ffn_time = self.estimate_ffn_time(batch_size, sequence_length)
        transfer_time = self.estimate_transfer_time(
            batch_size, sequence_length, attention_ratio
        )
        
        # AFD total time (parallel with pipeline overlap)
        # Attention and FFN can run in parallel with pipelining
        # Total time ≈ max(attention_time, ffn_time) + transfer_overhead
        ffn_time_per_attention = ffn_time / attention_ratio
        total_time = max(attention_time, ffn_time_per_attention) + transfer_time
        
        # Memory estimates
        attention_mem, ffn_mem = self.estimate_memory(
            batch_size, sequence_length, attention_ratio
        )
        
        # Utilization
        attention_util = attention_time / total_time
        ffn_util = ffn_time_per_attention / total_time
        
        # Generate latency distribution (simulated)
        latencies = np.random.normal(total_time, total_time * 0.1, self.config.num_iterations)
        latencies = np.maximum(latencies, total_time * 0.5)  # Floor
        
        return BenchmarkResult(
            mode="afd",
            batch_size=batch_size,
            sequence_length=sequence_length,
            attention_ratio=attention_ratio,
            avg_latency_ms=float(np.mean(latencies)),
            p50_latency_ms=float(np.percentile(latencies, 50)),
            p99_latency_ms=float(np.percentile(latencies, 99)),
            max_latency_ms=float(np.max(latencies)),
            tokens_per_second=batch_size * sequence_length / (total_time / 1000),
            requests_per_second=1000 / total_time,
            attention_time_ms=attention_time,
            ffn_time_ms=ffn_time,
            transfer_time_ms=transfer_time,
            attention_memory_gb=attention_mem,
            ffn_memory_gb=ffn_mem,
            attention_utilization=attention_util,
            ffn_utilization=ffn_util,
        )
    
    def simulate_baseline(
        self,
        batch_size: int,
        sequence_length: int,
    ) -> BenchmarkResult:
        """Simulate baseline (aggregated) serving.
        
        Args:
            batch_size: Batch size
            sequence_length: Sequence length
            
        Returns:
            BenchmarkResult with simulated metrics
        """
        # In aggregated mode, attention and FFN run sequentially
        attention_time = self.estimate_attention_time(batch_size, sequence_length)
        ffn_time = self.estimate_ffn_time(batch_size, sequence_length)
        
        # Total time = attention + FFN (sequential)
        total_time = attention_time + ffn_time
        
        # Memory (all on one GPU)
        attention_mem, ffn_mem = self.estimate_memory(batch_size, sequence_length, 1)
        total_memory = attention_mem + ffn_mem
        
        # Generate latency distribution
        latencies = np.random.normal(total_time, total_time * 0.15, self.config.num_iterations)
        latencies = np.maximum(latencies, total_time * 0.5)
        
        return BenchmarkResult(
            mode="baseline",
            batch_size=batch_size,
            sequence_length=sequence_length,
            avg_latency_ms=float(np.mean(latencies)),
            p50_latency_ms=float(np.percentile(latencies, 50)),
            p99_latency_ms=float(np.percentile(latencies, 99)),
            max_latency_ms=float(np.max(latencies)),
            tokens_per_second=batch_size * sequence_length / (total_time / 1000),
            requests_per_second=1000 / total_time,
            attention_time_ms=attention_time,
            ffn_time_ms=ffn_time,
            transfer_time_ms=0.0,
            attention_memory_gb=total_memory,
            ffn_memory_gb=0.0,
            attention_utilization=attention_time / total_time,
            ffn_utilization=ffn_time / total_time,
        )


class AFDBenchmark:
    """Main benchmark suite for AFD evaluation."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.simulator = AFDSimulator(config)
        self.results: List[BenchmarkResult] = []
    
    def run_sweep(self) -> List[BenchmarkResult]:
        """Run comprehensive benchmark sweep."""
        logger.info("Starting AFD benchmark sweep...")
        
        # Baseline benchmarks
        logger.info("Running baseline benchmarks...")
        for batch_size in self.config.batch_sizes:
            for seq_len in self.config.sequence_lengths:
                result = self.simulator.simulate_baseline(batch_size, seq_len)
                self.results.append(result)
                logger.info(
                    f"Baseline: batch={batch_size}, seq={seq_len}, "
                    f"latency={result.avg_latency_ms:.2f}ms"
                )
        
        # AFD benchmarks
        logger.info("Running AFD benchmarks...")
        for ratio in self.config.attention_ratios:
            for batch_size in self.config.batch_sizes:
                for seq_len in self.config.sequence_lengths:
                    result = self.simulator.simulate_benchmark(
                        batch_size, seq_len, ratio
                    )
                    self.results.append(result)
                    logger.info(
                        f"AFD r={ratio}: batch={batch_size}, seq={seq_len}, "
                        f"latency={result.avg_latency_ms:.2f}ms"
                    )
        
        return self.results
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze benchmark results."""
        if not self.results:
            return {}
        
        analysis = {
            "config": {
                "model": self.config.model_path,
                "hidden_dim": self.config.hidden_dim,
                "ffn_expansion": self.config.ffn_expansion,
                "num_layers": self.config.num_layers,
            },
            "summary": {},
            "best_configurations": {},
            "speedup_analysis": {},
        }
        
        # Group by configuration
        baseline_results = [r for r in self.results if r.mode == "baseline"]
        afd_results = [r for r in self.results if r.mode == "afd"]
        
        # Calculate speedups
        speedups = []
        for afd_result in afd_results:
            matching_baseline = [
                r for r in baseline_results
                if r.batch_size == afd_result.batch_size
                and r.sequence_length == afd_result.sequence_length
            ]
            if matching_baseline:
                baseline = matching_baseline[0]
                speedup = baseline.avg_latency_ms / afd_result.avg_latency_ms
                speedups.append({
                    "batch_size": afd_result.batch_size,
                    "sequence_length": afd_result.sequence_length,
                    "attention_ratio": afd_result.attention_ratio,
                    "speedup": speedup,
                    "baseline_latency_ms": baseline.avg_latency_ms,
                    "afd_latency_ms": afd_result.avg_latency_ms,
                })
        
        analysis["speedup_analysis"] = speedups
        
        # Find best configuration per workload
        best_configs = {}
        for batch_size in self.config.batch_sizes:
            for seq_len in self.config.sequence_lengths:
                workload_key = f"batch_{batch_size}_seq_{seq_len}"
                workload_results = [
                    s for s in speedups
                    if s["batch_size"] == batch_size and s["sequence_length"] == seq_len
                ]
                if workload_results:
                    best = max(workload_results, key=lambda x: x["speedup"])
                    best_configs[workload_key] = best
        
        analysis["best_configurations"] = best_configs
        
        # Overall summary
        analysis["summary"] = {
            "total_benchmarks": len(self.results),
            "avg_speedup": statistics.mean(s["speedup"] for s in speedups) if speedups else 0,
            "max_speedup": max(s["speedup"] for s in speedups) if speedups else 0,
            "best_overall": max(speedups, key=lambda x: x["speedup"]) if speedups else None,
        }
        
        return analysis
    
    def export_results(self, output_path: str) -> None:
        """Export results to JSON file."""
        output = {
            "results": [r.to_dict() for r in self.results],
            "analysis": self.analyze_results(),
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Results exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="AFD Benchmark Suite")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Model path")
    parser.add_argument("--mode", choices=["afd", "baseline", "sweep"], default="sweep")
    parser.add_argument("--ratios", default="1,2,4,8", help="Attention ratios to test")
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,32", help="Batch sizes")
    parser.add_argument("--sequence-lengths", default="128,512,1024,2048", help="Sequence lengths")
    parser.add_argument("--output", default="afd_benchmark_results.json", help="Output file")
    
    args = parser.parse_args()
    
    # Parse configuration
    config = BenchmarkConfig(
        model_path=args.model,
        batch_sizes=[int(x) for x in args.batch_sizes.split(",")],
        sequence_lengths=[int(x) for x in args.sequence_lengths.split(",")],
        attention_ratios=[int(x) for x in args.ratios.split(",")],
    )
    
    # Run benchmark
    benchmark = AFDBenchmark(config)
    benchmark.run_sweep()
    
    # Analyze and export
    analysis = benchmark.analyze_results()
    benchmark.export_results(args.output)
    
    # Print summary
    print("\n" + "="*60)
    print("AFD BENCHMARK SUMMARY")
    print("="*60)
    print(f"Model: {config.model_path}")
    print(f"Total benchmarks: {analysis['summary']['total_benchmarks']}")
    print(f"Average speedup: {analysis['summary']['avg_speedup']:.2f}x")
    print(f"Maximum speedup: {analysis['summary']['max_speedup']:.2f}x")
    
    if analysis['summary']['best_overall']:
        best = analysis['summary']['best_overall']
        print(f"\nBest configuration:")
        print(f"  Batch size: {best['batch_size']}")
        print(f"  Sequence length: {best['sequence_length']}")
        print(f"  Attention ratio: {best['attention_ratio']}")
        print(f"  Speedup: {best['speedup']:.2f}x")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
