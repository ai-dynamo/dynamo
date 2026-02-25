#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# AFD End-to-End Performance Benchmark
# Compares AFD (Attention-FFN Disaggregation) vs Baseline (Aggregated) serving

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYNAMO_ROOT="/home/openclaw/.openclaw/workspace/dynamo"
MODEL_PATH="/raid/model_hub/Qwen3-32B-FP8"
RESULTS_DIR="${SCRIPT_DIR}/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Model configuration
HIDDEN_DIM=5120
FFN_EXPANSION=4
NUM_LAYERS=64

# Hardware configuration (H20-3e)
MEMORY_BANDWIDTH_GBPS=2039  # H20 memory bandwidth
COMPUTE_TFLOPS=148  # H20 FP8 compute

echo "=============================================="
echo "AFD End-to-End Performance Benchmark"
echo "=============================================="
echo "Model: Qwen3-32B-FP8"
echo "Model Path: ${MODEL_PATH}"
echo "GPU: 8x NVIDIA H20-3e"
echo "Timestamp: ${TIMESTAMP}"
echo ""

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Run analytical benchmark
echo "Running analytical performance simulation..."
echo ""

cd "${DYNAMO_ROOT}"

PYTHONPATH="${DYNAMO_ROOT}/components/src" python3 << 'PYTHON_SCRIPT'
import sys
sys.path.insert(0, '/home/openclaw/.openclaw/workspace/dynamo/components/src')

# Import benchmark
from dynamo.sglang.afd_benchmark import (
    BenchmarkConfig,
    AFDSimulator,
    AFDBenchmark,
)
import json

# Configure for Qwen3-32B-FP8
config = BenchmarkConfig(
    model_path="/raid/model_hub/Qwen3-32B-FP8",
    hidden_dim=5120,  # Qwen3-32B hidden dim
    ffn_expansion=4,  # Standard FFN expansion
    num_layers=64,    # Qwen3-32B layers
    batch_sizes=[1, 2, 4, 8, 16, 32, 64],
    sequence_lengths=[128, 512, 1024, 2048, 4096],
    attention_ratios=[1, 2, 4, 8],
    memory_bandwidth_gbps=2039.0,  # H20
    compute_tflops=148.0,  # H20 FP8
    num_iterations=100,
)

# Run benchmark
benchmark = AFDBenchmark(config)
results = benchmark.run_sweep()

# Analyze
analysis = benchmark.analyze_results()

# Print results
print("\n" + "="*70)
print("AFD PERFORMANCE ANALYSIS - Qwen3-32B-FP8 on H20")
print("="*70)

print(f"\nModel Configuration:")
print(f"  Hidden Dim: {config.hidden_dim}")
print(f"  FFN Expansion: {config.ffn_expansion}")
print(f"  Num Layers: {config.num_layers}")
print(f"  Total Parameters: ~32B")

print(f"\nHardware Configuration:")
print(f"  GPU: 8x NVIDIA H20-3e (143GB each)")
print(f"  Memory Bandwidth: {config.memory_bandwidth_gbps} GB/s")
print(f"  Compute (FP8): {config.compute_tflops} TFLOPS")

print(f"\nBenchmark Summary:")
print(f"  Total tests: {analysis['summary']['total_benchmarks']}")
print(f"  Average speedup: {analysis['summary']['avg_speedup']:.2f}x")
print(f"  Maximum speedup: {analysis['summary']['max_speedup']:.2f}x")

# Best configuration
if analysis['summary']['best_overall']:
    best = analysis['summary']['best_overall']
    print(f"\n🏆 BEST CONFIGURATION:")
    print(f"  Batch Size: {best['batch_size']}")
    print(f"  Sequence Length: {best['sequence_length']}")
    print(f"  Attention Ratio: {best['attention_ratio']}")
    print(f"  Baseline Latency: {best['baseline_latency_ms']:.2f}ms")
    print(f"  AFD Latency: {best['afd_latency_ms']:.2f}ms")
    print(f"  Speedup: {best['speedup']:.2f}x")

# Top 10 configurations
print(f"\n📊 TOP 10 CONFIGURATIONS BY SPEEDUP:")
sorted_speedups = sorted(
    analysis['speedup_analysis'],
    key=lambda x: x['speedup'],
    reverse=True
)[:10]

for i, s in enumerate(sorted_speedups, 1):
    print(f"  {i}. batch={s['batch_size']:3d}, seq={s['sequence_length']:4d}, "
          f"ratio={s['attention_ratio']}, speedup={s['speedup']:.2f}x")

# Save results
output = {
    "config": {
        "model": "Qwen3-32B-FP8",
        "hidden_dim": config.hidden_dim,
        "ffn_expansion": config.ffn_expansion,
        "num_layers": config.num_layers,
        "memory_bandwidth_gbps": config.memory_bandwidth_gbps,
        "compute_tflops": config.compute_tflops,
    },
    "results": [r.to_dict() for r in results],
    "analysis": analysis,
}

with open("/home/openclaw/.openclaw/workspace/dynamo/components/src/dynamo/sglang/tests/afd_e2e_results.json", 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✅ Results saved to: afd_e2e_results.json")
print("="*70)
PYTHON_SCRIPT

echo ""
echo "=============================================="
echo "Benchmark Complete!"
echo "=============================================="
