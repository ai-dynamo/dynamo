# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from collections.abc import Sequence

from dynamo.llm import KvRouterConfig, MockEngineArgs
from dynamo.profiler.utils.replay_optimize import (
    EngineSpec,
    HardwareSpec,
    ReplayOptimizeSpec,
    RouterSpec,
    SLASpec,
    WorkloadSpec,
    optimize_dense_disagg_with_replay,
)

MODEL = "Qwen/Qwen3-32B"
BACKEND = "vllm"
GPU_SKU = "h200_sxm"
TOTAL_GPUS = 16
OVERLAP_WEIGHTS = [0.0, 0.5, 1.0, 2.0]
RESULT_COLUMNS: Sequence[str] = (
    "prefill_tp",
    "decode_tp",
    "prefill_workers",
    "decode_workers",
    "overlap_score_weight",
    "total_gpus_used",
    "output_throughput_tok_s",
    "prefix_cache_reused_ratio",
    "mean_ttft_ms",
    "mean_tpot_ms",
    "mean_e2e_latency_ms",
)


def _build_workload(
    *,
    trace_file: str | None,
    arrival_speedup_ratio: float,
) -> WorkloadSpec:
    if trace_file is not None:
        return WorkloadSpec(
            traceFile=trace_file,
            arrivalSpeedupRatio=arrival_speedup_ratio,
        )

    return WorkloadSpec(
        isl=32768,
        osl=256,
        requestCount=5000,
        concurrency=200,
        sharedPrefixRatio=0.5,
        numPrefixGroups=50,
    )


def _engine_args(worker_type: str) -> MockEngineArgs:
    return MockEngineArgs(
        block_size=512,
        num_gpu_blocks=20000,
        enable_prefix_caching=True,
        worker_type=worker_type,
    )


def run_example(
    *,
    trace_file: str | None = None,
    arrival_speedup_ratio: float = 1.0,
    max_parallel_evals: int = 1,
) -> None:
    spec = ReplayOptimizeSpec(
        engine=EngineSpec(
            model=MODEL,
            backend=BACKEND,
            basePrefillEngineArgs=_engine_args("prefill"),
            baseDecodeEngineArgs=_engine_args("decode"),
        ),
        hardware=HardwareSpec(gpuSku=GPU_SKU, totalGpus=TOTAL_GPUS),
        workload=_build_workload(
            trace_file=trace_file,
            arrival_speedup_ratio=arrival_speedup_ratio,
        ),
        sla=SLASpec(ttft=50000.0, itl=100.0, e2eLatency=60000.0),
        router=RouterSpec(
            baseRouterConfig=KvRouterConfig(),
            overlapWeights=OVERLAP_WEIGHTS,
        ),
        maxParallelEvals=max_parallel_evals,
    )

    result = optimize_dense_disagg_with_replay(spec)

    print("Best feasible:")
    print(result.best_feasible)
    print()

    print("Top feasible states:")
    print(result.feasible_df[list(RESULT_COLUMNS)].head(10).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the replay_optimize disaggregated KV-router example."
    )
    parser.add_argument(
        "--trace-file",
        help="Optional Mooncake-style JSONL trace. If omitted, runs the synthetic workload.",
    )
    parser.add_argument(
        "--arrival-speedup-ratio",
        type=float,
        default=1.0,
        help="Arrival speedup ratio to use with --trace-file.",
    )
    parser.add_argument(
        "--max-parallel-evals",
        type=int,
        default=1,
        help="Number of concurrent replay state evaluations.",
    )
    args = parser.parse_args()
    run_example(
        trace_file=args.trace_file,
        arrival_speedup_ratio=args.arrival_speedup_ratio,
        max_parallel_evals=args.max_parallel_evals,
    )


if __name__ == "__main__":
    main()
