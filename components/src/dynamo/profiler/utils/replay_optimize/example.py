# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from collections.abc import Sequence

from dynamo.llm import KvRouterConfig, MockEngineArgs
from dynamo.profiler.utils.replay_optimize import (
    SyntheticReplayWorkload,
    TraceReplayWorkload,
    optimize_dense_disagg_with_replay,
)

MODEL = "Qwen/Qwen3-32B"
BACKEND = "vllm"
SYSTEM = "h200_sxm"
MAX_TOTAL_GPUS = 16
OVERLAP_SCORE_WEIGHTS = (0.0, 0.5, 1.0, 2.0)
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
) -> SyntheticReplayWorkload | TraceReplayWorkload:
    if trace_file is not None:
        return TraceReplayWorkload(
            trace_file=trace_file,
            arrival_speedup_ratio=arrival_speedup_ratio,
        )

    return SyntheticReplayWorkload(
        isl=32768,
        osl=256,
        request_count=5000,
        replay_concurrency=200,
        shared_prefix_ratio=0.5,
        num_prefix_groups=50,
    )


def _build_engine_args(*, worker_type: str) -> MockEngineArgs:
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
    result = optimize_dense_disagg_with_replay(
        model=MODEL,
        backend=BACKEND,
        system=SYSTEM,
        workload=_build_workload(
            trace_file=trace_file,
            arrival_speedup_ratio=arrival_speedup_ratio,
        ),
        base_prefill_engine_args=_build_engine_args(worker_type="prefill"),
        base_decode_engine_args=_build_engine_args(worker_type="decode"),
        base_router_config=KvRouterConfig(),
        max_total_gpus=MAX_TOTAL_GPUS,
        constraints={
            "mean_ttft_ms": 50000.0,
            "mean_tpot_ms": 100.0,
            "mean_e2e_latency_ms": 60000.0,
        },
        overlap_score_weights=OVERLAP_SCORE_WEIGHTS,
        max_parallel_evals=max_parallel_evals,
    )

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
