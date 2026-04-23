// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod support;

#[path = "../kv_router/common/shared.rs"]
mod common;

#[path = "../kv_router/active_sequences_shared.rs"]
mod active_sequences_shared;

use active_sequences_shared::{generate_sequence_events, run_benchmark};
use common::process_mooncake_trace;

const BLOCK_SIZE: u32 = 128;
const NUM_GPU_BLOCKS: usize = 16384;
const TRACE_SIMULATION_DURATION_MS: Option<u64> = None;
const BENCHMARK_DURATION_MS: u64 = 4000;
const NUM_UNIQUE_INFERENCE_WORKERS: usize = 10;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn active_sequences_trace_replays_without_warnings_or_leaks() -> anyhow::Result<()> {
    let warning_count = support::warning_counter(&["dynamo_kv_router::sequences", "dynamo_mocker"]);
    support::reset_warning_count(&warning_count);

    let fixture = support::fixture_path("mooncake_trace_1000.jsonl")?;
    let traces =
        process_mooncake_trace(&fixture, BLOCK_SIZE, 1, 1, NUM_UNIQUE_INFERENCE_WORKERS, 42)?;
    let sequence_traces = generate_sequence_events(
        &traces,
        NUM_GPU_BLOCKS,
        BLOCK_SIZE,
        TRACE_SIMULATION_DURATION_MS,
    )
    .await?;
    let run = run_benchmark(&sequence_traces, BLOCK_SIZE, BENCHMARK_DURATION_MS, 1).await?;

    assert!(
        run.kept_up,
        "benchmark replay fell behind in test profile; increase BENCHMARK_DURATION_MS if this becomes too tight"
    );
    assert!(
        run.results.ops_throughput > 0.0,
        "benchmark replay should record positive throughput"
    );
    assert_eq!(
        warning_count.load(std::sync::atomic::Ordering::Relaxed),
        0,
        "sequence replay emitted warn/error logs from dynamo_kv_router::sequences or dynamo_mocker"
    );

    Ok(())
}
