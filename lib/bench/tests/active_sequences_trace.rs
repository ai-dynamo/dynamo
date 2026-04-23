// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[path = "../kv_router/common/shared.rs"]
mod common;

#[path = "../kv_router/active_sequences_shared.rs"]
mod active_sequences_shared;

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use active_sequences_shared::{generate_sequence_events, run_benchmark};
use anyhow::Context;
use common::process_mooncake_trace;
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::Registry;
use tracing_subscriber::layer::{Context as LayerContext, Layer};
use tracing_subscriber::prelude::*;

const BLOCK_SIZE: u32 = 128;
const NUM_GPU_BLOCKS: usize = 16384;
const TRACE_SIMULATION_DURATION_MS: u64 = 1000;
const BENCHMARK_DURATION_MS: u64 = 4000;
const NUM_UNIQUE_INFERENCE_WORKERS: usize = 10;

struct WarningCounterLayer {
    count: Arc<AtomicUsize>,
}

impl<S> Layer<S> for WarningCounterLayer
where
    S: Subscriber,
{
    fn on_event(&self, event: &Event<'_>, _ctx: LayerContext<'_, S>) {
        let metadata = event.metadata();
        let target = metadata.target();
        if matches!(*metadata.level(), Level::WARN | Level::ERROR)
            && (target.starts_with("dynamo_kv_router::sequences")
                || target.starts_with("dynamo_mocker"))
        {
            self.count.fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn warning_counter() -> Arc<AtomicUsize> {
    static COUNTER: OnceLock<Arc<AtomicUsize>> = OnceLock::new();

    COUNTER
        .get_or_init(|| {
            let count = Arc::new(AtomicUsize::new(0));
            let subscriber = Registry::default().with(WarningCounterLayer {
                count: Arc::clone(&count),
            });
            tracing::subscriber::set_global_default(subscriber)
                .expect("global warning counter subscriber should initialize once");
            count
        })
        .clone()
}

#[tokio::test(flavor = "current_thread")]
async fn active_sequences_trace_replays_without_warnings_or_leaks() -> anyhow::Result<()> {
    let warning_count = warning_counter();
    warning_count.store(0, Ordering::Relaxed);

    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/mooncake_trace_1000.jsonl");
    let fixture = fixture
        .to_str()
        .context("active-sequences trace fixture path is not valid UTF-8")?;

    let traces =
        process_mooncake_trace(fixture, BLOCK_SIZE, 1, 1, NUM_UNIQUE_INFERENCE_WORKERS, 42)?;
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
        warning_count.load(Ordering::Relaxed),
        0,
        "sequence replay emitted warn/error logs from dynamo_kv_router::sequences or dynamo_mocker"
    );

    Ok(())
}
