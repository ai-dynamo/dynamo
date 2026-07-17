// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod support;

#[path = "../kv_router/active_sequences_open_loop.rs"]
mod active_sequences_open_loop;
#[path = "../kv_router/active_sequences_shared.rs"]
mod active_sequences_shared;

use std::collections::HashMap;

use active_sequences_open_loop::{
    ActiveOperationKind, ActiveSequencesRunConfig, accumulate_projection_digest,
    prepare_active_sequences_corpus, run_active_sequences_benchmark, summarize_worker_projections,
};
use active_sequences_shared::{SequenceTrace, SequenceTraceEntry, generate_sequence_events};
use dynamo_bench::kv_router_common::replay::{NoopSequencePublisher, process_mooncake_trace};
use dynamo_kv_router::protocols::{PrefillLoadHint, WorkerWithDpRank};
use dynamo_kv_router::{ActiveSequenceStride, ActiveSequencesMultiWorker, SequenceRequest};

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
    let original_traces = sequence_traces.clone();
    let corpus =
        prepare_active_sequences_corpus(sequence_traces, BLOCK_SIZE, BENCHMARK_DURATION_MS, 1)?;
    for operation in corpus
        .operations
        .iter()
        .filter(|operation| operation.kind == ActiveOperationKind::ProjectAndAdd)
    {
        let SequenceTraceEntry::Add { block_hashes, .. } =
            &original_traces[operation.worker_id as usize][operation.source_ordinal].entry
        else {
            panic!("prepared add points to a non-Add source entry");
        };
        assert_eq!(
            corpus.operation_hashes(operation.id)?,
            block_hashes,
            "flattened hash slab diverged for operation {}",
            operation.id
        );
    }
    drop(original_traces);
    let result = run_active_sequences_benchmark(
        corpus,
        ActiveSequencesRunConfig {
            operation_lanes: NUM_UNIQUE_INFERENCE_WORKERS,
            spin_us: 50,
            issue_lag_diagnostic_threshold_us: 250,
        },
    )
    .await?;

    assert!(
        result.kept_up,
        "benchmark replay fell behind in test profile; increase BENCHMARK_DURATION_MS if this becomes too tight"
    );
    assert!(
        result.achieved_logical_ops_per_sec > 0.0,
        "benchmark replay should record positive throughput"
    );
    assert!(result.generator_valid, "{:?}", result.failure_reasons);
    assert!(result.final_state_empty);
    assert_eq!(result.total_adds, result.total_frees);
    assert_eq!(
        warning_count.load(std::sync::atomic::Ordering::Relaxed),
        0,
        "sequence replay emitted warn/error logs from dynamo_kv_router::sequences or dynamo_mocker"
    );

    Ok(())
}

fn small_sequence_trace() -> Vec<Vec<SequenceTrace>> {
    vec![vec![
        SequenceTrace {
            entry: SequenceTraceEntry::Add {
                request_id: "request-a".to_string(),
                block_hashes: vec![1, 2],
                isl: 256,
                output_length: 8,
            },
            timestamp_us: 10,
        },
        SequenceTrace {
            entry: SequenceTraceEntry::PrefillComplete {
                request_id: "request-a".to_string(),
            },
            timestamp_us: 10,
        },
        SequenceTrace {
            entry: SequenceTraceEntry::Free {
                request_id: "request-a".to_string(),
            },
            timestamp_us: 10,
        },
    ]]
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn active_sequences_replay_matches_sequential_direct_oracle() -> anyhow::Result<()> {
    let result = run_active_sequences_benchmark(
        prepare_active_sequences_corpus(small_sequence_trace(), BLOCK_SIZE, 100, 1)?,
        ActiveSequencesRunConfig {
            operation_lanes: 1,
            spin_us: 50,
            issue_lag_diagnostic_threshold_us: 250,
        },
    )
    .await?;

    let oracle = ActiveSequencesMultiWorker::new_without_expiry(
        NoopSequencePublisher,
        BLOCK_SIZE as usize,
        HashMap::from([(0, (0, 1))]),
        false,
        0,
        "bench",
    );
    let now = tokio::time::Instant::now();
    let hashes = ActiveSequenceStride::ONE.sample_dense(vec![1, 2]);
    let projections = oracle.project_worker_loads(Some(&hashes), now);
    let projection_count = projections.len() as u64;
    let (projection_inspected, projection_digest) = summarize_worker_projections(&projections);
    let projection_digest = accumulate_projection_digest(0, 0, projection_digest);
    oracle.add_request(
        SequenceRequest {
            request_id: "0:request-a".to_string(),
            token_sequence: Some(hashes),
            track_prefill_tokens: true,
            expected_output_tokens: Some(8),
            prefill_load_hint: Some(PrefillLoadHint {
                initial_effective_prefill_tokens: 256,
                expected_prefill_duration: None,
            }),
            worker: WorkerWithDpRank::from_worker_id(0),
            lora_name: None,
        },
        now,
    )?;
    oracle.mark_prefill_completed(&"0:request-a".to_string(), now)?;
    oracle.free(&"0:request-a".to_string(), now)?;
    oracle.assert_completely_drained(now);

    assert!(result.generator_valid, "{:?}", result.failure_reasons);
    assert!(result.final_state_empty);
    assert_eq!(result.worker_projections_produced, projection_count);
    assert_eq!(result.worker_projections_inspected, projection_inspected);
    assert_eq!(result.worker_projection_digest, projection_digest);
    assert_eq!(result.total_adds, 1);
    assert_eq!(result.total_prefill_completes, 1);
    assert_eq!(result.total_frees, 1);
    Ok(())
}
