// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use dynamo_kv_router::protocols::{PrefillLoadHint, WorkerWithDpRank};
use dynamo_kv_router::{ActiveSequencesMultiWorker, SequenceRequest};
use dynamo_mocker::loadgen::Trace;
use dynamo_tokens::SequenceHash;
use tokio::time::{Duration, Instant};

use crate::common::{
    BenchmarkRun, NoopSequencePublisher, compute_benchmark_run, generate_replay_artifacts,
    make_progress_bar, rescale_trace_timestamps,
};

/// A single timestamped entry in a worker's sequence trace.
#[derive(Clone)]
pub enum SequenceTraceEntry {
    Add {
        request_id: String,
        block_hashes: Vec<SequenceHash>,
        isl: usize,
        output_length: u64,
    },
    PrefillComplete {
        request_id: String,
    },
    Free {
        request_id: String,
    },
}

/// A timestamped sequence trace entry for benchmark replay.
#[derive(Clone)]
pub struct SequenceTrace {
    pub entry: SequenceTraceEntry,
    pub timestamp_us: u64,
}

/// Pre-computed metadata for a request, stored before submission so the
/// output signal can look it up by UUID.
struct RequestMetadata {
    block_hashes: Vec<SequenceHash>,
    isl: usize,
    output_length: u64,
}

/// Run requests through the mocker to produce sequence lifecycle events
/// (add / prefill_complete / free) with realistic timing.
pub async fn generate_sequence_events(
    traces: &[Trace],
    num_gpu_blocks: usize,
    block_size: u32,
    trace_simulation_duration_ms: Option<u64>,
) -> anyhow::Result<Vec<Vec<SequenceTrace>>> {
    println!("Generating sequence events...");
    let artifacts = generate_replay_artifacts(
        traces,
        num_gpu_blocks,
        block_size,
        trace_simulation_duration_ms,
    )
    .await?;
    let mut all_traces = Vec::with_capacity(artifacts.len());

    for artifact in artifacts {
        let metadata = artifact
            .requests
            .iter()
            .map(|request| {
                (
                    request.uuid,
                    RequestMetadata {
                        block_hashes: request.replay_hashes.sequence_hashes.clone(),
                        isl: request.input_length,
                        output_length: request.output_length as u64,
                    },
                )
            })
            .collect::<HashMap<_, _>>();

        let mut entries = Vec::new();
        let mut seen = HashMap::new();

        for timed_signal in artifact.output_signals {
            let signal = timed_signal.signal;
            let request_id = signal.uuid.to_string();

            if let std::collections::hash_map::Entry::Vacant(entry) = seen.entry(signal.uuid) {
                entry.insert(());
                if let Some(meta) = metadata.get(&signal.uuid) {
                    entries.push(SequenceTrace {
                        entry: SequenceTraceEntry::Add {
                            request_id: request_id.clone(),
                            block_hashes: meta.block_hashes.clone(),
                            isl: meta.isl,
                            output_length: meta.output_length,
                        },
                        timestamp_us: timed_signal.timestamp_us,
                    });
                    entries.push(SequenceTrace {
                        entry: SequenceTraceEntry::PrefillComplete {
                            request_id: request_id.clone(),
                        },
                        timestamp_us: timed_signal.timestamp_us,
                    });
                }
            }

            if signal.completed {
                entries.push(SequenceTrace {
                    entry: SequenceTraceEntry::Free { request_id },
                    timestamp_us: timed_signal.timestamp_us,
                });
            }
        }

        all_traces.push(entries);
    }

    let total_adds = all_traces
        .iter()
        .flatten()
        .filter(|e| matches!(e.entry, SequenceTraceEntry::Add { .. }))
        .count();
    let total_frees = all_traces
        .iter()
        .flatten()
        .filter(|e| matches!(e.entry, SequenceTraceEntry::Free { .. }))
        .count();

    println!("Add events: {}, Free events: {}", total_adds, total_frees);

    Ok(all_traces)
}

/// Run the benchmark: replay sequence trace entries against a shared
/// ActiveSequencesMultiWorker, measuring potential_blocks_and_tokens /
/// add_request / mark_prefill_completed / free latency.
pub async fn run_benchmark(
    traces: &[Vec<SequenceTrace>],
    block_size: u32,
    benchmark_duration_ms: u64,
    inference_worker_duplication_factor: usize,
) -> anyhow::Result<BenchmarkRun> {
    let scaled = rescale_trace_timestamps(
        traces,
        benchmark_duration_ms,
        |entry| entry.timestamp_us,
        |entry, timestamp_us| SequenceTrace {
            entry: entry.entry.clone(),
            timestamp_us,
        },
    );
    let num_trace_workers = scaled.len();

    let total_workers = num_trace_workers * inference_worker_duplication_factor;
    let dp_range: HashMap<u64, (u32, u32)> =
        (0..total_workers as u64).map(|id| (id, (0, 1))).collect();
    let multi = Arc::new(ActiveSequencesMultiWorker::new(
        NoopSequencePublisher,
        block_size as usize,
        dp_range,
        false,
        0,
        "bench",
    ));

    let total_entries: u64 = scaled.iter().map(|t| t.len() as u64).sum::<u64>()
        * inference_worker_duplication_factor as u64;
    let total_blocks: usize = scaled
        .iter()
        .flat_map(|t| t.iter())
        .map(|entry| match &entry.entry {
            SequenceTraceEntry::Add { block_hashes, .. } => block_hashes.len(),
            _ => 0,
        })
        .sum::<usize>()
        * inference_worker_duplication_factor;

    let progress = make_progress_bar(Some(total_entries));

    let mut tasks = Vec::new();
    for replica in 0..inference_worker_duplication_factor {
        for (trace_idx, worker_trace) in scaled.iter().enumerate() {
            let worker_id = (replica * num_trace_workers + trace_idx) as u64;
            let worker = WorkerWithDpRank::from_worker_id(worker_id);
            let trace = make_unique_trace(worker_trace, worker_id);
            let progress = progress.clone();
            let multi = Arc::clone(&multi);

            tasks.push(tokio::spawn(async move {
                let capacity = trace.len();
                let mut latencies: Vec<u64> = Vec::with_capacity(capacity);

                let mut target = Instant::now();
                let mut iter = trace.into_iter().peekable();
                let mut local_count: u64 = 0;

                while let Some(entry) = iter.next() {
                    let entry_ts = entry.timestamp_us;

                    let start = minstant::Instant::now();
                    apply_entry(&multi, worker, entry.entry).await;
                    latencies.push(start.elapsed().as_nanos() as u64);
                    local_count += 1;

                    while iter.peek().is_some_and(|e| e.timestamp_us == entry_ts) {
                        let e = iter.next().unwrap();
                        let start = minstant::Instant::now();
                        apply_entry(&multi, worker, e.entry).await;
                        latencies.push(start.elapsed().as_nanos() as u64);
                        local_count += 1;
                    }

                    if let Some(next) = iter.peek() {
                        target += Duration::from_micros(next.timestamp_us - entry_ts);
                    }

                    if target > Instant::now() {
                        tokio::time::sleep_until(target).await;
                    }

                    if local_count > 100 {
                        progress.inc(local_count);
                        local_count = 0;
                    }
                }

                progress.inc(local_count);

                Ok::<_, anyhow::Error>(latencies)
            }));
        }
    }

    let mut all_latencies = Vec::new();
    for task in tasks {
        all_latencies.extend(task.await??);
    }

    let total_duration = progress.elapsed();
    multi.assert_completely_drained(Instant::now());

    let run = compute_benchmark_run(
        all_latencies.len(),
        total_blocks,
        benchmark_duration_ms,
        total_duration,
        all_latencies,
    );

    println!(
        "Ops Throughput: offered={} ops/s achieved={} ops/s (potential_blocks_and_tokens + add + prefill_complete + free)",
        run.results.offered_ops_throughput, run.results.ops_throughput
    );
    println!(
        "Block Throughput: offered={} block ops/s achieved={} block ops/s",
        run.results.offered_block_throughput, run.results.block_throughput
    );
    println!("Latency p99: {}us", run.results.latency_p99_us);

    Ok(run)
}

fn make_unique_trace(trace: &[SequenceTrace], worker_id: u64) -> Vec<SequenceTrace> {
    trace
        .iter()
        .map(|entry| {
            let new_entry = match &entry.entry {
                SequenceTraceEntry::Add {
                    request_id,
                    block_hashes,
                    isl,
                    output_length,
                } => SequenceTraceEntry::Add {
                    request_id: format!("{worker_id}:{request_id}"),
                    block_hashes: block_hashes.clone(),
                    isl: *isl,
                    output_length: *output_length,
                },
                SequenceTraceEntry::PrefillComplete { request_id } => {
                    SequenceTraceEntry::PrefillComplete {
                        request_id: format!("{worker_id}:{request_id}"),
                    }
                }
                SequenceTraceEntry::Free { request_id } => SequenceTraceEntry::Free {
                    request_id: format!("{worker_id}:{request_id}"),
                },
            };
            SequenceTrace {
                entry: new_entry,
                timestamp_us: entry.timestamp_us,
            }
        })
        .collect()
}

async fn apply_entry(
    multi: &ActiveSequencesMultiWorker<NoopSequencePublisher>,
    worker: WorkerWithDpRank,
    entry: SequenceTraceEntry,
) {
    let decay_now = tokio::time::Instant::now();
    match entry {
        SequenceTraceEntry::Add {
            request_id,
            block_hashes,
            isl,
            output_length,
        } => {
            let _ = multi.potential_blocks_and_tokens(Some(&block_hashes), isl, HashMap::new());
            let _ = multi.add_request(
                SequenceRequest {
                    request_id,
                    token_sequence: Some(block_hashes),
                    track_prefill_tokens: true,
                    expected_output_tokens: Some(output_length as u32),
                    prefill_load_hint: Some(PrefillLoadHint {
                        initial_effective_prefill_tokens: isl,
                        expected_prefill_duration: None,
                    }),
                    worker,
                    lora_name: None,
                },
                decay_now,
            );
        }
        SequenceTraceEntry::PrefillComplete { request_id } => {
            let _ = multi.mark_prefill_completed(&request_id, decay_now);
        }
        SequenceTraceEntry::Free { request_id } => {
            let _ = multi.free(&request_id, decay_now);
        }
    }
}
