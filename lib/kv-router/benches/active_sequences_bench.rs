// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[path = "common/mod.rs"]
mod common;
use common::*;

use clap::Parser;
use dynamo_kv_router::protocols::{ActiveLoad, ActiveSequenceEvent, WorkerWithDpRank};
use dynamo_kv_router::{
    ActiveSequencesMultiWorker, OverlapScores, SequencePublisher, SequenceRequest,
};
use dynamo_mocker::common::protocols::{DirectRequest, OutputSignal};
use dynamo_mocker::scheduler::Scheduler;
use dynamo_tokens::SequenceHash;
use std::collections::HashMap;
use std::future;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio::time::{Duration, Instant};
use uuid::Uuid;

struct NoopSequencePublisher;

impl SequencePublisher for NoopSequencePublisher {
    fn publish_event(
        &self,
        _event: &ActiveSequenceEvent,
    ) -> impl future::Future<Output = anyhow::Result<()>> + Send {
        future::ready(Ok(()))
    }

    fn publish_load(&self, _load: ActiveLoad) {}

    fn observe_load(&self, _: &WorkerWithDpRank, _: &str, _: usize, _: usize) {}
}

#[derive(Parser, Debug)]
#[clap(
    version,
    about = "ActiveSequences add_request/free throughput benchmark"
)]
struct Args {
    #[clap(flatten)]
    common: CommonArgs,

    /// Output path for the sweep plot SVG.
    #[clap(long, default_value = "active_seq_sweep_plot.svg")]
    sweep_output: String,
}

/// Pre-computed metadata for a request, stored before submission so the
/// output signal can look it up by UUID.
struct RequestMetadata {
    block_hashes: Vec<SequenceHash>,
    isl: usize,
    output_length: u64,
}

/// A single timestamped entry in a worker's sequence trace.
#[derive(Clone)]
enum SequenceTraceEntry {
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
struct SequenceTrace {
    entry: SequenceTraceEntry,
    timestamp_us: u64,
}

/// Run requests through the mocker to produce sequence lifecycle events
/// (add / prefill_complete / free) with realistic timing.
///
/// For each worker we:
/// 1. Create a Scheduler with an output_tx channel (no KvCacheEventSink needed)
/// 2. Pre-compute block hashes for each request
/// 3. Drain OutputSignal: first signal per UUID → Add + PrefillComplete,
///    completed=true → Free
/// 4. Collect timestamps for later replay
async fn generate_sequence_events(
    traces: &[Vec<MooncakeRequest>],
    num_gpu_blocks: usize,
    block_size: u32,
    trace_simulation_duration_ms: u64,
) -> anyhow::Result<Vec<Vec<SequenceTrace>>> {
    println!("Generating sequence events...");
    let sched_args = default_mock_engine_args(num_gpu_blocks, block_size as usize)?;

    let scaled_traces: Vec<_> = traces
        .iter()
        .map(|worker_trace| scale_mooncake_trace(worker_trace, trace_simulation_duration_ms))
        .collect();

    let progress = make_progress_bar(Some(traces.iter().map(|w| w.len() as u64).sum::<u64>()));

    let mut tasks: Vec<JoinHandle<anyhow::Result<Vec<SequenceTrace>>>> = Vec::new();

    for worker_trace in scaled_traces {
        let sched_args = sched_args.clone();
        let progress = progress.clone();

        tasks.push(tokio::spawn(async move {
            let (output_tx, mut output_rx) = mpsc::unbounded_channel::<OutputSignal>();

            // No KvCacheEventSink — we only need output signals
            let scheduler = Scheduler::new(sched_args, 0, Some(output_tx), None, None);

            // Pre-compute metadata for each request before submission
            let mut metadata: HashMap<Uuid, RequestMetadata> = HashMap::new();
            for req in &worker_trace {
                let block_hashes: Vec<SequenceHash> = req
                    .hash_ids
                    .iter()
                    .map(|&id| local_block_hash_from_id(id, block_size).0)
                    .collect();
                let isl = req.hash_ids.len() * block_size as usize;
                metadata.insert(
                    req.uuid,
                    RequestMetadata {
                        block_hashes,
                        isl,
                        output_length: req.output_length,
                    },
                );
            }

            // Spawn drain task that converts OutputSignals → SequenceTrace entries
            let drain_handle: JoinHandle<Vec<SequenceTrace>> = tokio::spawn(async move {
                let mut entries = Vec::new();
                let mut seen: HashMap<Uuid, bool> = HashMap::new();

                while let Some(signal) = output_rx.recv().await {
                    let request_id = signal.uuid.to_string();

                    if let std::collections::hash_map::Entry::Vacant(e) = seen.entry(signal.uuid) {
                        e.insert(false);

                        if let Some(meta) = metadata.get(&signal.uuid) {
                            entries.push(SequenceTrace {
                                entry: SequenceTraceEntry::Add {
                                    request_id: request_id.clone(),
                                    block_hashes: meta.block_hashes.clone(),
                                    isl: meta.isl,
                                    output_length: meta.output_length,
                                },
                                timestamp_us: 0, // rescaled later
                            });
                            entries.push(SequenceTrace {
                                entry: SequenceTraceEntry::PrefillComplete {
                                    request_id: request_id.clone(),
                                },
                                timestamp_us: 0,
                            });
                        }
                    }

                    if signal.completed {
                        seen.insert(signal.uuid, true);
                        entries.push(SequenceTrace {
                            entry: SequenceTraceEntry::Free { request_id },
                            timestamp_us: 0,
                        });
                    }
                }

                entries
            });

            // Submit requests at scaled timing
            let mut i = 0;
            let mut target = Instant::now();
            let start = target;

            while i < worker_trace.len() {
                let prev_i = i;
                scheduler
                    .receive(DirectRequest {
                        tokens: tokens_from_request(&worker_trace[i], block_size),
                        max_output_tokens: worker_trace[i].output_length as usize,
                        uuid: Some(worker_trace[i].uuid),
                        dp_rank: 0,
                    })
                    .await;
                i += 1;

                while i < worker_trace.len()
                    && worker_trace[i].timestamp == worker_trace[i - 1].timestamp
                {
                    scheduler
                        .receive(DirectRequest {
                            tokens: tokens_from_request(&worker_trace[i], block_size),
                            max_output_tokens: worker_trace[i].output_length as usize,
                            uuid: Some(worker_trace[i].uuid),
                            dp_rank: 0,
                        })
                        .await;
                    i += 1;
                }

                if i < worker_trace.len() {
                    target += Duration::from_millis(
                        worker_trace[i].timestamp - worker_trace[i - 1].timestamp,
                    );
                }

                tokio::time::sleep_until(target).await;
                progress.inc((i - prev_i) as u64);
            }

            // Drop scheduler → CancelGuard fires → background task exits →
            // output_tx dropped → drain task sees None
            drop(scheduler);

            let mut entries = drain_handle.await?;

            // Assign monotonically increasing timestamps based on entry order
            let total_us = (Instant::now() - start).as_micros() as u64;
            let num_entries = entries.len() as u64;
            for (idx, entry) in entries.iter_mut().enumerate() {
                entry.timestamp_us = if num_entries > 1 {
                    idx as u64 * total_us / (num_entries - 1)
                } else {
                    0
                };
            }

            Ok(entries)
        }));
    }

    let mut all_traces = Vec::new();
    for task in tasks {
        all_traces.push(task.await??);
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

/// Rescale sequence trace timestamps into the benchmark duration.
fn rescale_traces(
    traces: &[Vec<SequenceTrace>],
    benchmark_duration_ms: u64,
) -> Vec<Vec<SequenceTrace>> {
    traces
        .iter()
        .map(|worker_trace| {
            if worker_trace.is_empty() {
                return Vec::new();
            }
            let max_ts = worker_trace
                .last()
                .map(|e| e.timestamp_us)
                .unwrap_or(1)
                .max(1);
            let target_us = benchmark_duration_ms * 1000;
            worker_trace
                .iter()
                .map(|entry| SequenceTrace {
                    entry: entry.entry.clone(),
                    timestamp_us: entry.timestamp_us * target_us / max_ts,
                })
                .collect()
        })
        .collect()
}

/// Run the benchmark: replay sequence trace entries against a shared
/// ActiveSequencesMultiWorker, measuring potential_blocks_and_tokens /
/// add_request / mark_prefill_completed / free latency.
async fn run_benchmark(
    traces: &[Vec<SequenceTrace>],
    block_size: u32,
    benchmark_duration_ms: u64,
    inference_worker_duplication_factor: usize,
) -> anyhow::Result<BenchmarkResults> {
    let scaled = rescale_traces(traces, benchmark_duration_ms);
    let num_trace_workers = scaled.len();

    // Total bench workers = trace workers × duplication factor.
    // Each gets a unique WorkerWithDpRank in the shared multi-worker.
    let total_workers = num_trace_workers * inference_worker_duplication_factor;
    let dp_sizes: HashMap<u64, u32> = (0..total_workers as u64).map(|id| (id, 1)).collect();
    let multi = Arc::new(ActiveSequencesMultiWorker::new(
        NoopSequencePublisher,
        block_size as usize,
        dp_sizes,
        false,
        0,
        "bench",
    ));

    let total_entries: u64 = scaled.iter().map(|t| t.len() as u64).sum::<u64>()
        * inference_worker_duplication_factor as u64;

    // Count blocks before consuming traces
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

            // Make request IDs unique per worker so the shared map has no conflicts
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

                    // Process all entries at the same timestamp
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

    if progress.elapsed() > Duration::from_millis(benchmark_duration_ms * 11 / 10) {
        eprintln!(
            "WARNING: Benchmarker could not keep up. Rerun with a larger --benchmark-duration-ms."
        );
    }

    let total_duration = progress.elapsed();
    let total_ops = all_latencies.len();

    let offered_ops_throughput = total_ops as f32 / benchmark_duration_ms as f32 * 1000.0;
    let ops_throughput = total_ops as f32 / total_duration.as_millis() as f32 * 1000.0;
    let offered_block_throughput = total_blocks as f32 / benchmark_duration_ms as f32 * 1000.0;
    let block_throughput = total_blocks as f32 / total_duration.as_millis() as f32 * 1000.0;

    all_latencies.sort_unstable();
    let latency_p99_us = if all_latencies.is_empty() {
        0.0
    } else {
        all_latencies[all_latencies.len() * 99 / 100] as f32 / 1000.0
    };

    println!(
        "Ops Throughput: {} ops/s (potential_blocks_and_tokens + add + prefill_complete + free)",
        ops_throughput
    );
    println!("Block Throughput: {} block ops/s", block_throughput);
    println!("Latency p99: {}us", latency_p99_us);

    Ok(BenchmarkResults {
        offered_ops_throughput,
        ops_throughput,
        offered_block_throughput,
        block_throughput,
        latency_p99_us,
    })
}

/// Make request IDs unique by prefixing with the worker ID, so the shared
/// request_to_worker map has no conflicts when traces are duplicated.
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
    match entry {
        SequenceTraceEntry::Add {
            request_id,
            block_hashes,
            isl,
            output_length,
        } => {
            let _ = multi.potential_blocks_and_tokens(
                Some(&block_hashes),
                isl,
                OverlapScores::default(),
            );
            let _ = multi
                .add_request(SequenceRequest {
                    request_id,
                    token_sequence: Some(block_hashes),
                    isl,
                    overlap: 0,
                    expected_output_tokens: Some(output_length as u32),
                    worker,
                    lora_name: None,
                })
                .await;
        }
        SequenceTraceEntry::PrefillComplete { request_id } => {
            let _ = multi.mark_prefill_completed(&request_id).await;
        }
        SequenceTraceEntry::Free { request_id } => {
            let _ = multi.free(&request_id).await;
        }
    }
}

async fn run_tests() -> anyhow::Result<()> {
    use std::fs::File;
    use std::io::Write;

    let path = std::env::temp_dir().join(format!(
        "active_seq_bench_test_{}.jsonl",
        std::process::id()
    ));
    {
        let mut f = File::create(&path)?;
        for (i, (hash_ids, output_length)) in
            [(&[0u64, 1, 2] as &[u64], 10u64), (&[0, 1, 3, 4], 10)]
                .iter()
                .enumerate()
        {
            writeln!(
                f,
                "{}",
                serde_json::json!({
                    "timestamp": i as u64,
                    "hash_ids": hash_ids,
                    "output_length": output_length,
                })
            )?;
        }
    }

    let traces = process_mooncake_trace(path.to_str().unwrap(), 1, 1, 2, 42)?;
    std::fs::remove_file(&path).ok();

    println!(
        "Loaded {} workers, {} total requests",
        traces.len(),
        traces.iter().map(|t| t.len()).sum::<usize>()
    );

    let seq_traces = generate_sequence_events(&traces, 1048576, 512, 100).await?;

    let total_adds = seq_traces
        .iter()
        .flatten()
        .filter(|e| matches!(e.entry, SequenceTraceEntry::Add { .. }))
        .count();
    let total_frees = seq_traces
        .iter()
        .flatten()
        .filter(|e| matches!(e.entry, SequenceTraceEntry::Free { .. }))
        .count();

    assert!(total_adds > 0, "expected at least one Add event");
    assert!(total_frees > 0, "expected at least one Free event");
    assert_eq!(total_adds, total_frees, "adds and frees should match");

    println!("All tests passed.");
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if args.common.test {
        return run_tests().await;
    }

    let path = args
        .common
        .mooncake_trace_path
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("mooncake_trace_path is required for benchmarking"))?;
    let traces = process_mooncake_trace(
        path,
        args.common.trace_length_factor,
        args.common.trace_duplication_factor,
        args.common.num_unique_inference_workers,
        args.common.seed,
    )?;

    let seq_traces = generate_sequence_events(
        &traces,
        args.common.num_gpu_blocks,
        args.common.block_size,
        args.common.trace_simulation_duration_ms,
    )
    .await?;

    if args.common.sweep {
        let durations = compute_sweep_durations(
            args.common.sweep_min_ms,
            args.common.sweep_max_ms,
            args.common.sweep_steps,
        );

        let mut results: Vec<(u64, BenchmarkResults)> = Vec::new();
        for &dur_ms in &durations {
            println!("\n=== Sweep: benchmark_duration_ms = {} ===", dur_ms);
            let result = run_benchmark(
                &seq_traces,
                args.common.block_size,
                dur_ms,
                args.common.inference_worker_duplication_factor,
            )
            .await?;
            results.push((dur_ms, result));
        }

        print_sweep_summary("active-sequences", &results);

        let all_results = vec![("active-sequences", results)];
        plot_sweep(&all_results, &args.sweep_output)?;
    } else {
        run_benchmark(
            &seq_traces,
            args.common.block_size,
            args.common.benchmark_duration_ms,
            args.common.inference_worker_duplication_factor,
        )
        .await?;
    }

    Ok(())
}
