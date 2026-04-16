// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[path = "common/mod.rs"]
mod common;
use common::*;

use clap::Parser;
use common::NoopSequencePublisher;
use dynamo_kv_router::protocols::WorkerWithDpRank;
use dynamo_kv_router::{ActiveSequencesMultiWorker, SequenceRequest};
use dynamo_mocker::loadgen::Trace;
use dynamo_tokens::SequenceHash;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::time::{Duration, Instant};

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
    traces: &[Trace],
    num_gpu_blocks: usize,
    block_size: u32,
    trace_simulation_duration_ms: u64,
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

    // Keep the post-run drain check out of the measured benchmark interval.
    let total_duration = progress.elapsed();
    multi.assert_completely_drained(Instant::now());

    if total_duration > Duration::from_millis(benchmark_duration_ms * 11 / 10) {
        eprintln!(
            "WARNING: Benchmarker could not keep up. Rerun with a larger --benchmark-duration-ms."
        );
    }
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
        "Ops Throughput: offered={} ops/s achieved={} ops/s (potential_blocks_and_tokens + add + prefill_complete + free)",
        offered_ops_throughput, ops_throughput
    );
    println!(
        "Block Throughput: offered={} block ops/s achieved={} block ops/s",
        offered_block_throughput, block_throughput
    );
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
    let decay_now = tokio::time::Instant::now();
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
                HashMap::new(),
            );
            let _ = multi
                .add_request(SequenceRequest {
                    request_id,
                    token_sequence: Some(block_hashes),
                    isl,
                    cached_tokens: 0,
                    track_prefill_tokens: true,
                    expected_output_tokens: Some(output_length as u32),
                    prefill_load_hint: None,
                    worker,
                    lora_name: None,
                })
                .await;
        }
        SequenceTraceEntry::PrefillComplete { request_id } => {
            let _ = multi.mark_prefill_completed(&request_id, decay_now);
        }
        SequenceTraceEntry::Free { request_id } => {
            let _ = multi.free(&request_id, decay_now);
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
        writeln!(
            f,
            "{}",
            serde_json::json!({
                "session_id": "session-a",
                "timestamp": 0,
                "input_length": 4,
                "hash_ids": [0u64, 1, 2, 3],
                "output_length": 10u64,
            })
        )?;
        writeln!(
            f,
            "{}",
            serde_json::json!({
                "session_id": "session-a",
                "delay": 5.0,
                "input_length": 4,
                "hash_ids": [4u64, 5, 6, 7],
                "output_length": 10u64,
            })
        )?;
    }

    let traces = process_mooncake_trace(path.to_str().unwrap(), 512, 1, 1, 1, 42)?;
    std::fs::remove_file(&path).ok();

    println!(
        "Loaded {} workers, {} total requests",
        traces.len(),
        traces
            .iter()
            .map(|trace| trace
                .sessions
                .iter()
                .map(|session| session.turns.len())
                .sum::<usize>())
            .sum::<usize>()
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
    for trace in &seq_traces {
        assert!(
            trace
                .windows(2)
                .all(|window| window[1].timestamp_us >= window[0].timestamp_us)
        );
    }
    let first_free_us = seq_traces[0]
        .iter()
        .find_map(|entry| match entry.entry {
            SequenceTraceEntry::Free { .. } => Some(entry.timestamp_us),
            _ => None,
        })
        .unwrap();
    let second_add_us = seq_traces[0]
        .iter()
        .filter_map(|entry| match entry.entry {
            SequenceTraceEntry::Add { .. } => Some(entry.timestamp_us),
            _ => None,
        })
        .nth(1)
        .unwrap();
    assert!(second_add_us >= first_free_us);

    println!("All tests passed.");
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    init_sequence_logging(args.common.sequence_logs);

    if args.common.test {
        return run_tests().await;
    }

    let path = match args.common.mooncake_trace_path.as_deref() {
        Some(p) => p,
        None => {
            eprintln!("No mooncake_trace_path provided, skipping benchmark");
            return Ok(());
        }
    };
    let traces = process_mooncake_trace(
        path,
        args.common.block_size,
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
