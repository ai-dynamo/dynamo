// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[path = "mooncake_open_loop.rs"]
mod mooncake_open_loop;
#[path = "mooncake_shared.rs"]
mod mooncake_shared;
#[path = "scaling_diag.rs"]
mod scaling_diag;

use clap::{Parser, Subcommand};
use dynamo_bench::kv_router_common::args::CommonArgs;
use dynamo_bench::kv_router_common::issuer::pin_current_thread_to_cpus;
use dynamo_bench::kv_router_common::replay::generate_replay_artifacts;
use dynamo_bench::kv_router_common::sweep::compute_sweep_durations;
use dynamo_kv_router::indexer::KvIndexerMetrics;
use dynamo_kv_router::{ConcurrentRadixTreeCompressed, PositionalIndexer, ThreadPoolIndexer};
use mooncake_open_loop::{
    OpenLoopConfig, OpenLoopResult, RunProvenance, parse_cpu_list, prepare_mooncake_corpus,
    prepare_open_loop_trial, run_correctness_check, run_open_loop, validate_cpu_partition,
};
use mooncake_shared::{
    MooncakeBenchmarkConfig, MooncakeIndexerConfig, MooncakeIndexerKind, PreparedMooncakeBenchmark,
    merge_worker_traces, prepare_scaled_benchmark,
};
use scaling_diag::{
    CorrectnessReport, NullIndexer, PrepTimings, THREAD_NAME_EVENT, THREAD_NAME_TOKIO,
    current_thread_name, set_current_thread_name, workload_diagnostics,
};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// CRTC coverage slots are u16, so one tree holds at most this many ranks.
const MAX_CRTC_RANKS: usize = 1 << 16;

#[cfg(target_os = "linux")]
const PRE_RUN_QUIESCENCE_MS: u64 = 5_000;
#[cfg(not(target_os = "linux"))]
const PRE_RUN_QUIESCENCE_MS: u64 = 0;

/// Indexer backend selection and its backend-specific parameters.
#[derive(Subcommand, Debug, Clone)]
enum IndexerArgs {
    /// Position-based nested map indexer with jump search.
    NestedMap {
        /// Number of positions to skip during jump search before scanning back.
        #[clap(long, default_value = "8")]
        jump_size: usize,

        /// Number of OS threads that consume and apply KV cache events.
        #[clap(long, default_value = "16")]
        num_event_workers: usize,
    },

    /// Compressed concurrent radix tree indexer (compressed edges).
    ConcurrentRadixTreeCompressed {
        /// Number of OS threads that consume and apply KV cache events.
        #[clap(long, default_value = "16")]
        num_event_workers: usize,
    },

    /// Harness-ceiling control: accepts every event and returns no matches, with the same
    /// event threads, query lanes, and issuers as a real backend.
    Null {
        /// Number of OS threads that consume KV cache events.
        #[clap(long, default_value = "16")]
        num_event_workers: usize,
    },
}

impl IndexerArgs {
    fn to_config(&self) -> MooncakeIndexerConfig {
        match self {
            IndexerArgs::NestedMap {
                jump_size,
                num_event_workers,
            } => MooncakeIndexerConfig::nested_map(*jump_size, *num_event_workers),
            IndexerArgs::ConcurrentRadixTreeCompressed { num_event_workers } => {
                MooncakeIndexerConfig::concurrent_radix_tree_compressed(*num_event_workers)
            }
            IndexerArgs::Null { num_event_workers } => {
                MooncakeIndexerConfig::null(*num_event_workers)
            }
        }
    }
}

#[derive(Parser, Debug)]
#[clap(version, about, long_about = None)]
struct Args {
    #[clap(flatten)]
    common: CommonArgs,

    /// Number of persistent logical query lanes.
    #[clap(long, default_value = "128")]
    query_lanes: usize,

    /// Number of native event-issuer threads.
    #[clap(long, default_value = "8")]
    issuer_threads: usize,

    /// Busy-spin interval after the absolute issuer sleep.
    #[clap(long, default_value = "75")]
    issuer_spin_us: u64,

    /// Diagnostic threshold for reporting late issue operations. It is not a validity gate.
    #[clap(long, default_value = "50")]
    issue_lag_diagnostic_threshold_us: u64,

    /// Comma-separated logical CPUs or ranges for parallel deadline issuers.
    #[clap(long)]
    issuer_cpus: Option<String>,

    /// Logical CPU for the timed query issuer. The parent coordinator is parked
    /// on this CPU while scoped issuer threads run.
    #[clap(long)]
    query_issuer_cpu: Option<usize>,

    /// Comma-separated logical CPUs or ranges used by query and event workers.
    #[clap(long)]
    backend_cpus: Option<String>,

    /// JSON output path for a benchmark result.
    #[clap(long, default_value = "mooncake_result.json")]
    result_json_output: String,

    /// Comma-separated list of indexer names to benchmark and compare on the
    /// same plot. Overrides the subcommand indexer when present. Valid names:
    /// nested-map, concurrent-radix-tree-compressed.
    #[clap(long, value_delimiter = ',')]
    compare: Vec<String>,

    /// Number of OS threads for event processing with `--compare` or when no
    /// subcommand is given (the default concurrent-radix-tree-compressed run).
    #[clap(long, default_value = "16")]
    num_event_workers: usize,

    /// Number of additional concurrent tokio tasks that issue find_matches in a
    /// tight loop to stress the read path.  These tasks run alongside the normal
    /// trace-replay workers.  Set to 0 (default) to disable.
    #[clap(long, default_value = "0")]
    find_matches_concurrency: usize,

    /// Use approximate routing-decision writes instead of offline-generated KV events.
    #[clap(long)]
    approx: bool,

    /// Number of independent benchmark trials to run over the same generated
    /// benchmark input. Each trial builds a fresh indexer.
    #[clap(long, default_value = "1")]
    benchmark_runs: usize,

    /// Prepare the corpus, write its per-worker workload shape to the result path, and exit
    /// without building an indexer or timing anything.
    #[clap(long)]
    prep_only: bool,

    /// Instead of a timed run, replay the corpus quiescently and compare the backend's scores
    /// with an independent reference index for about this many evenly spaced queries.
    #[clap(long, default_value = "0")]
    correctness_check_queries: usize,

    /// Skip registering every rank before the measured window.
    #[clap(long)]
    no_pre_register_ranks: bool,

    /// Idle keep-alive for Tokio blocking threads (corpus generation), so they exit during
    /// pre-run quiescence instead of inside the measured window.
    #[clap(long, default_value = "500")]
    blocking_thread_keep_alive_ms: u64,

    /// Harness guard: maximum p99 read and update issue lag.
    #[clap(long, default_value = "250")]
    guard_lag_p99_us: u64,

    /// Harness guard: every issuer's active fraction must stay below this value.
    #[clap(long, default_value = "0.7")]
    guard_issuer_active_fraction: f64,

    /// Indexer backend to benchmark. Defaults to concurrent-radix-tree-compressed
    /// with `--num-event-workers` event threads.
    #[clap(subcommand)]
    indexer: Option<IndexerArgs>,
}

impl Args {
    fn get_indexer(&self) -> IndexerArgs {
        self.indexer
            .clone()
            .unwrap_or(IndexerArgs::ConcurrentRadixTreeCompressed {
                num_event_workers: self.num_event_workers,
            })
    }
}

fn validate_args(args: &Args) -> anyhow::Result<()> {
    if args.common.test {
        anyhow::bail!(
            "mooncake_bench no longer supports --test; run `cargo test --package dynamo-bench --test mooncake_trace` instead"
        );
    }
    if args.benchmark_runs == 0 {
        anyhow::bail!("--benchmark-runs must be at least 1");
    }
    if args.common.sweep && args.benchmark_runs != 1 {
        anyhow::bail!("--benchmark-runs is only supported outside --sweep mode");
    }
    if args.query_lanes == 0 {
        anyhow::bail!("--query-lanes must be at least 1");
    }
    if args.issuer_threads == 0 {
        anyhow::bail!("--issuer-threads must be at least 1");
    }
    if args.num_event_workers > u16::MAX as usize {
        anyhow::bail!("--num-event-workers exceeds the u16 queue-ID space");
    }
    if args.approx {
        anyhow::bail!("corrected Mooncake replay does not support --approx");
    }
    if args.find_matches_concurrency != 0 {
        anyhow::bail!("corrected Mooncake replay does not support --find-matches-concurrency");
    }
    if !args.common.sweep && args.benchmark_runs != 1 {
        anyhow::bail!("repetitions must use fresh processes; invoke one trial per process");
    }
    let ranks = args
        .common
        .num_unique_inference_workers
        .saturating_mul(args.common.inference_worker_duplication_factor);
    if ranks > MAX_CRTC_RANKS {
        anyhow::bail!("{ranks} ranks exceed the CRTC slot capacity of {MAX_CRTC_RANKS}");
    }
    if (args.prep_only || args.correctness_check_queries > 0) && args.common.sweep {
        anyhow::bail!("--prep-only and --correctness-check-queries do not support --sweep");
    }
    if args.common.mooncake_trace_path.is_none() {
        return Ok(());
    }

    for name in indexer_names(args) {
        let config = if args.compare.is_empty() {
            args.get_indexer().to_config()
        } else {
            MooncakeIndexerConfig::from_short_name(&name, args.num_event_workers)?
        };
        if !matches!(
            config.kind,
            MooncakeIndexerKind::NestedMap
                | MooncakeIndexerKind::ConcurrentRadixTreeCompressed
                | MooncakeIndexerKind::Null
        ) {
            anyhow::bail!(
                "corrected Mooncake replay supports only nested-map, concurrent-radix-tree-compressed, and null; got {name}"
            );
        }
        if config.kind == MooncakeIndexerKind::Null && args.correctness_check_queries > 0 {
            anyhow::bail!("the null backend returns no matches; it has no score check");
        }
    }
    Ok(())
}

fn indexer_names(args: &Args) -> Vec<String> {
    if args.compare.is_empty() {
        vec![args.get_indexer().to_config().short_name().to_string()]
    } else {
        args.compare.clone()
    }
}

fn indexer_config(args: &Args, name: &str) -> anyhow::Result<MooncakeIndexerConfig> {
    if args.compare.is_empty() {
        Ok(args.get_indexer().to_config())
    } else {
        MooncakeIndexerConfig::from_short_name(name, args.num_event_workers)
    }
}

fn open_loop_config(args: &Args) -> anyhow::Result<OpenLoopConfig> {
    let config = parse_open_loop_config(args)?;
    validate_cpu_partition(
        &config.issuer_cpus,
        config.query_issuer_cpu,
        &config.backend_cpus,
    )?;
    Ok(config)
}

fn parse_open_loop_config(args: &Args) -> anyhow::Result<OpenLoopConfig> {
    let backend_cpus = args
        .backend_cpus
        .as_deref()
        .map(parse_cpu_list)
        .transpose()?
        .unwrap_or_default();
    let issuer_cpus = args
        .issuer_cpus
        .as_deref()
        .map(parse_cpu_list)
        .transpose()?
        .unwrap_or_default();
    let issuer_threads = if issuer_cpus.is_empty() {
        args.issuer_threads
    } else {
        issuer_cpus.len()
    };
    Ok(OpenLoopConfig {
        query_lanes: args.query_lanes,
        issuer_threads,
        spin_us: args.issuer_spin_us,
        issue_lag_diagnostic_threshold_us: args.issue_lag_diagnostic_threshold_us,
        pre_run_quiescence_ms: PRE_RUN_QUIESCENCE_MS,
        issuer_cpus,
        query_issuer_cpu: args.query_issuer_cpu,
        backend_cpus,
        pre_register_ranks: !args.no_pre_register_ranks,
        guard_lag_p99_us: args.guard_lag_p99_us,
        guard_issuer_active_fraction: args.guard_issuer_active_fraction,
    })
}

/// Build an indexer with its event threads named for per-thread CPU accounting. The threads
/// are spawned inside `dynamo-kv-router` and inherit the creating thread's name.
fn with_event_thread_name<R>(build: impl FnOnce() -> R) -> R {
    let original = current_thread_name();
    set_current_thread_name(THREAD_NAME_EVENT);
    let built = build();
    set_current_thread_name(&original);
    built
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1e3
}

async fn run_open_loop_for_config(
    args: &Args,
    config: &MooncakeIndexerConfig,
    prepared: PreparedMooncakeBenchmark,
    bench_config: MooncakeBenchmarkConfig,
    mut timings: PrepTimings,
) -> anyhow::Result<OpenLoopResult> {
    if config.num_event_workers > u16::MAX as usize {
        anyhow::bail!("Mooncake event-worker count exceeds the u16 queue-ID space");
    }
    let workload = workload_diagnostics(&prepared, args.common.num_gpu_blocks);
    let started = Instant::now();
    let corpus =
        prepare_mooncake_corpus(prepared, bench_config.inference_worker_duplication_factor)?;
    let trial = prepare_open_loop_trial(corpus, args.query_lanes)?;
    timings.corpus_and_dispatch_ms = elapsed_ms(started);
    let started = Instant::now();
    quiesce_prepared_heap();
    timings.quiescence_ms = elapsed_ms(started);
    let metrics = || Some(Arc::new(KvIndexerMetrics::new_unregistered()));
    let open_config = parse_open_loop_config(args)?;
    pin_current_thread_to_cpus(&open_config.backend_cpus)?;

    let mut result = match config.kind {
        MooncakeIndexerKind::NestedMap => {
            let indexer = with_event_thread_name(|| {
                Arc::new(ThreadPoolIndexer::new_with_metrics(
                    PositionalIndexer::new(config.jump_size),
                    config.num_event_workers,
                    args.common.block_size,
                    metrics(),
                ))
            });
            run_backend(config.short_name(), indexer, trial, open_config).await
        }
        MooncakeIndexerKind::ConcurrentRadixTreeCompressed => {
            let indexer = with_event_thread_name(|| {
                Arc::new(ThreadPoolIndexer::new_with_metrics(
                    ConcurrentRadixTreeCompressed::new(),
                    config.num_event_workers,
                    args.common.block_size,
                    metrics(),
                ))
            });
            run_backend(config.short_name(), indexer, trial, open_config).await
        }
        MooncakeIndexerKind::Null => {
            let indexer = with_event_thread_name(|| {
                Arc::new(ThreadPoolIndexer::new_with_metrics(
                    NullIndexer::default(),
                    config.num_event_workers,
                    args.common.block_size,
                    metrics(),
                ))
            });
            run_backend(config.short_name(), indexer, trial, open_config).await
        }
        MooncakeIndexerKind::RadixTree | MooncakeIndexerKind::BranchShardedCrtc => {
            anyhow::bail!(
                "{} is not supported by corrected Mooncake replay",
                config.short_name()
            )
        }
    }?;
    result.prep = Some(timings);
    result.workload = Some(workload);
    Ok(result)
}

async fn run_correctness_for_config(
    args: &Args,
    config: &MooncakeIndexerConfig,
    prepared: PreparedMooncakeBenchmark,
    bench_config: MooncakeBenchmarkConfig,
) -> anyhow::Result<CorrectnessReport> {
    let corpus =
        prepare_mooncake_corpus(prepared, bench_config.inference_worker_duplication_factor)?;
    let backend_cpus = parse_open_loop_config(args)?.backend_cpus;
    pin_current_thread_to_cpus(&backend_cpus)?;
    let pre_register = !args.no_pre_register_ranks;
    let checked = args.correctness_check_queries;
    match config.kind {
        MooncakeIndexerKind::NestedMap => {
            let indexer = Arc::new(ThreadPoolIndexer::new(
                PositionalIndexer::new(config.jump_size),
                config.num_event_workers,
                args.common.block_size,
            ));
            run_correctness_check(config.short_name(), indexer, corpus, checked, pre_register).await
        }
        MooncakeIndexerKind::ConcurrentRadixTreeCompressed => {
            let indexer = Arc::new(ThreadPoolIndexer::new(
                ConcurrentRadixTreeCompressed::new(),
                config.num_event_workers,
                args.common.block_size,
            ));
            run_correctness_check(config.short_name(), indexer, corpus, checked, pre_register).await
        }
        MooncakeIndexerKind::RadixTree
        | MooncakeIndexerKind::BranchShardedCrtc
        | MooncakeIndexerKind::Null => {
            anyhow::bail!("{} has no score check", config.short_name())
        }
    }
}

#[cfg(target_os = "linux")]
fn quiesce_prepared_heap() {
    // Corpus construction releases large, multi-threaded preparation arenas.
    // Return their free pages and let reclamation settle before backend workers start.
    unsafe {
        libc::malloc_trim(0);
    }
    std::thread::sleep(std::time::Duration::from_millis(PRE_RUN_QUIESCENCE_MS));
}

#[cfg(not(target_os = "linux"))]
fn quiesce_prepared_heap() {}

async fn run_backend<T: dynamo_kv_router::indexer::SyncIndexer>(
    backend_name: &str,
    indexer: Arc<ThreadPoolIndexer<T>>,
    trial: mooncake_open_loop::PreparedOpenLoopTrial,
    open_config: OpenLoopConfig,
) -> anyhow::Result<OpenLoopResult> {
    let coordinator_cpus = open_config.backend_cpus.clone();
    if let Some(cpu) = open_config.query_issuer_cpu {
        pin_current_thread_to_cpus(&[cpu])?;
    }
    let result = run_open_loop(backend_name, indexer, trial, open_config).await;
    // Restore the coordinator mask. Otherwise blocking-pool threads spawned while the
    // next sweep or compare cell generates events inherit the single query-issuer CPU.
    let restored = pin_current_thread_to_cpus(&coordinator_cpus);
    let result = result?;
    restored?;
    Ok(result)
}

fn print_scaling_summary(result: &OpenLoopResult) {
    let scaling = &result.scaling;
    println!(
        "Matched ranks per query mean/p99/max: {:.1}/{}/{} | >256: {}",
        scaling.matched_ranks_per_query.mean,
        scaling.matched_ranks_per_query.p99,
        scaling.matched_ranks_per_query.max,
        scaling.queries_with_more_than_256_matches,
    );
    println!(
        "Registration: {} ranks in {:.1} ms | threads at window start: {} | max issuer active: {:.3} | failed events: {} | guard pass: {}",
        scaling.registration.ranks,
        scaling.registration.elapsed_ms,
        scaling.threads.threads_at_window_start,
        scaling.harness_guard.max_issuer_active_fraction,
        scaling.failed_events,
        scaling.harness_guard.pass,
    );
}

fn print_open_loop_result(result: &OpenLoopResult) {
    println!(
        "Offered logical throughput: {:.0} ops/s | achieved: {:.0} ops/s",
        result.offered_logical_ops_per_sec, result.achieved_logical_ops_per_sec
    );
    println!(
        "Offered block throughput: {:.0} block ops/s | achieved: {:.0} block ops/s",
        result.offered_block_ops_per_sec, result.achieved_block_ops_per_sec
    );
    println!(
        "Query service p50/p99: {:.1}/{:.1} us | queue p99: {:.1} us",
        result.query_service.p50_ns as f64 / 1_000.0,
        result.query_service.p99_ns as f64 / 1_000.0,
        result.query_queue_wait.p99_ns as f64 / 1_000.0,
    );
    println!(
        "generator_valid={} kept_up={} issue_span={:.3} ms drain={:.3} ms",
        result.generator_valid,
        result.kept_up,
        result.issue_span_ns as f64 / 1e6,
        result.drain_ns as f64 / 1e6,
    );
    print_scaling_summary(result);
    if !result.backend_timing_report.is_empty() {
        println!("{}", result.backend_timing_report);
    }
}

fn open_loop_output_path(base: &str, backend: &str, duration_ms: Option<u64>) -> String {
    let stem = base.trim_end_matches(".json");
    match duration_ms {
        Some(duration_ms) => format!("{stem}_{backend}_{duration_ms}ms.json"),
        None => format!("{stem}_{backend}.json"),
    }
}

fn write_open_loop_result(path: &str, result: &OpenLoopResult) -> anyhow::Result<()> {
    std::fs::write(path, serde_json::to_string_pretty(result)?)?;
    println!("Mooncake result written to {path}");
    Ok(())
}

fn run_provenance(args: &Args, config: &MooncakeIndexerConfig) -> anyhow::Result<RunProvenance> {
    let common = &args.common;
    let file_sha256 = |path: &std::path::Path| -> anyhow::Result<String> {
        let mut hasher = Sha256::new();
        std::io::copy(&mut std::fs::File::open(path)?, &mut hasher)?;
        Ok(format!("{:x}", hasher.finalize()))
    };
    let trace_sha256 = common
        .mooncake_trace_path
        .as_deref()
        .map(|path| file_sha256(std::path::Path::new(path)))
        .transpose()?;
    let binary = std::env::current_exe().ok();
    let binary_sha256 = binary.as_deref().map(file_sha256).transpose()?;
    Ok(RunProvenance {
        argv: std::env::args().collect(),
        binary: binary.map(|path| path.display().to_string()),
        binary_sha256,
        trace_path: common.mooncake_trace_path.clone(),
        trace_sha256,
        trace_block_size: common.trace_block_size,
        num_gpu_blocks: common.num_gpu_blocks,
        num_unique_inference_workers: common.num_unique_inference_workers,
        inference_worker_duplication_factor: common.inference_worker_duplication_factor,
        trace_length_factor: common.trace_length_factor,
        trace_duplication_factor: common.trace_duplication_factor,
        trace_simulation_duration_ms: common.trace_simulation_duration_ms,
        seed: common.seed,
        jump_size: matches!(config.kind, MooncakeIndexerKind::NestedMap)
            .then_some(config.jump_size),
        issuer_spin_us: args.issuer_spin_us,
        issue_lag_diagnostic_threshold_us: args.issue_lag_diagnostic_threshold_us,
    })
}

fn benchmark_config(args: &Args, benchmark_duration_ms: u64) -> MooncakeBenchmarkConfig {
    MooncakeBenchmarkConfig {
        benchmark_duration_ms,
        inference_worker_duplication_factor: args.common.inference_worker_duplication_factor,
    }
}

async fn prepare_benchmark(
    args: &Args,
    benchmark_duration_ms: u64,
) -> anyhow::Result<Option<(PreparedMooncakeBenchmark, PrepTimings)>> {
    let Some(path) = args.common.mooncake_trace_path.as_deref() else {
        eprintln!("No mooncake_trace_path provided, skipping benchmark");
        return Ok(None);
    };

    let mut timings = PrepTimings::default();
    let started = Instant::now();
    let traces = args.common.load_mooncake_trace(path)?;
    timings.trace_load_ms = elapsed_ms(started);
    let started = Instant::now();
    let artifacts = generate_replay_artifacts(
        &traces,
        args.common.num_gpu_blocks,
        args.common.block_size,
        args.common.trace_simulation_duration_ms,
    )
    .await?;
    drop(traces);
    timings.simulation_ms = elapsed_ms(started);
    let started = Instant::now();
    let merged = merge_worker_traces(artifacts, args.common.block_size)?;
    let prepared = prepare_scaled_benchmark(merged, benchmark_duration_ms);
    timings.merge_and_rescale_ms = elapsed_ms(started);
    Ok(Some((prepared, timings)))
}

#[derive(serde::Serialize)]
struct PrepOnlyReport {
    mode: &'static str,
    prep: PrepTimings,
    workload: scaling_diag::WorkloadDiagnostics,
    provenance: RunProvenance,
}

async fn run_prep_only_mode(args: &Args, indexer_names: &[String]) -> anyhow::Result<()> {
    let name = indexer_names.first().map(String::as_str).unwrap_or("null");
    let config = indexer_config(args, name)?;
    let provenance = run_provenance(args, &config)?;
    let Some((prepared, prep)) = prepare_benchmark(args, args.common.benchmark_duration_ms).await?
    else {
        return Ok(());
    };
    let report = PrepOnlyReport {
        mode: "prep_only",
        prep,
        workload: workload_diagnostics(&prepared, args.common.num_gpu_blocks),
        provenance,
    };
    let json = serde_json::to_string_pretty(&report)?;
    println!("{json}");
    std::fs::write(&args.result_json_output, json)?;
    Ok(())
}

async fn run_correctness_mode(args: &Args, indexer_names: &[String]) -> anyhow::Result<()> {
    for name in indexer_names {
        let config = indexer_config(args, name)?;
        let bench_config = benchmark_config(args, args.common.benchmark_duration_ms);
        let Some((prepared, _)) =
            prepare_benchmark(args, bench_config.benchmark_duration_ms).await?
        else {
            return Ok(());
        };
        let report = run_correctness_for_config(args, &config, prepared, bench_config).await?;
        println!(
            "Correctness {}: workers={} checked={} mismatches={} pass={}",
            report.backend, report.workers, report.checked_queries, report.mismatches, report.pass
        );
        let path = if indexer_names.len() == 1 {
            args.result_json_output.clone()
        } else {
            open_loop_output_path(&args.result_json_output, config.short_name(), None)
        };
        std::fs::write(&path, serde_json::to_string_pretty(&report)?)?;
        println!("Correctness report written to {path}");
    }
    Ok(())
}

async fn run_open_loop_repeated_mode(args: &Args, indexer_names: &[String]) -> anyhow::Result<()> {
    for name in indexer_names {
        let config = indexer_config(args, name)?;
        // Record provenance before the run so it describes the inputs actually read.
        let provenance = run_provenance(args, &config)?;
        let bench_config = benchmark_config(args, args.common.benchmark_duration_ms);
        let Some((prepared, timings)) =
            prepare_benchmark(args, bench_config.benchmark_duration_ms).await?
        else {
            return Ok(());
        };
        let mut result =
            run_open_loop_for_config(args, &config, prepared, bench_config, timings).await?;
        result.provenance = Some(provenance);
        print_open_loop_result(&result);
        let path = if indexer_names.len() == 1 {
            args.result_json_output.clone()
        } else {
            open_loop_output_path(&args.result_json_output, config.short_name(), None)
        };
        write_open_loop_result(&path, &result)?;
    }
    Ok(())
}

async fn run_open_loop_sweep_mode(args: &Args, indexer_names: &[String]) -> anyhow::Result<()> {
    let durations = compute_sweep_durations(
        args.common.sweep_min_ms,
        args.common.sweep_max_ms,
        args.common.sweep_steps,
    )?;

    for name in indexer_names {
        let config = indexer_config(args, name)?;
        let provenance = run_provenance(args, &config)?;
        for &duration_ms in durations.iter().rev() {
            println!(
                "\n=== Mooncake sweep: backend={} benchmark_duration_ms={} ===",
                config.short_name(),
                duration_ms
            );
            let bench_config = benchmark_config(args, duration_ms);
            let Some((prepared, timings)) =
                prepare_benchmark(args, bench_config.benchmark_duration_ms).await?
            else {
                return Ok(());
            };
            let mut result =
                run_open_loop_for_config(args, &config, prepared, bench_config, timings).await?;
            result.provenance = Some(provenance.clone());
            print_open_loop_result(&result);
            let path = open_loop_output_path(
                &args.result_json_output,
                config.short_name(),
                Some(duration_ms),
            );
            write_open_loop_result(&path, &result)?;
        }
    }
    Ok(())
}

async fn async_main(args: Args) -> anyhow::Result<()> {
    let indexer_names = indexer_names(&args);

    if args.prep_only {
        run_prep_only_mode(&args, &indexer_names).await?;
    } else if args.correctness_check_queries > 0 {
        run_correctness_mode(&args, &indexer_names).await?;
    } else if args.common.sweep {
        run_open_loop_sweep_mode(&args, &indexer_names).await?;
    } else {
        run_open_loop_repeated_mode(&args, &indexer_names).await?;
    }

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    validate_args(&args)?;
    let config = open_loop_config(&args)?;

    let mut runtime = tokio::runtime::Builder::new_multi_thread();
    runtime
        .enable_all()
        .thread_name(THREAD_NAME_TOKIO)
        .thread_keep_alive(Duration::from_millis(args.blocking_thread_keep_alive_ms));
    if !config.backend_cpus.is_empty() {
        pin_current_thread_to_cpus(&config.backend_cpus)?;
        // One simulation thread per backend CPU is as fast as the 512-thread default and
        // leaves fewer threads to retire before the window.
        runtime
            .worker_threads(config.backend_cpus.len())
            .max_blocking_threads(config.backend_cpus.len());
    }

    let runtime = runtime.build()?;
    runtime.block_on(async_main(args))
}
