// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[path = "mooncake_open_loop.rs"]
mod mooncake_open_loop;
#[path = "mooncake_shared.rs"]
mod mooncake_shared;

use clap::{Parser, Subcommand};
use dynamo_bench::kv_router_common::args::CommonArgs;
use dynamo_bench::kv_router_common::issuer::pin_current_thread_to_cpus;
use dynamo_bench::kv_router_common::replay::{generate_replay_artifacts, process_mooncake_trace};
use dynamo_bench::kv_router_common::sweep::compute_sweep_durations;
use dynamo_kv_router::indexer::KvIndexerMetrics;
use dynamo_kv_router::{
    ConcurrentRadixTree, ConcurrentRadixTreeCompressed, PositionalIndexer, ThreadPoolIndexer,
};
use mooncake_open_loop::{
    OpenLoopConfig, OpenLoopResult, parse_cpu_list, prepare_mooncake_corpus,
    prepare_open_loop_trial, run_open_loop, validate_cpu_partition,
};
use mooncake_shared::{
    MooncakeBenchmarkConfig, MooncakeIndexerConfig, MooncakeIndexerKind, PreparedMooncakeBenchmark,
    merge_worker_traces, prepare_scaled_benchmark,
};
use std::sync::Arc;

#[cfg(target_os = "linux")]
const PRE_RUN_QUIESCENCE_MS: u64 = 5_000;
#[cfg(not(target_os = "linux"))]
const PRE_RUN_QUIESCENCE_MS: u64 = 0;

/// Indexer backend selection and its backend-specific parameters.
#[derive(Subcommand, Debug, Clone)]
enum IndexerArgs {
    /// Single-threaded radix tree indexer.
    RadixTree {},

    /// Position-based nested map indexer with jump search.
    NestedMap {
        /// Number of positions to skip during jump search before scanning back.
        #[clap(long, default_value = "8")]
        jump_size: usize,

        /// Number of OS threads that consume and apply KV cache events.
        #[clap(long, default_value = "16")]
        num_event_workers: usize,
    },

    /// Lock-based concurrent radix tree indexer.
    ConcurrentRadixTree {
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

    /// Branch-sharded CRTC: N independent CRTC shards routed by a bounded
    /// prefix trie with structural anchors for depth-boundary suffixes.
    /// find_matches touches at most one shard (no scatter-gather).
    BranchShardedCrtc {
        /// Number of independent CRTC shards.
        #[clap(long, default_value = "2")]
        num_shards: usize,

        /// Number of OS event-worker threads per shard.
        #[clap(long, default_value = "4")]
        num_event_workers_per_shard: usize,

        /// Maximum routing-trie depth before dispatching suffixes to one shard.
        /// K=2 is the recommended default: depth=1 often produces too few
        /// distinct branches, while depth=2 exposes more branch diversity.
        #[clap(long, default_value = "2")]
        prefix_depth: usize,
    },
}

impl IndexerArgs {
    fn to_config(&self) -> MooncakeIndexerConfig {
        match self {
            IndexerArgs::RadixTree {} => MooncakeIndexerConfig::radix_tree(),
            IndexerArgs::NestedMap {
                jump_size,
                num_event_workers,
            } => MooncakeIndexerConfig::nested_map(*jump_size, *num_event_workers),
            IndexerArgs::ConcurrentRadixTree { num_event_workers } => {
                MooncakeIndexerConfig::concurrent_radix_tree(*num_event_workers)
            }
            IndexerArgs::ConcurrentRadixTreeCompressed { num_event_workers } => {
                MooncakeIndexerConfig::concurrent_radix_tree_compressed(*num_event_workers)
            }
            IndexerArgs::BranchShardedCrtc {
                num_shards,
                num_event_workers_per_shard,
                prefix_depth,
            } => MooncakeIndexerConfig::branch_sharded_crtc(
                *num_shards,
                *num_event_workers_per_shard,
                *prefix_depth,
            ),
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
    /// radix-tree, nested-map, concurrent-radix-tree,
    /// concurrent-radix-tree-compressed, branch-sharded-crtc.
    #[clap(long, value_delimiter = ',')]
    compare: Vec<String>,

    /// Number of OS threads for event processing in compare mode. Applies to
    /// indexers that use a thread pool (nested-map, concurrent-radix-tree,
    /// concurrent-radix-tree-compressed, branch-sharded-crtc).
    /// Ignored by radix-tree.
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

    /// Indexer backend to benchmark (defaults to radix-tree if not specified).
    #[clap(subcommand)]
    indexer: Option<IndexerArgs>,
}

impl Args {
    /// Return the indexer config, falling back to RadixTree if none was specified.
    fn get_indexer(&self) -> IndexerArgs {
        self.indexer.clone().unwrap_or(IndexerArgs::RadixTree {})
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
                | MooncakeIndexerKind::ConcurrentRadixTree
                | MooncakeIndexerKind::ConcurrentRadixTreeCompressed
        ) {
            anyhow::bail!(
                "corrected Mooncake replay supports only nested-map, concurrent-radix-tree, and concurrent-radix-tree-compressed; got {name}"
            );
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
    })
}

async fn run_open_loop_for_config(
    args: &Args,
    config: &MooncakeIndexerConfig,
    prepared: PreparedMooncakeBenchmark,
    bench_config: MooncakeBenchmarkConfig,
) -> anyhow::Result<OpenLoopResult> {
    if config.num_event_workers > u16::MAX as usize {
        anyhow::bail!("Mooncake event-worker count exceeds the u16 queue-ID space");
    }
    let corpus =
        prepare_mooncake_corpus(prepared, bench_config.inference_worker_duplication_factor)?;
    let trial = prepare_open_loop_trial(corpus, args.query_lanes)?;
    quiesce_prepared_heap();
    let metrics = || Some(Arc::new(KvIndexerMetrics::new_unregistered()));
    let open_config = parse_open_loop_config(args)?;
    pin_current_thread_to_cpus(&open_config.backend_cpus)?;

    match config.kind {
        MooncakeIndexerKind::NestedMap => {
            let indexer = Arc::new(ThreadPoolIndexer::new_with_metrics(
                PositionalIndexer::new(config.jump_size),
                config.num_event_workers,
                args.common.block_size,
                metrics(),
            ));
            run_backend(config.short_name(), indexer, trial, open_config).await
        }
        MooncakeIndexerKind::ConcurrentRadixTree => {
            let indexer = Arc::new(ThreadPoolIndexer::new_with_metrics(
                ConcurrentRadixTree::new(),
                config.num_event_workers,
                args.common.block_size,
                metrics(),
            ));
            run_backend(config.short_name(), indexer, trial, open_config).await
        }
        MooncakeIndexerKind::ConcurrentRadixTreeCompressed => {
            let indexer = Arc::new(ThreadPoolIndexer::new_with_metrics(
                ConcurrentRadixTreeCompressed::new(),
                config.num_event_workers,
                args.common.block_size,
                metrics(),
            ));
            run_backend(config.short_name(), indexer, trial, open_config).await
        }
        MooncakeIndexerKind::RadixTree | MooncakeIndexerKind::BranchShardedCrtc => {
            anyhow::bail!(
                "{} is not supported by corrected Mooncake replay",
                config.short_name()
            )
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
    if let Some(cpu) = open_config.query_issuer_cpu {
        pin_current_thread_to_cpus(&[cpu])?;
    }
    run_open_loop(backend_name, indexer, trial, open_config).await
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

fn benchmark_config(args: &Args, benchmark_duration_ms: u64) -> MooncakeBenchmarkConfig {
    MooncakeBenchmarkConfig {
        benchmark_duration_ms,
        inference_worker_duplication_factor: args.common.inference_worker_duplication_factor,
    }
}

async fn prepare_benchmark(
    args: &Args,
    benchmark_duration_ms: u64,
) -> anyhow::Result<Option<PreparedMooncakeBenchmark>> {
    let Some(path) = args.common.mooncake_trace_path.as_deref() else {
        eprintln!("No mooncake_trace_path provided, skipping benchmark");
        return Ok(None);
    };

    let traces = process_mooncake_trace(
        path,
        args.common.block_size,
        args.common.trace_length_factor,
        args.common.trace_duplication_factor,
        args.common.num_unique_inference_workers,
        args.common.seed,
    )?;
    let artifacts = generate_replay_artifacts(
        &traces,
        args.common.num_gpu_blocks,
        args.common.block_size,
        args.common.trace_simulation_duration_ms,
    )
    .await?;
    drop(traces);
    let merged = merge_worker_traces(artifacts, args.common.block_size)?;
    Ok(Some(prepare_scaled_benchmark(
        merged,
        benchmark_duration_ms,
    )))
}

async fn run_open_loop_repeated_mode(args: &Args, indexer_names: &[String]) -> anyhow::Result<()> {
    for name in indexer_names {
        let config = indexer_config(args, name)?;
        let bench_config = benchmark_config(args, args.common.benchmark_duration_ms);
        let Some(prepared) = prepare_benchmark(args, bench_config.benchmark_duration_ms).await?
        else {
            return Ok(());
        };
        let result = run_open_loop_for_config(args, &config, prepared, bench_config).await?;
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
    );

    for name in indexer_names {
        let config = indexer_config(args, name)?;
        for &duration_ms in durations.iter().rev() {
            println!(
                "\n=== Mooncake sweep: backend={} benchmark_duration_ms={} ===",
                config.short_name(),
                duration_ms
            );
            let bench_config = benchmark_config(args, duration_ms);
            let Some(prepared) =
                prepare_benchmark(args, bench_config.benchmark_duration_ms).await?
            else {
                return Ok(());
            };
            let result = run_open_loop_for_config(args, &config, prepared, bench_config).await?;
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

    if args.common.sweep {
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
    runtime.enable_all();
    if !config.backend_cpus.is_empty() {
        pin_current_thread_to_cpus(&config.backend_cpus)?;
        runtime.worker_threads(config.backend_cpus.len());
    }

    let runtime = runtime.build()?;
    runtime.block_on(async_main(args))
}
