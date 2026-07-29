// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[path = "active_sequences_open_loop.rs"]
mod active_sequences_open_loop;
#[path = "active_sequences_shared.rs"]
mod active_sequences_shared;

use active_sequences_open_loop::{
    ActiveSequencesResult, ActiveSequencesRunConfig, PreparedActiveSequencesCorpus,
    prepare_active_sequences_corpus, run_active_sequences_benchmark,
};
use active_sequences_shared::generate_sequence_events;
use clap::Parser;
use dynamo_bench::kv_router_common::args::CommonArgs;
use dynamo_bench::kv_router_common::replay::process_mooncake_trace;
use dynamo_bench::kv_router_common::sweep::compute_sweep_durations;
use tracing_subscriber::EnvFilter;

fn init_sequence_logging(enabled: bool) {
    if !enabled {
        return;
    }

    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new(
            "error,dynamo_kv_router::sequences=warn,dynamo_mocker=warn",
        ))
        .with_writer(std::io::stderr)
        .try_init();
}

#[derive(Parser, Debug)]
#[clap(
    version,
    about = "ActiveSequences add_request/free throughput benchmark"
)]
struct Args {
    #[clap(flatten)]
    common: CommonArgs,

    /// Number of persistent sticky operation lanes.
    #[clap(long, default_value = "128")]
    operation_lanes: usize,

    /// Busy-spin interval after the deadline issuer sleeps.
    #[clap(long, default_value = "75")]
    issuer_spin_us: u64,

    /// Diagnostic late-issue threshold; it is not a validity gate.
    #[clap(long, default_value = "50")]
    issue_lag_diagnostic_threshold_us: u64,

    /// JSON output path for one benchmark result.
    #[clap(long, default_value = "active_sequences_result.json")]
    result_json_output: String,
}

fn validate_args(args: &Args) -> anyhow::Result<()> {
    if args.common.test {
        anyhow::bail!(
            "active_sequences_bench no longer supports --test; run `cargo test --package dynamo-bench --test active_sequences_trace` instead"
        );
    }
    if args.operation_lanes == 0 {
        anyhow::bail!("--operation-lanes must be at least 1");
    }
    if args.operation_lanes > u16::MAX as usize {
        anyhow::bail!("--operation-lanes exceeds the u16 lane-ID space");
    }
    Ok(())
}

fn run_config(args: &Args) -> ActiveSequencesRunConfig {
    ActiveSequencesRunConfig {
        operation_lanes: args.operation_lanes,
        spin_us: args.issuer_spin_us,
        issue_lag_diagnostic_threshold_us: args.issue_lag_diagnostic_threshold_us,
    }
}

async fn prepare_benchmark(
    args: &Args,
    benchmark_duration_ms: u64,
) -> anyhow::Result<Option<PreparedActiveSequencesCorpus>> {
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
    let seq_traces = generate_sequence_events(
        &traces,
        args.common.num_gpu_blocks,
        args.common.block_size,
        args.common.trace_simulation_duration_ms,
    )
    .await?;
    drop(traces);

    Ok(Some(prepare_active_sequences_corpus(
        seq_traces,
        args.common.block_size,
        benchmark_duration_ms,
        args.common.inference_worker_duplication_factor,
    )?))
}

fn print_result(result: &ActiveSequencesResult) {
    println!(
        "Logical throughput: offered={:.0} ops/s achieved={:.0} ops/s",
        result.offered_logical_ops_per_sec, result.achieved_logical_ops_per_sec
    );
    println!(
        "Logical block visits: offered={:.0}/s achieved={:.0}/s input_blocks={}",
        result.offered_logical_block_visits_per_sec,
        result.achieved_logical_block_visits_per_sec,
        result.total_input_blocks,
    );
    println!(
        "Service p99 (us): project={:.1} add={:.1} project+add={:.1} prefill_complete={:.1} free={:.1}",
        result.project_service.p99_ns as f64 / 1_000.0,
        result.add_service.p99_ns as f64 / 1_000.0,
        result.project_and_add_service.p99_ns as f64 / 1_000.0,
        result.prefill_complete_service.p99_ns as f64 / 1_000.0,
        result.free_service.p99_ns as f64 / 1_000.0,
    );
    println!(
        "generator_valid={} kept_up={} issue_span={:.3}ms drain={:.3}ms",
        result.generator_valid,
        result.kept_up,
        result.issue_span_ns as f64 / 1e6,
        result.drain_ns as f64 / 1e6,
    );
}

fn result_path(base: &str, duration_ms: Option<u64>) -> String {
    let stem = base.trim_end_matches(".json");
    match duration_ms {
        Some(duration_ms) => format!("{stem}_{duration_ms}ms.json"),
        None => base.to_string(),
    }
}

fn write_result(path: &str, result: &ActiveSequencesResult) -> anyhow::Result<()> {
    std::fs::write(path, serde_json::to_string_pretty(result)?)?;
    println!("Active Sequences result written to {path}");
    Ok(())
}

async fn run_cell(
    args: &Args,
    benchmark_duration_ms: u64,
    output_duration: Option<u64>,
) -> anyhow::Result<()> {
    let Some(corpus) = prepare_benchmark(args, benchmark_duration_ms).await? else {
        return Ok(());
    };
    let result = run_active_sequences_benchmark(corpus, run_config(args)).await?;
    print_result(&result);
    write_result(
        &result_path(&args.result_json_output, output_duration),
        &result,
    )?;
    if !result.kept_up {
        eprintln!(
            "WARNING: Active Sequences replay did not keep up; inspect issue, queue, and drain metrics"
        );
    }
    Ok(())
}

async fn async_main(args: Args) -> anyhow::Result<()> {
    if args.common.sweep {
        let durations = compute_sweep_durations(
            args.common.sweep_min_ms,
            args.common.sweep_max_ms,
            args.common.sweep_steps,
        );
        for duration_ms in durations.into_iter().rev() {
            println!("\n=== Active Sequences sweep: benchmark_duration_ms={duration_ms} ===");
            run_cell(&args, duration_ms, Some(duration_ms)).await?;
        }
        return Ok(());
    }

    run_cell(&args, args.common.benchmark_duration_ms, None).await
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    validate_args(&args)?;
    init_sequence_logging(args.common.sequence_logs);
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(async_main(args))
}
