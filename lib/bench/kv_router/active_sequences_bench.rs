// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[path = "jemalloc.rs"]
mod jemalloc;

#[path = "active_sequences_open_loop.rs"]
mod active_sequences_open_loop;
#[path = "active_sequences_shared.rs"]
mod active_sequences_shared;

use active_sequences_open_loop::{
    ActiveSequencesResult, ActiveSequencesRunConfig, PreparedActiveSequencesCorpus,
    prepare_active_sequences_corpus, prepare_active_sequences_trial,
    run_active_sequences_benchmark, run_active_sequences_trial,
};
use active_sequences_shared::generate_sequence_events;
use clap::Parser;
use dynamo_bench::kv_router_common::args::CommonArgs;
use dynamo_bench::kv_router_common::issuer::{parse_cpu_list, pin_current_thread_to_cpus};
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

    /// Optional CPU for the absolute-deadline issuer thread.
    #[clap(long)]
    issuer_cpu: Option<usize>,

    /// Optional CPU list/ranges for the Tokio lane runtime, for example 0-7.
    #[clap(long)]
    backend_cpus: Option<String>,

    /// Attach an expected prefill duration of ISL / rate to every add, exercising the
    /// modeled (AIC-style) prefill-load path. Omit for the unmodeled default path.
    #[clap(long)]
    modeled_prefill_tokens_per_sec: Option<u64>,

    /// Replay the prepared corpus this many times in one process, each against a fresh
    /// tracker. Later trials skip trace generation, which keeps profiles free of setup work.
    #[clap(long, default_value = "1")]
    trials: usize,

    /// JSON output path for one benchmark result.
    #[clap(long, default_value = "active_sequences_result.json")]
    result_json_output: String,
}

fn validate_args(args: &Args, backend_cpus: Option<&[usize]>) -> anyhow::Result<()> {
    if args.common.test {
        anyhow::bail!(
            "active_sequences_bench no longer supports --test; run `cargo test --package dynamo-bench --test active_sequences_trace` instead"
        );
    }
    if args.trials == 0 {
        anyhow::bail!("--trials must be at least 1");
    }
    if args.trials > 1 && args.common.sweep {
        anyhow::bail!("--trials does not support --sweep");
    }
    if args.modeled_prefill_tokens_per_sec == Some(0) {
        anyhow::bail!("--modeled-prefill-tokens-per-sec must be positive");
    }
    if args.operation_lanes == 0 {
        anyhow::bail!("--operation-lanes must be at least 1");
    }
    if args.operation_lanes > u16::MAX as usize {
        anyhow::bail!("--operation-lanes exceeds the u16 lane-ID space");
    }
    if args.issuer_cpu.is_some() && backend_cpus.is_none() {
        anyhow::bail!("--backend-cpus is required when --issuer-cpu is set");
    }
    if let (Some(issuer), Some(cpus)) = (args.issuer_cpu, backend_cpus)
        && cpus.contains(&issuer)
    {
        anyhow::bail!("--issuer-cpu must be disjoint from --backend-cpus");
    }
    Ok(())
}

fn run_config(args: &Args, backend_cpus: &[usize]) -> ActiveSequencesRunConfig {
    ActiveSequencesRunConfig {
        operation_lanes: args.operation_lanes,
        spin_us: args.issuer_spin_us,
        issue_lag_diagnostic_threshold_us: args.issue_lag_diagnostic_threshold_us,
        issuer_cpu: args.issuer_cpu,
        backend_cpus: backend_cpus.to_vec(),
        modeled_prefill_tokens_per_sec: args.modeled_prefill_tokens_per_sec,
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
    let traces = args.common.load_mooncake_trace(path)?;
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
        "Tracker service: {:.0} ns/op; backend CPU: {} ns/op",
        result.tracker_service_ns_per_op,
        result
            .backend_cpu_ns_per_op
            .map_or_else(|| "n/a".to_string(), |value| format!("{value:.0}")),
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
    backend_cpus: &[usize],
    benchmark_duration_ms: u64,
    output_duration: Option<u64>,
) -> anyhow::Result<()> {
    let Some(corpus) = prepare_benchmark(args, benchmark_duration_ms).await? else {
        return Ok(());
    };
    let path = result_path(&args.result_json_output, output_duration);
    // Earlier trials prepare from the shared corpus; the last consumes it so its
    // storage is released before that timed run.
    for trial in 0..args.trials - 1 {
        let prepared = prepare_active_sequences_trial(
            &corpus,
            args.operation_lanes,
            args.modeled_prefill_tokens_per_sec,
        )?;
        let result = run_active_sequences_trial(prepared, run_config(args, backend_cpus)).await?;
        report_trial(&result, &trial_path(args, &path, trial))?;
    }
    let result = run_active_sequences_benchmark(corpus, run_config(args, backend_cpus)).await?;
    report_trial(&result, &trial_path(args, &path, args.trials - 1))
}

fn trial_path(args: &Args, path: &str, trial: usize) -> String {
    if args.trials == 1 {
        return path.to_string();
    }
    format!("{}_trial{trial}.json", path.trim_end_matches(".json"))
}

fn report_trial(result: &ActiveSequencesResult, path: &str) -> anyhow::Result<()> {
    print_result(result);
    write_result(path, result)?;
    if !result.kept_up {
        eprintln!(
            "WARNING: Active Sequences replay did not keep up; inspect issue, queue, and drain metrics"
        );
    }
    Ok(())
}

async fn async_main(args: Args, backend_cpus: Vec<usize>) -> anyhow::Result<()> {
    if args.common.sweep {
        let durations = compute_sweep_durations(
            args.common.sweep_min_ms,
            args.common.sweep_max_ms,
            args.common.sweep_steps,
        )?;
        for duration_ms in durations.into_iter().rev() {
            println!("\n=== Active Sequences sweep: benchmark_duration_ms={duration_ms} ===");
            run_cell(&args, &backend_cpus, duration_ms, Some(duration_ms)).await?;
        }
        return Ok(());
    }

    run_cell(
        &args,
        &backend_cpus,
        args.common.benchmark_duration_ms,
        None,
    )
    .await
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let backend_cpus = args
        .backend_cpus
        .as_deref()
        .map(parse_cpu_list)
        .transpose()?;
    validate_args(&args, backend_cpus.as_deref())?;
    init_sequence_logging(args.common.sequence_logs);
    let mut runtime = tokio::runtime::Builder::new_multi_thread();
    runtime.enable_all();
    if let Some(cpus) = backend_cpus.as_deref() {
        // Runtime workers inherit this mask; size the pool to it to avoid oversubscription.
        pin_current_thread_to_cpus(cpus)?;
        runtime.worker_threads(cpus.len());
    }
    runtime
        .build()?
        .block_on(async_main(args, backend_cpus.unwrap_or_default()))
}
