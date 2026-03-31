// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust-native offline replay benchmark entrypoint.
//!
//! Useful for profiling replay itself without the Python CLI wrapper. This keeps
//! the default mocker perf model unless CLI overrides are provided.

use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use dynamo_mocker::common::protocols::MockEngineArgs;
use dynamo_mocker::replay::{ReplayRouterMode, simulate_trace_file_with_router_mode};

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum RouterModeArg {
    RoundRobin,
    KvRouter,
}

impl From<RouterModeArg> for ReplayRouterMode {
    fn from(value: RouterModeArg) -> Self {
        match value {
            RouterModeArg::RoundRobin => ReplayRouterMode::RoundRobin,
            RouterModeArg::KvRouter => ReplayRouterMode::KvRouter,
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "offline_replay_bench")]
#[command(about = "Run offline replay directly in Rust for benchmarking and profiling")]
struct Args {
    /// Mooncake trace JSONL file
    trace_file: PathBuf,

    /// Number of aggregated workers
    #[arg(long, default_value_t = 4)]
    num_workers: usize,

    /// Router mode for multi-worker replay
    #[arg(long, value_enum, default_value_t = RouterModeArg::KvRouter)]
    router_mode: RouterModeArg,

    /// Compress trace arrival timestamps by this factor
    #[arg(long, default_value_t = 4.0)]
    arrival_speedup_ratio: f64,

    /// Mocker block size; defaults to 512 for Mooncake traces
    #[arg(long, default_value_t = 512)]
    block_size: usize,

    /// Override max running requests per worker
    #[arg(long)]
    max_num_seqs: Option<usize>,

    /// Override batched token budget per worker pass
    #[arg(long)]
    max_num_batched_tokens: Option<usize>,

    /// Global speedup multiplier for the default perf model
    #[arg(long)]
    speedup_ratio: Option<f64>,

    /// Additional decode-only speedup multiplier
    #[arg(long)]
    decode_speedup_ratio: Option<f64>,

    /// Explicit planner profile NPZ to use for perf-model timing
    #[arg(long)]
    planner_profile_data: Option<PathBuf>,

    /// Optional path to write the full replay report as pretty JSON
    #[arg(long)]
    report_json: Option<PathBuf>,

    /// Number of times to rerun the same replay in-process
    #[arg(long, default_value_t = 1)]
    iterations: usize,
}

fn build_engine_args(args: &Args) -> Result<MockEngineArgs> {
    let mut builder = MockEngineArgs::builder();
    builder = builder.block_size(args.block_size);
    if let Some(max_num_seqs) = args.max_num_seqs {
        builder = builder.max_num_seqs(Some(max_num_seqs));
    }
    if let Some(max_num_batched_tokens) = args.max_num_batched_tokens {
        builder = builder.max_num_batched_tokens(Some(max_num_batched_tokens));
    }
    if let Some(speedup_ratio) = args.speedup_ratio {
        builder = builder.speedup_ratio(speedup_ratio);
    }
    if let Some(decode_speedup_ratio) = args.decode_speedup_ratio {
        builder = builder.decode_speedup_ratio(decode_speedup_ratio);
    }
    if let Some(planner_profile_data) = args.planner_profile_data.as_ref() {
        builder = builder.planner_profile_data(Some(planner_profile_data.clone()));
    }
    builder
        .build()
        .context("failed to build replay engine args")?
        .normalized()
}

fn main() -> Result<()> {
    let args = Args::parse();
    let engine_args = build_engine_args(&args)?;
    let started_at = Instant::now();
    let mut last_report = None;
    for _ in 0..args.iterations {
        last_report = Some(simulate_trace_file_with_router_mode(
            engine_args.clone(),
            None,
            &args.trace_file,
            args.num_workers,
            args.arrival_speedup_ratio,
            args.router_mode.into(),
        )?);
    }
    let report = last_report.expect("iterations must be at least 1");
    let process_wall_time_ms = started_at.elapsed().as_secs_f64() * 1000.0;

    if let Some(report_path) = args.report_json.as_ref() {
        let file = File::create(report_path)
            .with_context(|| format!("failed to create report file at {:?}", report_path))?;
        serde_json::to_writer_pretty(file, &report)
            .with_context(|| format!("failed to write report JSON to {:?}", report_path))?;
        println!("Saved report to {}", report_path.display());
    }

    println!("Offline replay report");
    println!(
        "  completed_requests: {}",
        report.request_counts.completed_requests
    );
    println!(
        "  request_throughput_rps: {:.6}",
        report.throughput.request_throughput_rps
    );
    println!(
        "  output_throughput_tok_s: {:.6}",
        report.throughput.output_throughput_tok_s
    );
    println!("  mean_ttft_ms: {:.6}", report.latency.ttft.mean_ms);
    println!("  mean_e2e_latency_ms: {:.6}", report.latency.e2e.mean_ms);
    println!(
        "  prefix_cache_reused_ratio: {:.6}",
        report.prefix_cache_reused_ratio
    );
    println!("  wall_time_ms: {:.6}", report.throughput.wall_time_ms);
    println!("  process_wall_time_ms: {:.6}", process_wall_time_ms);

    Ok(())
}
