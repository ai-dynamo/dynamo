// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust-native offline replay benchmark entrypoint.
//!
//! Useful for profiling replay itself without the Python CLI wrapper. This
//! bench intentionally uses the mocker's internal polynomial perf model so the
//! measurements stay focused on replay and router overhead.
//!
//! Run with: cargo bench --package dynamo-bench --bench offline_replay_bench -- --help

use std::collections::BTreeMap;
use std::fs::File;
use std::path::PathBuf;

#[cfg(feature = "mocker-kvbm-offload")]
use anyhow::ensure;
use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use dynamo_mocker::common::protocols::{
    EngineType, KvTransferTimingMode, MockEngineArgs, SglangArgs, WorkerType,
};
use dynamo_mocker::loadgen::Trace;
use dynamo_mocker::replay::{
    OfflineDisaggReplayConfig, ReplayRouterMode, SlaThresholds, TraceSimulationReport,
    simulate_loaded_trace_disagg_with_router_mode_and_options,
    simulate_loaded_trace_with_router_mode_and_options,
};
use serde_json::{Map, Value, json};

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum RouterModeArg {
    RoundRobin,
    KvRouter,
}

impl RouterModeArg {
    fn as_str(self) -> &'static str {
        match self {
            Self::RoundRobin => "round-robin",
            Self::KvRouter => "kv-router",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum ServingModeArg {
    Aggregated,
    Disagg,
}

impl ServingModeArg {
    fn as_str(self) -> &'static str {
        match self {
            Self::Aggregated => "aggregated",
            Self::Disagg => "disagg",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum EngineTypeArg {
    Vllm,
    Sglang,
    Trtllm,
}

impl EngineTypeArg {
    fn as_str(self) -> &'static str {
        match self {
            Self::Vllm => "vllm",
            Self::Sglang => "sglang",
            Self::Trtllm => "trtllm",
        }
    }

    fn native_router_event_visibility(self) -> &'static str {
        match self {
            Self::Vllm | Self::Trtllm => "pass-start",
            Self::Sglang => "pass-end",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum KvTransferTimingModeArg {
    FullPrompt,
    DestinationMissing,
}

impl From<EngineTypeArg> for EngineType {
    fn from(value: EngineTypeArg) -> Self {
        match value {
            EngineTypeArg::Vllm => EngineType::Vllm,
            EngineTypeArg::Sglang => EngineType::Sglang,
            EngineTypeArg::Trtllm => EngineType::Trtllm,
        }
    }
}

impl From<RouterModeArg> for ReplayRouterMode {
    fn from(value: RouterModeArg) -> Self {
        match value {
            RouterModeArg::RoundRobin => ReplayRouterMode::RoundRobin,
            RouterModeArg::KvRouter => ReplayRouterMode::KvRouter,
        }
    }
}

impl From<KvTransferTimingModeArg> for KvTransferTimingMode {
    fn from(value: KvTransferTimingModeArg) -> Self {
        match value {
            KvTransferTimingModeArg::FullPrompt => KvTransferTimingMode::FullPrompt,
            KvTransferTimingModeArg::DestinationMissing => KvTransferTimingMode::DestinationMissing,
        }
    }
}

fn is_bench_harness_invocation() -> bool {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    args.is_empty() || args.iter().all(|arg| arg == "--bench")
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

    /// Serving topology to simulate
    #[arg(long, value_enum, default_value_t = ServingModeArg::Aggregated)]
    serving_mode: ServingModeArg,

    /// Number of prefill workers in disaggregated mode
    #[arg(long, default_value_t = 1)]
    num_prefill_workers: usize,

    /// Number of decode workers in disaggregated mode
    #[arg(long, default_value_t = 1)]
    num_decode_workers: usize,

    /// Mock engine scheduling mode
    #[arg(long, value_enum, default_value_t = EngineTypeArg::Vllm)]
    engine_type: EngineTypeArg,

    /// Router mode for multi-worker replay
    #[arg(long, value_enum, default_value_t = RouterModeArg::KvRouter)]
    router_mode: RouterModeArg,

    /// Compress trace arrival timestamps by this factor
    #[arg(long, default_value_t = 4.0)]
    arrival_speedup_ratio: f64,

    /// Trace hash block size used to expand hash_ids into tokens
    #[arg(long, default_value_t = 512)]
    trace_block_size: usize,

    /// Engine/router block size used for replay hashing and mock execution
    #[arg(long, default_value_t = 64)]
    block_size: usize,

    /// Override GPU KV-cache block capacity per worker
    #[arg(long)]
    num_gpu_blocks: Option<usize>,

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

    /// KV-cache bytes per token for disaggregated transfer and offload timing
    #[arg(long)]
    kv_bytes_per_token: Option<usize>,

    /// Disaggregated KV-transfer bandwidth in GB/s
    #[arg(long)]
    kv_transfer_bandwidth: Option<f64>,

    /// Disaggregated transfer timing model
    #[arg(long, value_enum, default_value_t = KvTransferTimingModeArg::FullPrompt)]
    kv_transfer_timing_mode: KvTransferTimingModeArg,

    /// KVBM G2 host-memory block capacity
    #[cfg(feature = "mocker-kvbm-offload")]
    #[arg(long)]
    num_g2_blocks: Option<usize>,

    /// KVBM G3 shared lower-tier block capacity
    #[cfg(feature = "mocker-kvbm-offload")]
    #[arg(long)]
    num_g3_blocks: Option<usize>,

    /// Enable KVBM mock G4 object storage
    #[cfg(feature = "mocker-kvbm-offload")]
    #[arg(long)]
    enable_g4_storage: bool,

    /// KVBM G1-to-G2 offload batch size
    #[cfg(feature = "mocker-kvbm-offload")]
    #[arg(long)]
    offload_batch_size: Option<usize>,

    #[cfg(feature = "mocker-kvbm-offload")]
    #[arg(long)]
    bandwidth_g1_to_g2_gbps: Option<f64>,

    #[cfg(feature = "mocker-kvbm-offload")]
    #[arg(long)]
    bandwidth_g2_to_g1_gbps: Option<f64>,

    #[cfg(feature = "mocker-kvbm-offload")]
    #[arg(long)]
    bandwidth_g2_to_g3_gbps: Option<f64>,

    #[cfg(feature = "mocker-kvbm-offload")]
    #[arg(long)]
    bandwidth_g3_to_g2_gbps: Option<f64>,

    #[cfg(feature = "mocker-kvbm-offload")]
    #[arg(long)]
    bandwidth_g2_to_g4_gbps: Option<f64>,

    #[cfg(feature = "mocker-kvbm-offload")]
    #[arg(long)]
    bandwidth_g4_to_g2_gbps: Option<f64>,

    /// Optional path to write the full replay report as pretty JSON
    #[arg(long)]
    report_json: Option<PathBuf>,

    /// Number of times to rerun the same replay in-process
    #[arg(long, default_value_t = 1)]
    iterations: usize,

    /// Emit one JSON object per measured replay iteration to this path.
    #[arg(long)]
    timings_jsonl: Option<PathBuf>,

    /// Emit one canonical full replay report per iteration for parity checks.
    #[arg(long)]
    canonical_reports_jsonl: Option<PathBuf>,

    /// Ignored -- passed by cargo bench
    #[arg(long, hide = true)]
    bench: bool,
}

fn build_engine_args(args: &Args) -> Result<MockEngineArgs> {
    let mut builder = MockEngineArgs::builder()
        .engine_type(args.engine_type.into())
        .block_size(args.block_size)
        .kv_bytes_per_token(args.kv_bytes_per_token)
        .kv_transfer_bandwidth(args.kv_transfer_bandwidth)
        .kv_transfer_timing_mode(args.kv_transfer_timing_mode.into());
    if args.engine_type == EngineTypeArg::Sglang {
        builder = builder.sglang(Some(SglangArgs {
            page_size: Some(args.block_size),
            ..Default::default()
        }));
    }
    if let Some(max_num_seqs) = args.max_num_seqs {
        builder = builder.max_num_seqs(Some(max_num_seqs));
    }
    if let Some(num_gpu_blocks) = args.num_gpu_blocks {
        builder = builder.num_gpu_blocks(num_gpu_blocks);
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
    #[cfg(feature = "mocker-kvbm-offload")]
    {
        if args.num_g2_blocks.is_some() {
            ensure!(
                args.engine_type == EngineTypeArg::Vllm,
                "KVBM offload requires --engine-type vllm"
            );
            ensure!(
                args.kv_bytes_per_token.is_some(),
                "KVBM offload requires --kv-bytes-per-token"
            );
        }
        builder = builder
            .num_g2_blocks(args.num_g2_blocks)
            .num_g3_blocks(args.num_g3_blocks)
            .enable_g4_storage(args.enable_g4_storage)
            .offload_batch_size(args.offload_batch_size)
            .bandwidth_g1_to_g2_gbps(args.bandwidth_g1_to_g2_gbps)
            .bandwidth_g2_to_g1_gbps(args.bandwidth_g2_to_g1_gbps)
            .bandwidth_g2_to_g3_gbps(args.bandwidth_g2_to_g3_gbps)
            .bandwidth_g3_to_g2_gbps(args.bandwidth_g3_to_g2_gbps)
            .bandwidth_g2_to_g4_gbps(args.bandwidth_g2_to_g4_gbps)
            .bandwidth_g4_to_g2_gbps(args.bandwidth_g4_to_g2_gbps);
    }
    builder
        .build()
        .context("failed to build replay engine args")?
        .normalized()
}

fn canonicalize_json(value: Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.into_iter().map(canonicalize_json).collect()),
        Value::Object(values) => Value::Object(
            values
                .into_iter()
                .map(|(key, value)| (key, canonicalize_json(value)))
                .collect::<BTreeMap<_, _>>()
                .into_iter()
                .collect::<Map<_, _>>(),
        ),
        scalar => scalar,
    }
}

fn canonical_report(report: &TraceSimulationReport, engine_type: EngineTypeArg) -> Result<Value> {
    let mut per_request = serde_json::to_value(&report.per_request)?;
    if let Value::Array(records) = &mut per_request {
        records.sort_unstable_by(|left, right| {
            let left_uuid = left.get("uuid").and_then(Value::as_str).unwrap_or_default();
            let right_uuid = right
                .get("uuid")
                .and_then(Value::as_str)
                .unwrap_or_default();
            left_uuid.cmp(right_uuid)
        });
    }

    let mut summary = serde_json::to_value(report)?;
    let summary = summary
        .as_object_mut()
        .context("serialized replay report must be an object")?;
    summary.remove("wall_time_ms");
    summary.remove("processed_tokens_per_s");
    summary.remove("processed_output_tokens_per_s");
    Ok(canonicalize_json(json!({
        "engine_type": engine_type.as_str(),
        "native_router_event_visibility": engine_type.native_router_event_visibility(),
        "replay_bench": cfg!(feature = "replay-bench"),
        "summary": summary,
        "per_request": per_request,
    })))
}

fn main() -> Result<()> {
    if is_bench_harness_invocation() {
        eprintln!("offline_replay_bench: skipping no-arg harness invocation");
        return Ok(());
    }

    let args = Args::parse();
    let engine_args = build_engine_args(&args)?;
    let trace = Trace::from_mooncake(&args.trace_file, args.trace_block_size)?;
    anyhow::ensure!(args.iterations > 0, "iterations must be greater than 0");
    let mut timing_writer = args
        .timings_jsonl
        .as_ref()
        .map(|path| {
            File::create(path)
                .map(std::io::BufWriter::new)
                .with_context(|| format!("failed to create timing output at {path:?}"))
        })
        .transpose()?;
    let mut canonical_writer = args
        .canonical_reports_jsonl
        .as_ref()
        .map(|path| {
            File::create(path)
                .map(std::io::BufWriter::new)
                .with_context(|| format!("failed to create canonical report output at {path:?}"))
        })
        .transpose()?;
    let record_per_request = canonical_writer.is_some();
    let mut last_report = None;
    for iteration in 0..args.iterations {
        let report = match args.serving_mode {
            ServingModeArg::Aggregated => simulate_loaded_trace_with_router_mode_and_options(
                engine_args.clone(),
                None,
                None,
                trace.clone(),
                args.num_workers,
                args.arrival_speedup_ratio,
                args.router_mode.into(),
                record_per_request,
                None,
                SlaThresholds::default(),
            )?,
            ServingModeArg::Disagg => {
                let mut prefill_args = engine_args.clone();
                prefill_args.worker_type = WorkerType::Prefill;
                let mut decode_args = engine_args.clone();
                decode_args.worker_type = WorkerType::Decode;
                simulate_loaded_trace_disagg_with_router_mode_and_options(
                    OfflineDisaggReplayConfig {
                        prefill_args,
                        decode_args,
                        num_prefill_workers: args.num_prefill_workers,
                        num_decode_workers: args.num_decode_workers,
                    },
                    None,
                    None,
                    trace.clone(),
                    args.arrival_speedup_ratio,
                    args.router_mode.into(),
                    record_per_request,
                    None,
                    SlaThresholds::default(),
                )?
            }
        };
        if let Some(writer) = timing_writer.as_mut() {
            use std::io::Write;

            serde_json::to_writer(
                &mut *writer,
                &serde_json::json!({
                    "iteration": iteration,
                    "wall_time_ms": report.throughput.wall_time_ms,
                    "serving_mode": args.serving_mode.as_str(),
                    "router_mode": args.router_mode.as_str(),
                    "engine_type": args.engine_type.as_str(),
                    "native_router_event_visibility": args.engine_type.native_router_event_visibility(),
                    "replay_bench": cfg!(feature = "replay-bench"),
                }),
            )?;
            writer.write_all(b"\n")?;
        }
        if let Some(writer) = canonical_writer.as_mut() {
            use std::io::Write;

            serde_json::to_writer(&mut *writer, &canonical_report(&report, args.engine_type)?)?;
            writer.write_all(b"\n")?;
        }
        last_report = Some(report);
    }
    let report = last_report.expect("iterations must be at least 1");

    if let Some(report_path) = args.report_json.as_ref() {
        let file = File::create(report_path)
            .with_context(|| format!("failed to create report file at {:?}", report_path))?;
        serde_json::to_writer_pretty(file, &report)
            .with_context(|| format!("failed to write report JSON to {:?}", report_path))?;
        println!("Saved report to {}", report_path.display());
    }

    println!("Offline replay report");
    println!("{report}");

    Ok(())
}
