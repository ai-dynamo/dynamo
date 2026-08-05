// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust-native offline replay benchmark entrypoint.
//!
//! Useful for profiling replay itself without the Python CLI wrapper. This
//! bench intentionally uses the mocker's internal polynomial perf model so the
//! measurements stay focused on replay and router overhead.
//!
//! Run with: cargo bench --package dynamo-bench --bench offline_replay_bench -- --help

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use dynamo_mocker::common::protocols::{
    EngineType, KvTransferTimingMode, MockEngineArgs, SglangArgs, WorkerType,
};
use dynamo_mocker::loadgen::Trace;
use dynamo_mocker::replay::{
    CanonicalDeterminismMetadata, CanonicalEngineConfig, CanonicalExecutionMetadata,
    CanonicalReplayCoverage, CanonicalReplayMetadata, CanonicalReplayRecord,
    CanonicalSemanticFeatures, CanonicalSlaMetadata, CanonicalWorkloadMetadata,
    OfflineDisaggReplayConfig, OfflineRuntimeEvidence, ReplayArgsMode, ReplayCaptureOptions,
    ReplayDeterminism, ReplayRouterMode, SlaThresholds, TraceSimulationReport,
    canonical_router_metadata, canonical_topology,
    simulate_loaded_trace_disagg_with_router_mode_and_options,
    simulate_loaded_trace_with_router_mode_and_options, with_replay_determinism,
    with_runtime_evidence,
};
use serde_json::Value;

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
    /// Requires building with the `replay-bench` Cargo feature.
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

fn canonical_report(
    report: &TraceSimulationReport,
    args: &Args,
    engine_args: &MockEngineArgs,
    workload_digest: &str,
    evidence: OfflineRuntimeEvidence,
) -> Result<CanonicalReplayRecord> {
    let engine_config = match args.serving_mode {
        ServingModeArg::Aggregated => CanonicalEngineConfig::aggregated(engine_args)?,
        ServingModeArg::Disagg => {
            let mut prefill_args = engine_args.clone();
            prefill_args.worker_type = WorkerType::Prefill;
            let mut decode_args = engine_args.clone();
            decode_args.worker_type = WorkerType::Decode;
            CanonicalEngineConfig::disaggregated(&prefill_args, &decode_args)?
        }
    };
    let topology = match args.serving_mode {
        ServingModeArg::Aggregated => ReplayArgsMode::Aggregated,
        ServingModeArg::Disagg => ReplayArgsMode::Disagg,
    };
    let router_mode = ReplayRouterMode::from(args.router_mode);
    let metadata = CanonicalReplayMetadata {
        replay_bench: true,
        byte_identity_scope: "same_target_toolchain_semantic_features".to_string(),
        workload: CanonicalWorkloadMetadata::Trace {
            format: "mooncake".to_string(),
            block_size: Some(args.trace_block_size),
            digest: workload_digest.to_string(),
        },
        execution: CanonicalExecutionMetadata {
            topology: canonical_topology(topology),
            num_workers: args.num_workers,
            num_prefill_workers: args.num_prefill_workers,
            num_decode_workers: args.num_decode_workers,
            replay_concurrency: None,
            arrival_speedup_ratio: args.arrival_speedup_ratio,
            max_sim_time_ms: None,
            aic_prefill_load_estimator: None,
            aic_performance_model_implementation: None,
            aic_prefill_load_estimator_implementation: None,
        },
        engine_config,
        router: canonical_router_metadata(router_mode, None)?,
        sla: CanonicalSlaMetadata {
            ttft_ms: None,
            itl_ms: None,
            e2e_ms: None,
        },
        determinism: CanonicalDeterminismMetadata::canonical_v1(),
        semantic_features: CanonicalSemanticFeatures {
            canonical_replay: true,
            mocker_kvbm_offload: cfg!(feature = "mocker-kvbm-offload"),
            aic_forward_pass: false,
        },
    };
    let coverage = CanonicalReplayCoverage {
        capture_per_request: true,
        capture_planner_details: false,
        capture_canonical_evidence: true,
        per_request_records: report.per_request.len(),
        pressure: evidence.pressure,
        kv_ingest: evidence.kv_ingest,
    };
    CanonicalReplayRecord::build(report, &metadata, &coverage, Value::Null)
}

fn main() -> Result<()> {
    if is_bench_harness_invocation() {
        eprintln!("offline_replay_bench: skipping no-arg harness invocation");
        return Ok(());
    }

    let args = Args::parse();
    anyhow::ensure!(
        args.canonical_reports_jsonl.is_none() || cfg!(feature = "replay-bench"),
        "--canonical-reports-jsonl requires building with --features replay-bench"
    );
    let engine_args = build_engine_args(&args)?;
    let canonical_workload = if args.canonical_reports_jsonl.is_some() {
        let trace_bytes = std::fs::read(&args.trace_file)
            .with_context(|| format!("failed to read trace input at {:?}", args.trace_file))?;
        let mut workload_hasher = blake3::Hasher::new();
        workload_hasher.update(b"dynamo.offline-replay.trace.v1");
        workload_hasher.update(&(trace_bytes.len() as u64).to_be_bytes());
        workload_hasher.update(&trace_bytes);
        Some((trace_bytes, workload_hasher.finalize().to_hex().to_string()))
    } else {
        None
    };
    let trace = Trace::from_mooncake(&args.trace_file, args.trace_block_size)?;
    if let Some((trace_bytes, _)) = canonical_workload.as_ref() {
        ensure!(
            std::fs::read(&args.trace_file)? == *trace_bytes,
            "trace input changed while it was being loaded"
        );
    }
    anyhow::ensure!(args.iterations > 0, "iterations must be greater than 0");
    let mut timing_writer = args
        .timings_jsonl
        .as_ref()
        .map(|path| {
            File::create(path)
                .map(BufWriter::new)
                .with_context(|| format!("failed to create timing output at {path:?}"))
        })
        .transpose()?;
    let mut canonical_writer = args
        .canonical_reports_jsonl
        .as_ref()
        .map(|path| {
            File::create(path)
                .map(BufWriter::new)
                .with_context(|| format!("failed to create canonical report output at {path:?}"))
        })
        .transpose()?;
    let record_per_request = canonical_writer.is_some();
    let mut first_canonical_line: Option<Vec<u8>> = None;
    let mut last_report = None;
    for iteration in 0..args.iterations {
        let determinism = if canonical_writer.is_some() {
            ReplayDeterminism::CanonicalV1
        } else {
            ReplayDeterminism::Random
        };
        let capture_options = ReplayCaptureOptions {
            capture_per_request: record_per_request,
            capture_planner_details: false,
            capture_canonical_evidence: canonical_writer.is_some(),
            determinism,
        };
        let (report, runtime_evidence) = with_runtime_evidence(capture_options, || {
            with_replay_determinism(determinism, || -> Result<_> {
                match args.serving_mode {
                    ServingModeArg::Aggregated => {
                        simulate_loaded_trace_with_router_mode_and_options(
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
                        )
                    }
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
                        )
                    }
                }
            })
        });
        let report = report?;
        if let Some(writer) = timing_writer.as_mut() {
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
            let line = canonical_report(
                &report,
                &args,
                &engine_args,
                &canonical_workload
                    .as_ref()
                    .expect("canonical writer requires canonical workload identity")
                    .1,
                runtime_evidence,
            )?
            .into_json_line()
            .context("failed to encode canonical replay report")?;
            if let Some(first) = first_canonical_line.as_ref() {
                ensure!(
                    line == *first,
                    "canonical replay output changed between iterations 0 and {iteration}"
                );
            } else {
                first_canonical_line = Some(line.clone());
            }
            writer.write_all(&line)?;
        }
        last_report = Some(report);
    }
    if let Some((trace_bytes, _)) = canonical_workload.as_ref() {
        ensure!(
            std::fs::read(&args.trace_file)? == *trace_bytes,
            "trace input changed during replay"
        );
    }
    if let Some(writer) = timing_writer.as_mut() {
        writer
            .flush()
            .context("failed to flush timings JSONL output")?;
    }
    if let Some(writer) = canonical_writer.as_mut() {
        writer
            .flush()
            .context("failed to flush canonical report JSONL output")?;
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
