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
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_mocker::common::perf_model::PerfModel;
use dynamo_mocker::common::protocols::{
    EngineType, KvTransferTimingMode, MockerConfig, WorkerType,
};
use dynamo_mocker::loadgen::Trace;
use dynamo_mocker::replay::{
    CanonicalReplayCoverage, CanonicalReplayRecord, OfflineDisaggReplayConfig,
    ReplayCaptureOptions, ReplayDeterminism, ReplayRouterMode, SlaThresholds,
    TraceSimulationReport, simulate_loaded_trace_disagg_with_router_mode_and_capture_options,
    simulate_loaded_trace_with_router_mode_and_capture_options,
};
use serde_json::{Value, json};

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

    fn canonical_name(self) -> &'static str {
        match self {
            Self::RoundRobin => "round_robin",
            Self::KvRouter => "kv_router",
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

    fn canonical_name(self) -> &'static str {
        match self {
            Self::Aggregated => "aggregated",
            Self::Disagg => "disaggregated",
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
            Self::Vllm | Self::Sglang | Self::Trtllm => "pass-end",
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

    /// Queue KV-router requests while every worker's active prefill tokens
    /// exceed this fraction of its batched-token budget. Unset admits every
    /// request immediately.
    #[arg(long)]
    router_queue_threshold: Option<f64>,

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

    /// Optional path to write the full replay report as pretty JSON
    #[arg(long)]
    report_json: Option<PathBuf>,

    /// Number of times to rerun the same replay in-process
    #[arg(long, default_value_t = 1)]
    iterations: usize,

    /// Emit one JSON object per measured replay iteration to this path.
    ///
    /// `replay_call_ms` and `replay_cpu_ms` cover only the replay call, after
    /// trace loading and cloning. `wall_time_ms` is the replayer's own timer,
    /// which also covers engine setup and report finalization. Replay runs on
    /// the calling thread, so `replay_cpu_ms` excludes time the thread spent
    /// preempted on a shared host.
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

fn build_engine_args(args: &Args) -> Result<MockerConfig> {
    let mut rank = serde_json::json!({
        "backend":EngineType::from(args.engine_type),"block_size":args.block_size,
        "kv_transfer_bytes_per_token":args.kv_bytes_per_token,
        "kv_transfer_bandwidth":args.kv_transfer_bandwidth,
        "kv_transfer_timing_mode":KvTransferTimingMode::from(args.kv_transfer_timing_mode)
    });
    for (key, value) in [
        ("max_num_seqs", args.max_num_seqs),
        ("num_gpu_blocks", args.num_gpu_blocks),
        ("max_num_batched_tokens", args.max_num_batched_tokens),
    ] {
        if let Some(value) = value {
            rank[key] = serde_json::json!(value);
        }
    }
    for (key, value) in [
        ("speedup_ratio", args.speedup_ratio),
        ("decode_speedup_ratio", args.decode_speedup_ratio),
    ] {
        if let Some(value) = value {
            rank[key] = serde_json::json!(value);
        }
    }
    MockerConfig::from_value(serde_json::json!({
        "engine": rank
    }))
    .context("invalid replay engine config")
}

fn router_config(args: &Args) -> Result<Option<KvRouterConfig>> {
    let Some(threshold) = args.router_queue_threshold else {
        return Ok(None);
    };
    ensure!(
        args.router_mode == RouterModeArg::KvRouter,
        "--router-queue-threshold requires --router-mode kv-router"
    );
    let config = KvRouterConfig {
        router_queue_threshold: Some(threshold),
        ..KvRouterConfig::default()
    };
    config
        .validate()
        .map_err(|error| anyhow::anyhow!("invalid --router-queue-threshold: {error}"))?;
    Ok(Some(config))
}

fn canonical_capture_options(enabled: bool) -> ReplayCaptureOptions {
    ReplayCaptureOptions {
        capture_per_request: enabled,
        capture_lifecycle_evidence: false,
        capture_canonical_evidence: enabled,
        determinism: if enabled {
            ReplayDeterminism::CanonicalV1
        } else {
            ReplayDeterminism::Random
        },
    }
}

fn canonical_engine_pool_metadata(args: &MockerConfig) -> Result<Value> {
    ensure!(
        !matches!(args.perf_model.as_ref(), PerfModel::Interpolated { .. }),
        "canonical replay does not support dynamo_profile timing"
    );
    ensure!(
        args.runtime.response_replay_trace_path.is_none(),
        "canonical replay does not support response_replay_trace_path"
    );
    let ais_config = args.ais_perf_config();
    ensure!(
        ais_config.is_none_or(|config| config["backend_version"].as_str().is_some()),
        "canonical AIS replay requires a resolved backend version"
    );
    let mut metadata = serde_json::to_value(args)?;
    let metadata = metadata
        .as_object_mut()
        .context("serialized replay engine configuration must be an object")?;
    metadata.insert(
        "performance_model".to_string(),
        json!({
            "kind": if ais_config.is_some() {
                "ais_callback"
            } else {
                "builtin_polynomial"
            },
            "ais": ais_config,
        }),
    );
    Ok(Value::Object(metadata.clone()))
}

fn canonical_engine_config(args: &Args, engine_args: &MockerConfig) -> Result<Value> {
    match args.serving_mode {
        ServingModeArg::Aggregated => Ok(json!({
            "aggregated": canonical_engine_pool_metadata(engine_args)?,
        })),
        ServingModeArg::Disagg => {
            let mut prefill_args = engine_args.clone();
            prefill_args.worker_type = WorkerType::Prefill;
            let mut decode_args = engine_args.clone();
            decode_args.worker_type = WorkerType::Decode;
            Ok(json!({
                "prefill": canonical_engine_pool_metadata(&prefill_args)?,
                "decode": canonical_engine_pool_metadata(&decode_args)?,
            }))
        }
    }
}

fn canonical_metadata(
    args: &Args,
    engine_args: &MockerConfig,
    workload_digest: &str,
) -> Result<Value> {
    let router_config = match args.router_mode {
        RouterModeArg::RoundRobin => Value::Null,
        RouterModeArg::KvRouter => serde_json::to_value(router_config(args)?.unwrap_or_default())?,
    };
    Ok(json!({
        "replay_bench": cfg!(feature = "replay-bench"),
        "byte_identity_scope": "same_target_toolchain_semantic_features",
        "workload": {
            "kind": "trace",
            "format": "mooncake",
            "block_size": args.trace_block_size,
            "digest": workload_digest,
        },
        "execution": {
            "topology": args.serving_mode.canonical_name(),
            "num_workers": args.num_workers,
            "num_prefill_workers": args.num_prefill_workers,
            "num_decode_workers": args.num_decode_workers,
            "replay_concurrency": Value::Null,
            "arrival_speedup_ratio": args.arrival_speedup_ratio,
            "max_sim_time_ms": Value::Null,
            "ais_prefill_load_estimator": Value::Null,
            "ais_performance_model_implementation": Value::Null,
            "ais_prefill_load_estimator_implementation": Value::Null,
        },
        "engine_config": canonical_engine_config(args, engine_args)?,
        "router": {
            "mode": args.router_mode.canonical_name(),
            "config": router_config,
        },
        "sla": {
            "ttft_ms": Value::Null,
            "itl_ms": Value::Null,
            "e2e_ms": Value::Null,
        },
        "determinism": {
            "request_ids": "ordinal_u128_v1",
            "selection": "default_worker_selector_seeded_v1",
            "seed": 0xd1a0_5eed_u64,
            "candidate_order": ["worker_id", "dp_rank"],
        },
        "semantic_features": {
            "canonical_replay": true,
            "mocker_kvbm_offload": false,
            "ais_forward_pass": false,
        },
    }))
}

fn canonical_report(
    report: &TraceSimulationReport,
    args: &Args,
    engine_args: &MockerConfig,
    workload_digest: &str,
    capture_options: ReplayCaptureOptions,
) -> Result<CanonicalReplayRecord> {
    let metadata = canonical_metadata(args, engine_args, workload_digest)?;
    let coverage = CanonicalReplayCoverage::from_report(report, capture_options);
    CanonicalReplayRecord::build(report, metadata, &coverage, Value::Null)
}

#[cfg(target_os = "linux")]
fn thread_cpu_time_ms() -> Option<f64> {
    let mut timestamp = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut timestamp) };
    (rc == 0).then(|| timestamp.tv_sec as f64 * 1_000.0 + timestamp.tv_nsec as f64 / 1_000_000.0)
}

#[cfg(not(target_os = "linux"))]
fn thread_cpu_time_ms() -> Option<f64> {
    None
}

/// Process high-water RSS so far, so later iterations report the maximum over
/// all earlier ones.
#[cfg(target_os = "linux")]
fn peak_rss_kib() -> Option<i64> {
    let mut usage = unsafe { std::mem::zeroed::<libc::rusage>() };
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    (rc == 0).then_some(usage.ru_maxrss)
}

#[cfg(not(target_os = "linux"))]
fn peak_rss_kib() -> Option<i64> {
    None
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
    let router_config = router_config(&args)?;
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
    let trace_load_start = Instant::now();
    let trace = Trace::from_mooncake(&args.trace_file, args.trace_block_size)?;
    let trace_load_ms = trace_load_start.elapsed().as_secs_f64() * 1_000.0;
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
    let capture_options = canonical_capture_options(canonical_writer.is_some());
    let mut first_canonical_line: Option<Vec<u8>> = None;
    let mut last_report = None;
    for iteration in 0..args.iterations {
        let trace = trace.clone();
        let call_start = Instant::now();
        let cpu_start = thread_cpu_time_ms();
        let report = match args.serving_mode {
            ServingModeArg::Aggregated => {
                simulate_loaded_trace_with_router_mode_and_capture_options(
                    engine_args.clone(),
                    router_config.clone(),
                    None,
                    trace,
                    args.num_workers,
                    args.arrival_speedup_ratio,
                    args.router_mode.into(),
                    capture_options,
                    None,
                    SlaThresholds::default(),
                )?
            }
            ServingModeArg::Disagg => {
                let mut prefill_args = engine_args.clone();
                prefill_args.worker_type = WorkerType::Prefill;
                let mut decode_args = engine_args.clone();
                decode_args.worker_type = WorkerType::Decode;
                simulate_loaded_trace_disagg_with_router_mode_and_capture_options(
                    OfflineDisaggReplayConfig {
                        prefill_args,
                        decode_args,
                        num_prefill_workers: args.num_prefill_workers,
                        num_decode_workers: args.num_decode_workers,
                    },
                    router_config.clone(),
                    None,
                    trace,
                    args.arrival_speedup_ratio,
                    args.router_mode.into(),
                    capture_options,
                    None,
                    SlaThresholds::default(),
                )?
            }
        };
        let replay_cpu_ms = thread_cpu_time_ms()
            .zip(cpu_start)
            .map(|(end, start)| end - start);
        let replay_call_ms = call_start.elapsed().as_secs_f64() * 1_000.0;
        if let Some(writer) = timing_writer.as_mut() {
            serde_json::to_writer(
                &mut *writer,
                &serde_json::json!({
                    "iteration": iteration,
                    "wall_time_ms": report.throughput.wall_time_ms,
                    "replay_call_ms": replay_call_ms,
                    "replay_cpu_ms": replay_cpu_ms,
                    "trace_load_ms": trace_load_ms,
                    "peak_rss_kib": peak_rss_kib(),
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
                capture_options,
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
