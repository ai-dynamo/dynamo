// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Selection replay benchmark: drives the KV router's selection core in-process
//! with a deployment-shaped workload. See `SELECTION_BENCH.md` for the protocol.

#[path = "selection_replay/corpus.rs"]
mod corpus;
#[path = "selection_replay/engine.rs"]
mod engine;
#[path = "selection_replay/host.rs"]
mod host;
#[path = "selection_replay/kv_cache.rs"]
mod kv_cache;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, ValueEnum};
use dynamo_bench::kv_router_common::issuer::{
    parse_cpu_list, pin_current_thread_to_cpus, validate_cpu_partition,
};
use dynamo_kv_router::KvRouterConfig;
use dynamo_kv_router::scheduling::policy_config::BusyThresholds;
use serde::Serialize;

use corpus::{Corpus, Omitted, WorkloadParams};
use engine::{Limits, Mode, RunConfig, RunResult};
use host::{BenchHost, HostConfig, worker_request};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ModeArg {
    Fixed,
    Saturation,
}

#[derive(Parser, Debug)]
#[command(about = "Replay a deployment-shaped workload through the KV router's selection core")]
struct Args {
    /// Trace to replay: a Weka corpus directory or a Mooncake JSONL file.
    #[arg(long)]
    trace: Option<PathBuf>,
    /// The workload, `agentx` by default; ignored with `--load-corpus`.
    #[command(flatten)]
    workload: WorkloadParams,
    #[arg(long)]
    save_corpus: Option<PathBuf>,
    /// Replay a saved corpus instead of generating one.
    #[arg(long)]
    load_corpus: Option<PathBuf>,
    /// Generate (and save) the corpus, then exit.
    #[arg(long)]
    generate_only: bool,

    /// JSON object merged over the router config: `KvRouterConfig`'s defaults with
    /// replica sync on, as the AgentX frontend runs.
    #[arg(long)]
    router_config_json: Option<String>,
    /// Busy thresholds, as the frontend's runtime config sets them.
    #[arg(long)]
    active_decode_blocks_threshold: Option<f64>,
    #[arg(long)]
    active_prefill_tokens_threshold: Option<usize>,
    #[arg(long)]
    active_prefill_tokens_threshold_frac: Option<f64>,

    #[arg(long, value_enum, default_value = "fixed")]
    mode: ModeArg,
    /// Fixed mode: wall time = trace time / scale.
    #[arg(long, conflicts_with = "target_rate")]
    time_scale: Option<f64>,
    /// Fixed mode: the scale that offers this many window requests per second.
    #[arg(long)]
    target_rate: Option<f64>,
    /// Saturation mode: admissions allowed in flight at once.
    #[arg(long, default_value_t = 512)]
    admission_window: usize,

    /// CPUs for the router: its runtime and indexer threads.
    #[arg(long)]
    backend_cpus: Option<String>,
    /// CPUs for the bench's threads, one each: the request issuer, the sampler, the
    /// engine issuers, and prompt preparation.
    #[arg(long)]
    issuer_cpus: Option<String>,
    /// Threads that act as the booked workers: they cache prompts and output blocks,
    /// publish the KV events, and mark and release bookings.
    #[arg(long, default_value_t = 4)]
    engine_issuers: usize,
    #[arg(long, default_value_t = 2)]
    prep_threads: usize,

    #[arg(long, default_value = "selection_replay_result.json")]
    result_json: PathBuf,
    /// `perf record --control fifo:CTL,ACK`: enabled for the measured window only.
    #[arg(long)]
    perf_ctl_fifo: Option<PathBuf>,
    #[arg(long)]
    perf_ack_fifo: Option<PathBuf>,
    #[arg(long, default_value = "warn")]
    log_level: String,
}

fn router_config(json: Option<&str>) -> Result<KvRouterConfig> {
    let config = KvRouterConfig {
        router_replica_sync: true,
        ..KvRouterConfig::default()
    };
    let Some(json) = json else {
        return Ok(config);
    };
    let overrides: serde_json::Value =
        serde_json::from_str(json).context("--router-config-json")?;
    let serde_json::Value::Object(overrides) = overrides else {
        bail!("--router-config-json must be a JSON object");
    };
    let mut merged = serde_json::to_value(&config)?;
    for (key, value) in overrides {
        ensure!(
            merged.get(&key).is_some(),
            "--router-config-json: unknown field {key}"
        );
        merged[&key] = value;
    }
    serde_json::from_value(merged).context("--router-config-json")
}

#[derive(Serialize)]
struct Report {
    argv: Vec<String>,
    workload: WorkloadParams,
    corpus_digest: String,
    omitted: Omitted,
    /// Window requests per second of trace time: the offered rate at time scale 1.
    trace_window_rps: f64,
    router: KvRouterConfig,
    limits: Limits,
    runtime_threads: usize,
    run: Option<RunResult>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_max_level(
            args.log_level
                .parse::<tracing_subscriber::filter::LevelFilter>()
                .context("--log-level")?,
        )
        .init();
    if cfg!(feature = "router-bench") {
        bail!(
            "built with `router-bench`, which changes the code under test; build with \
             `--no-default-features --features selection-replay`"
        );
    }

    let cpu_list = |list: &Option<String>| -> Result<Vec<usize>> {
        Ok(list
            .as_deref()
            .map(parse_cpu_list)
            .transpose()?
            .unwrap_or_default())
    };
    let backend_cpus = cpu_list(&args.backend_cpus)?;
    let issuer_cpus = cpu_list(&args.issuer_cpus)?;
    validate_cpu_partition(&issuer_cpus, None, &backend_cpus)?;
    ensure!(
        backend_cpus.is_empty() || !issuer_cpus.is_empty(),
        "--backend-cpus needs --issuer-cpus too, or the bench's threads run on the router's CPUs"
    );

    let corpus = Arc::new(match &args.load_corpus {
        Some(path) => {
            let corpus = Corpus::load(path)?;
            eprintln!("loaded corpus {} ({})", path.display(), corpus.digest);
            corpus
        }
        None => {
            let Some(trace) = &args.trace else {
                bail!("--trace is required unless --load-corpus is given");
            };
            let started = std::time::Instant::now();
            let threads = std::thread::available_parallelism().map_or(8, |threads| threads.get());
            let corpus = corpus::generate(trace, &args.workload, threads)?;
            eprintln!(
                "generated corpus {} in {:.1} s: {} requests, {:?} omitted",
                corpus.digest,
                started.elapsed().as_secs_f64(),
                corpus.requests.len(),
                corpus.omitted,
            );
            if let Some(path) = &args.save_corpus {
                corpus.save(path)?;
            }
            corpus
        }
    });

    let router = router_config(args.router_config_json.as_deref())?;
    let params = &corpus.params;
    let host_config = HostConfig {
        router: router.clone(),
        block_size: params.block_size,
        busy: BusyThresholds {
            prefill_busy_threshold: args.active_prefill_tokens_threshold,
            prefill_busy_threshold_frac: args.active_prefill_tokens_threshold_frac,
            decode_busy_threshold_frac: args.active_decode_blocks_threshold,
        },
    };
    let workers: Vec<_> = (1..=params.workers as u64)
        .map(|worker_id| {
            worker_request(
                worker_id,
                params.block_size,
                params.dp_size,
                params.max_num_batched_tokens as u64,
                params.num_gpu_blocks as u64,
            )
        })
        .collect();

    // The router's runtime and indexer threads inherit this thread's CPUs.
    pin_current_thread_to_cpus(&backend_cpus)?;
    let router_cpus = if backend_cpus.is_empty() {
        std::thread::available_parallelism().map_or(8, |threads| threads.get())
    } else {
        backend_cpus.len()
    };
    let runtime_threads = router_cpus
        .saturating_sub(router.router_event_threads as usize)
        .max(1);

    let window_rate = engine::corpus_window_rate(&corpus)?;
    let limits = Limits::default();
    let run = if args.generate_only {
        None
    } else {
        let mode = match args.mode {
            ModeArg::Saturation => Mode::Saturation {
                admission_window: args.admission_window,
            },
            ModeArg::Fixed => Mode::Fixed {
                scale: args
                    .target_rate
                    .map_or(args.time_scale.unwrap_or(1.0), |rate| rate / window_rate),
            },
        };
        let config = RunConfig {
            mode,
            limits,
            issuer_cpus,
            engine_issuers: args.engine_issuers,
            prep_threads: args.prep_threads,
            perf_ctl_fifo: args.perf_ctl_fifo.clone(),
            perf_ack_fifo: args.perf_ack_fifo.clone(),
        };
        // A fresh runtime and service, so nothing carries over from setup.
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(runtime_threads)
            .enable_all()
            .build()?;
        let host = Arc::new(runtime.block_on(BenchHost::start(&host_config, &workers))?);
        let result = engine::run(&corpus, &host, &runtime, &config);
        runtime.block_on(host.shutdown());
        let result = result?;
        print_summary(&result);
        Some(result)
    };

    let report = Report {
        argv: std::env::args().collect(),
        workload: corpus.params.clone(),
        corpus_digest: corpus.digest.clone(),
        omitted: corpus.omitted.clone(),
        trace_window_rps: window_rate,
        router,
        limits,
        runtime_threads,
        run,
    };
    std::fs::write(&args.result_json, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("write {}", args.result_json.display()))?;
    Ok(())
}

fn print_summary(result: &RunResult) {
    let mode = match result.mode {
        Mode::Fixed { scale } => format!("fixed x{scale:.3}"),
        Mode::Saturation { admission_window } => format!("saturation w{admission_window}"),
    };
    let ms =
        |percentile: fn(&dynamo_bench::kv_router_common::results::PercentileStats<u64>) -> u64| {
            result
                .admission_latency_ns
                .as_ref()
                .map_or(0.0, |stats| percentile(stats) as f64 / 1e6)
        };
    println!(
        "{mode}: {:.0} req/s admitted, admission p50 {:.3} ms p99 {:.3} ms, \
         in-flight median {}, index backlog max {} end {}, kept_up={} {}",
        result.achieved_rps,
        ms(|stats| stats.p50),
        ms(|stats| stats.p99),
        result.booked_in_flight.median,
        result.max_index_backlog,
        result.index_backlog_at_window_end,
        result.kept_up,
        result.failure_reasons.join("; "),
    );
}
