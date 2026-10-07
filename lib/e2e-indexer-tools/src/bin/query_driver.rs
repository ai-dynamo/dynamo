// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! EXPERIMENT ONLY: side query driver for a serving indexer.
//!
//! Issues the serving indexer's real query RPC (`kv_indexer_query` on the worker component,
//! through the same `PushRouter` round-robin client a frontend's `RemoteIndexer` uses) with the
//! timed lookups of phantoms `[first, first + count)`, salted exactly like the phantom
//! publishers' events and scheduled open loop on the same wall-clock mapping, so lookups hit
//! the blocks the phantoms stored. Reports RTT and issue-lag percentiles, achieved rates, and
//! the fraction of lookups whose own phantom matched.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use clap::Parser;
use dynamo_e2e_indexer_tools::plan::{
    DEFAULT_WORKER_ID_BASE, PhantomLayout, TimeMap, natural_rates, resolve_speedup, salt_hash,
    unix_now_us,
};
use dynamo_e2e_indexer_tools::stats::{Counter, Latency, summarize_ms};
use dynamo_e2e_indexer_tools::stream::{Manifest, Queries, Sections, read_base};
use dynamo_kv_router::indexer::{
    IndexerQueryRequest, IndexerQueryResponse, KV_INDEXER_QUERY_ENDPOINT,
};
use dynamo_kv_router::protocols::{LocalBlockHash, WorkerWithDpRank};
use dynamo_runtime::pipeline::{ManyOut, PushRouter, RouterMode, SingleIn};
use dynamo_runtime::{DistributedRuntime, Runtime, Worker};
use futures::StreamExt;
use serde_json::json;
use tokio::time::Instant;

#[derive(Parser, Debug, Clone)]
#[command(about = "EXPERIMENT ONLY: drive a serving indexer's kv_indexer_query endpoint")]
struct Args {
    #[arg(long)]
    streams: PathBuf,
    #[arg(long)]
    total_phantoms: u64,
    /// First phantom whose lookups this process issues.
    #[arg(long, default_value_t = 0)]
    first_phantom: u64,
    /// Phantoms whose lookups this process issues (default: all from --first-phantom).
    #[arg(long)]
    count: Option<u64>,
    #[arg(long, default_value_t = DEFAULT_WORKER_ID_BASE, value_parser = parse_u64)]
    worker_id_base: u64,
    #[arg(long, default_value_t = 0)]
    salt_seed: u64,
    #[arg(long)]
    speedup: Option<f64>,
    #[arg(long)]
    target_write_blocks_per_sec: Option<f64>,
    #[arg(long)]
    start_at_unix_ms: u64,
    #[arg(long, default_value_t = 0)]
    start_spread_ms: u64,
    #[arg(long)]
    duration_s: Option<f64>,
    /// Worker component hosting the served indexer: <namespace>.<component>.
    #[arg(long)]
    component: String,
    /// Model name the serving indexer registered (IndexerQueryRequest.model_name).
    #[arg(long)]
    model_name: String,
    /// Ask for device-tier overlap only. Frontend routing sends `false` (tiered).
    #[arg(long, default_value_t = false)]
    device_only: bool,
    /// Maximum outstanding queries; the schedule waits (and lag grows) beyond it.
    #[arg(long, default_value_t = 1024)]
    max_inflight: usize,
    /// Issue only this deterministic fraction of the scheduled lookups.
    #[arg(long, default_value_t = 1.0)]
    query_fraction: f64,
    #[arg(long, default_value_t = 5.0)]
    report_interval_s: f64,
    #[arg(long, default_value_t = 60.0)]
    wait_for_indexer_s: f64,
    #[arg(long)]
    summary_out: Option<PathBuf>,
}

fn parse_u64(value: &str) -> Result<u64, String> {
    match value.strip_prefix("0x") {
        Some(hex) => u64::from_str_radix(hex, 16),
        None => value.parse(),
    }
    .map_err(|error| error.to_string())
}

#[derive(Default)]
struct Stats {
    issued: Counter,
    completed: Counter,
    errors: Counter,
    query_blocks: Counter,
    /// Lookups whose own phantom had a nonzero device overlap.
    self_hits: Counter,
    /// Sum over lookups of the own phantom's matched blocks.
    self_matched_blocks: Counter,
    /// Sum over lookups of the number of workers in the response.
    response_workers: Counter,
    rtt: Latency,
    issue_lag: Latency,
    inflight: AtomicU64,
    first_issue_us: AtomicU64,
    last_complete_us: AtomicU64,
    stop: AtomicBool,
}

struct Scheduled {
    deadline_unix_us: u64,
    phantom: u32,
    query: u32,
}

struct PhantomQueries {
    worker_id: u64,
    salt_key: u64,
    queries: Arc<Queries>,
}

fn main() -> Result<()> {
    dynamo_runtime::logging::init();
    let args = Args::parse();
    let worker = Worker::from_settings()?;
    worker.execute(move |runtime| app(runtime, args))
}

async fn app(runtime: Runtime, args: Args) -> Result<()> {
    let manifest = Manifest::read(&args.streams)?;
    let layout = PhantomLayout {
        total: args.total_phantoms,
        worker_id_base: args.worker_id_base,
        salt_seed: args.salt_seed,
    };
    layout.validate(&manifest)?;
    let count = args
        .count
        .unwrap_or(args.total_phantoms.saturating_sub(args.first_phantom));
    ensure!(
        count > 0 && args.first_phantom + count <= args.total_phantoms,
        "phantom range is empty or exceeds --total-phantoms"
    );
    ensure!(
        args.query_fraction > 0.0 && args.query_fraction <= 1.0,
        "--query-fraction must be in (0, 1]"
    );
    let speedup = resolve_speedup(
        &manifest,
        &layout,
        args.speedup,
        args.target_write_blocks_per_sec,
    )?;
    let map = TimeMap {
        start_at_unix_us: args.start_at_unix_ms * 1000,
        t0_us: manifest.t0_us,
        speedup,
    };
    let phantoms = args.first_phantom..args.first_phantom + count;
    let local_rates = natural_rates(&manifest, &layout, phantoms.clone());

    let mut loaded: Vec<(usize, Arc<Queries>)> = Vec::new();
    let mut phantom_queries = Vec::with_capacity(count as usize);
    for index in phantoms.clone() {
        let base = layout.base_of(index, manifest.bases.len());
        let queries = match loaded.iter().find(|(loaded_base, _)| *loaded_base == base) {
            Some((_, queries)) => queries.clone(),
            None => {
                let path = args.streams.join(&manifest.bases[base].file);
                let stream = read_base(
                    &path,
                    Sections {
                        events: false,
                        queries: true,
                    },
                )?;
                let queries = Arc::new(stream.queries);
                loaded.push((base, queries.clone()));
                queries
            }
        };
        phantom_queries.push(PhantomQueries {
            worker_id: layout.worker_id(index),
            salt_key: layout.salt_key(index),
            queries,
        });
    }

    let stop_at = args
        .duration_s
        .map(|seconds| map.start_at_unix_us + (seconds * 1e6) as u64);
    let mut schedule = Vec::new();
    for (local, (index, phantom)) in phantoms.clone().zip(&phantom_queries).enumerate() {
        let delay = layout.start_delay_us(index, args.start_spread_ms * 1000);
        for query in 0..phantom.queries.len() {
            // Deterministic thinning keyed by (phantom, query).
            let draw = dynamo_e2e_indexer_tools::plan::fmix64(phantom.salt_key ^ query as u64);
            if (draw as f64 / u64::MAX as f64) >= args.query_fraction {
                continue;
            }
            let deadline = map.wall_us(phantom.queries.ts_us[query], delay);
            if stop_at.is_some_and(|stop| deadline >= stop) {
                continue;
            }
            schedule.push(Scheduled {
                deadline_unix_us: deadline,
                phantom: local as u32,
                query: query as u32,
            });
        }
    }
    schedule.sort_unstable_by_key(|item| item.deadline_unix_us);
    let phantom_queries = Arc::new(phantom_queries);

    let (namespace, component) = args
        .component
        .split_once('.')
        .context("--component must be <namespace>.<component>")?;
    let drt = DistributedRuntime::from_settings(runtime.clone()).await?;
    let client = drt
        .namespace(namespace)?
        .component(component)?
        .endpoint(KV_INDEXER_QUERY_ENDPOINT)
        .client()
        .await?;
    tokio::time::timeout(
        Duration::from_secs_f64(args.wait_for_indexer_s),
        client.wait_for_instances(),
    )
    .await
    .context("timed out waiting for the served indexer's query endpoint")??;
    let router = Arc::new(
        PushRouter::<IndexerQueryRequest, IndexerQueryResponse>::from_client_no_fault_detection(
            client,
            RouterMode::RoundRobin,
        )
        .await?,
    );

    let plan = json!({
        "kind": "plan",
        "phantoms": [args.first_phantom, args.first_phantom + count],
        "speedup": speedup,
        "scheduled_queries": schedule.len(),
        "expected_queries_per_s": local_rates.queries * speedup * args.query_fraction,
        "expected_query_blocks_per_s": local_rates.query_blocks * speedup * args.query_fraction,
        "start_at_unix_ms": args.start_at_unix_ms,
        "device_only": args.device_only,
        "model_name": args.model_name,
    });
    println!("{plan}");

    let stats = Arc::new(Stats {
        first_issue_us: AtomicU64::new(u64::MAX),
        ..Stats::default()
    });
    let reporter = {
        let stats = stats.clone();
        let interval = Duration::from_secs_f64(args.report_interval_s);
        tokio::spawn(async move {
            let mut last = std::time::Instant::now();
            loop {
                tokio::time::sleep(interval).await;
                let seconds = last.elapsed().as_secs_f64();
                last = std::time::Instant::now();
                let completed = stats.completed.take_interval();
                let hits = stats.self_hits.take_interval();
                let line = json!({
                    "kind": "interval",
                    "t_unix_ms": unix_now_us() / 1000,
                    "issued_per_s": stats.issued.take_interval() as f64 / seconds,
                    "completed_per_s": completed as f64 / seconds,
                    "query_blocks_per_s": stats.query_blocks.take_interval() as f64 / seconds,
                    "self_hit_fraction": if completed > 0 { hits as f64 / completed as f64 } else { 0.0 },
                    "errors": stats.errors.total(),
                    "inflight": stats.inflight.load(Ordering::Relaxed),
                    "rtt": summarize_ms(&stats.rtt.take_interval()),
                    "issue_lag": summarize_ms(&stats.issue_lag.take_interval()),
                });
                println!("{line}");
            }
        })
    };

    let epoch = Instant::now();
    let epoch_unix_us = unix_now_us();
    let semaphore = Arc::new(tokio::sync::Semaphore::new(args.max_inflight.max(1)));
    let interrupted = {
        let stats = stats.clone();
        tokio::spawn(async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                stats.stop.store(true, Ordering::Relaxed);
            }
        })
    };
    for (sequence, item) in schedule.iter().enumerate() {
        if stats.stop.load(Ordering::Relaxed) {
            break;
        }
        let deadline =
            epoch + Duration::from_micros(item.deadline_unix_us.saturating_sub(epoch_unix_us));
        tokio::time::sleep_until(deadline).await;
        let permit = semaphore.clone().acquire_owned().await?;
        let issued_at = unix_now_us();
        stats
            .issue_lag
            .record(sequence, issued_at.saturating_sub(item.deadline_unix_us));
        stats.first_issue_us.fetch_min(issued_at, Ordering::Relaxed);

        let phantom = &phantom_queries[item.phantom as usize];
        let (_, hashes) = phantom.queries.get(item.query as usize);
        let request = IndexerQueryRequest {
            model_name: args.model_name.clone(),
            block_hashes: hashes
                .iter()
                .map(|&hash| LocalBlockHash(salt_hash(hash, phantom.salt_key)))
                .collect(),
            device_only: args.device_only,
        };
        let query_blocks = request.block_hashes.len() as u64;
        let worker = WorkerWithDpRank::new(phantom.worker_id, 0);
        let router = router.clone();
        let stats = stats.clone();
        stats.issued.add(1);
        stats.inflight.fetch_add(1, Ordering::Relaxed);
        tokio::spawn(async move {
            let _permit = permit;
            let started = Instant::now();
            let response = async {
                let mut stream: ManyOut<IndexerQueryResponse> =
                    router.round_robin(SingleIn::new(request)).await?;
                stream
                    .next()
                    .await
                    .context("served indexer returned an empty response")
            }
            .await;
            let rtt_us = started.elapsed().as_micros() as u64;
            stats.inflight.fetch_sub(1, Ordering::Relaxed);
            match response {
                Ok(IndexerQueryResponse::TieredScores(scores)) => {
                    stats.rtt.record(sequence, rtt_us);
                    stats.completed.add(1);
                    stats.query_blocks.add(query_blocks);
                    stats
                        .response_workers
                        .add(scores.device.scores.len() as u64);
                    let own = scores
                        .device
                        .scores
                        .iter()
                        .find(|(candidate, _)| *candidate == worker)
                        .map_or(0, |(_, matched)| u64::from(*matched));
                    if own > 0 {
                        stats.self_hits.add(1);
                    }
                    stats.self_matched_blocks.add(own);
                    stats
                        .last_complete_us
                        .fetch_max(unix_now_us(), Ordering::Relaxed);
                }
                Ok(IndexerQueryResponse::Error(message)) => {
                    if stats.errors.total() == 0 {
                        tracing::error!(%message, "served indexer query failed");
                    }
                    stats.errors.add(1);
                }
                Err(error) => {
                    if stats.errors.total() == 0 {
                        tracing::error!(%error, "served indexer query failed");
                    }
                    stats.errors.add(1);
                }
            }
        });
    }
    // Drain outstanding queries.
    let _ = semaphore
        .acquire_many(args.max_inflight.max(1) as u32)
        .await?;
    reporter.abort();
    interrupted.abort();

    let first = stats.first_issue_us.load(Ordering::Relaxed);
    let last = stats.last_complete_us.load(Ordering::Relaxed);
    let wall_s = if first == u64::MAX || last < first {
        0.0
    } else {
        (last - first) as f64 / 1e6
    };
    let completed = stats.completed.total();
    let summary = json!({
        "kind": "summary",
        "plan": plan,
        "issued": stats.issued.total(),
        "completed": completed,
        "errors": stats.errors.total(),
        "wall_s": wall_s,
        "achieved_queries_per_s": if wall_s > 0.0 { completed as f64 / wall_s } else { 0.0 },
        "achieved_query_blocks_per_s": if wall_s > 0.0 { stats.query_blocks.total() as f64 / wall_s } else { 0.0 },
        "self_hit_fraction": if completed > 0 { stats.self_hits.total() as f64 / completed as f64 } else { 0.0 },
        "self_matched_block_fraction": if stats.query_blocks.total() > 0 {
            stats.self_matched_blocks.total() as f64 / stats.query_blocks.total() as f64
        } else { 0.0 },
        "mean_response_workers": if completed > 0 { stats.response_workers.total() as f64 / completed as f64 } else { 0.0 },
        "rtt": summarize_ms(&stats.rtt.cumulative()),
        "issue_lag": summarize_ms(&stats.issue_lag.cumulative()),
    });
    println!("{summary}");
    if let Some(path) = &args.summary_out {
        std::fs::write(path, serde_json::to_string_pretty(&summary)?)?;
    }
    Ok(())
}
