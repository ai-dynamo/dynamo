// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Global Router for the pool benchmark: an exact per-worker KV index over
//! every pool's workers, and a minimal router that picks a pool per request.
//!
//! The index is the frontend's CRTC thread-pool indexer (`--threads` lanes). How
//! KV events reach it is the benchmark's variable:
//!
//! - `event-plane`: subscribe to every pool's workers on Dynamo's event plane,
//!   exactly as the frontend's KV router does (direct ZMQ, gap recovery).
//! - `velo-relay`: each pool's `gr_pool_relay` forwards its workers' event-plane
//!   payloads over velo, one stream per worker.
//!
//! `POST /route` takes the prompt as little-endian `u32` token ids and returns
//! the pool whose best worker matches the most blocks (ties round-robin; see
//! `gr_relay/route.rs`). Once a second (`--stats-interval-ms`) the host appends a
//! JSON line to `--stats-out` with lane depths, events handed to the lanes,
//! events applied, route stats, pool sizes and the velo stream counters.

#[path = "gr_relay/core_streams.rs"]
mod core_streams;
#[path = "gr_relay/common.rs"]
mod relay_common;
#[path = "gr_relay/route.rs"]
mod route;

use std::io::Write as _;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::Context as _;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use bytes::Bytes;
use clap::{Parser, ValueEnum};
use dynamo_kv_router::indexer::{KvIndexerMetrics, bench_counters};
use dynamo_kv_router::services::indexer::backend::{Indexer, create_indexer_with_metrics};
use dynamo_llm::discovery::ModelManager;
use dynamo_llm::kv_router::indexer::subscribe_kv_events_for_bench;
use dynamo_runtime::{DistributedRuntime, Runtime, Worker};
use prometheus::core::Collector as _;
use prometheus::proto::MetricType;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;
use velo::VeloMetrics;

use core_streams::{StreamStats, register_relay_handler, write_peer_file};
use relay_common::{VeloOptions, build_velo};
use route::{PoolMap, Router};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Ingress {
    EventPlane,
    VeloRelay,
}

#[derive(Debug, Clone, Parser)]
#[command(about = "Global Router host for the pool KV-ingest benchmark")]
struct Args {
    #[arg(long, value_enum)]
    ingress: Ingress,
    /// HTTP port for `/route` and `/health`.
    #[arg(long, default_value_t = 8092)]
    port: u16,
    /// CRTC indexer lanes.
    #[arg(long, default_value_t = 4)]
    threads: usize,
    #[arg(long, default_value_t = 128)]
    block_size: u32,
    /// Pools are the Dynamo namespaces `<namespace-prefix><index:02>`.
    #[arg(long, default_value_t = 64)]
    pools: usize,
    #[arg(long, default_value = "pool-")]
    namespace_prefix: String,
    /// Model label for the event-plane subscriber's health metrics.
    #[arg(long, default_value = "gr-bench")]
    model_name: String,
    /// Velo listen address (the NIC address relays reach); `velo-relay` only.
    #[arg(long)]
    velo_bind: Option<SocketAddr>,
    #[arg(long, default_value_t = 4)]
    velo_lanes: u16,
    #[arg(long, default_value_t = 2)]
    velo_linger_ms: u64,
    /// Where the host writes its velo peer info for the relays.
    #[arg(long)]
    peer_file: Option<PathBuf>,
    #[arg(long)]
    stats_out: PathBuf,
    #[arg(long, default_value_t = 1000)]
    stats_interval_ms: u64,
}

fn pool_namespace(prefix: &str, pool: usize) -> String {
    format!("{prefix}{pool:02}")
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    dynamo_runtime::logging::init();
    Worker::from_settings()?.execute(move |runtime| app(runtime, args))
}

async fn app(runtime: Runtime, args: Args) -> anyhow::Result<()> {
    let applied = Arc::new(KvIndexerMetrics::new_unregistered());
    let indexer = create_indexer_with_metrics(args.block_size, args.threads, Arc::clone(&applied));
    let pools = Arc::new(PoolMap::new(args.pools));
    let streams = Arc::new(StreamStats::default());
    let velo_registry = prometheus::Registry::new();
    let cancel = CancellationToken::new();

    // Everything that must outlive the server.
    let mut _drt = None;
    let mut _subscriptions = Vec::new();
    let mut _velo = None;
    match args.ingress {
        Ingress::EventPlane => {
            let drt = DistributedRuntime::from_settings(runtime.clone()).await?;
            let models = ModelManager::new();
            for pool in 0..args.pools {
                let namespace = pool_namespace(&args.namespace_prefix, pool);
                let endpoint = drt
                    .namespace(&namespace)?
                    .component("backend")?
                    .endpoint("generate");
                let watch = models
                    .get_or_create_kv_source_membership_watch(&endpoint)
                    .await
                    .with_context(|| format!("membership watch for {namespace}"))?;
                _subscriptions.push(
                    subscribe_kv_events_for_bench(
                        endpoint.clone(),
                        indexer.clone(),
                        watch,
                        args.block_size,
                        args.model_name.clone(),
                        cancel.child_token(),
                    )
                    .await
                    .with_context(|| format!("KV event subscription for {namespace}"))?,
                );
                // Pool membership from the serving instances; workers only join.
                let client = endpoint.client().await?;
                let pools = Arc::clone(&pools);
                tokio::spawn(async move {
                    loop {
                        for worker in client.instance_ids() {
                            pools.insert(worker, pool as u32);
                        }
                        tokio::time::sleep(Duration::from_secs(1)).await;
                    }
                });
            }
            tracing::info!(pools = args.pools, "event-plane ingress subscribed");
            _drt = Some(drt);
        }
        Ingress::VeloRelay => {
            let bind = args
                .velo_bind
                .context("--velo-bind is required for velo-relay")?;
            let peer_file = args
                .peer_file
                .as_deref()
                .context("--peer-file is required for velo-relay")?;
            let metrics = Arc::new(VeloMetrics::register(&velo_registry)?);
            let velo = build_velo(
                VeloOptions {
                    bind,
                    lanes: args.velo_lanes,
                    linger_ms: args.velo_linger_ms,
                },
                Some(metrics),
            )
            .await?;
            register_relay_handler(
                &velo,
                indexer.clone(),
                Arc::clone(&pools),
                Arc::clone(&streams),
            )?;
            write_peer_file(peer_file, &velo)?;
            tracing::info!(%bind, peer_file = %peer_file.display(), "velo relay ingress ready");
            _velo = Some(velo);
        }
    }

    let router = Arc::new(Router::new(
        indexer.clone(),
        Arc::clone(&pools),
        args.block_size,
    ));
    {
        let router = Arc::clone(&router);
        let streams = Arc::clone(&streams);
        let path = args.stats_out.clone();
        let interval = Duration::from_millis(args.stats_interval_ms.max(100));
        std::thread::Builder::new()
            .name("gr-stats".to_string())
            .spawn(move || {
                if let Err(error) =
                    write_stats(&path, interval, &router, &applied, &streams, &velo_registry)
                {
                    tracing::error!(%error, "stats writer stopped");
                }
            })?;
    }

    let result = serve(args.port, router).await;
    cancel.cancel();
    result
}

async fn route_handler(State(router): State<Arc<Router>>, body: Bytes) -> Response {
    let started = Instant::now();
    router.stats.routes.fetch_add(1, Ordering::Relaxed);
    if !body.len().is_multiple_of(4) {
        router.stats.errors.fetch_add(1, Ordering::Relaxed);
        return (
            StatusCode::BAD_REQUEST,
            "body must be little-endian u32 token ids",
        )
            .into_response();
    }
    let tokens: Vec<u32> = body
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().expect("4-byte chunk")))
        .collect();
    let reply = router.route(&tokens).await;
    router.stats.record(started.elapsed());
    match reply {
        Ok(reply) => Json(reply).into_response(),
        Err(error) => {
            router.stats.errors.fetch_add(1, Ordering::Relaxed);
            (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()).into_response()
        }
    }
}

/// Serve `/route` and `/health` with one hyper connection task per accepted
/// socket (as the frontend's server does), with Nagle off.
async fn serve(port: u16, router: Arc<Router>) -> anyhow::Result<()> {
    use hyper_util::rt::{TokioExecutor, TokioIo};
    use hyper_util::server::conn::auto::Builder;
    use hyper_util::service::TowerToHyperService;

    let app = axum::Router::new()
        .route("/route", post(route_handler))
        .route("/health", get(|| async { "ok" }))
        .with_state(router);
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await?;
    tracing::info!(port, "GR routing on /route");
    loop {
        let (stream, _) = listener.accept().await?;
        let _ = stream.set_nodelay(true);
        let service = TowerToHyperService::new(app.clone());
        tokio::spawn(async move {
            let _ = Builder::new(TokioExecutor::new())
                .serve_connection(TokioIo::new(stream), service)
                .await;
        });
    }
}

fn write_stats(
    path: &std::path::Path,
    interval: Duration,
    router: &Router,
    applied: &KvIndexerMetrics,
    streams: &StreamStats,
    velo_registry: &prometheus::Registry,
) -> anyhow::Result<()> {
    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
    loop {
        std::thread::sleep(interval);
        let lanes = lane_depths(&router.indexer);
        let (events, stored_blocks, removed_blocks, cleared) = bench_counters::snapshot();
        let sizes = router.pools.sizes();
        let line = json!({
            "t": SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64(),
            "lane_depths": lanes,
            "lane_backlog": lanes.iter().sum::<usize>(),
            "lanes_in": {
                "events": events,
                "stored_blocks": stored_blocks,
                "removed_blocks": removed_blocks,
                "cleared": cleared,
            },
            "applied": applied_by_label(applied),
            "route": router.stats.snapshot(),
            "pools_mapped_workers": sizes.iter().sum::<u32>(),
            "pools_min_workers": sizes.iter().copied().min().unwrap_or(0),
            "streams": streams.snapshot(),
            "velo": gather(velo_registry),
        });
        serde_json::to_writer(&mut file, &line)?;
        file.write_all(b"\n")?;
        file.flush()?;
    }
}

fn lane_depths(indexer: &Indexer) -> Vec<usize> {
    match indexer {
        Indexer::Concurrent { primary, .. } => primary.queue_depths(),
        _ => Vec::new(),
    }
}

/// `kv_cache_events_applied` by `"<event_type>/<status>"`.
fn applied_by_label(metrics: &KvIndexerMetrics) -> Value {
    let mut out = serde_json::Map::new();
    for family in metrics.kv_cache_events_applied.collect() {
        for metric in family.get_metric() {
            let label = metric
                .get_label()
                .iter()
                .map(|pair| pair.value())
                .collect::<Vec<_>>()
                .join("/");
            out.insert(label, json!(metric.get_counter().value()));
        }
    }
    Value::Object(out)
}

/// Each velo metric family summed over its label sets: counters and gauges by
/// value, histograms as `<name>_count` and `<name>_sum`.
fn gather(registry: &prometheus::Registry) -> Value {
    let mut out = serde_json::Map::new();
    for family in registry.gather() {
        let name = family.name();
        match family.get_field_type() {
            MetricType::COUNTER => {
                let total: f64 = family
                    .get_metric()
                    .iter()
                    .map(|m| m.get_counter().value())
                    .sum();
                out.insert(name.to_string(), json!(total));
            }
            MetricType::GAUGE => {
                let total: f64 = family
                    .get_metric()
                    .iter()
                    .map(|m| m.get_gauge().value())
                    .sum();
                out.insert(name.to_string(), json!(total));
            }
            MetricType::HISTOGRAM => {
                let (count, sum) = family.get_metric().iter().fold((0u64, 0.0), |(c, s), m| {
                    let h = m.get_histogram();
                    (c + h.get_sample_count(), s + h.get_sample_sum())
                });
                out.insert(format!("{name}_count"), json!(count));
                out.insert(format!("{name}_sum"), json!(sum));
            }
            _ => {}
        }
    }
    Value::Object(out)
}
