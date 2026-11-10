// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The selection core for the Global Router full-KV benchmark.
//!
//! It builds the standalone selection service exactly as
//! `python -m dynamo.select_service` does (router config from `DYN_*`
//! environment variables, default policy, `--threads` CRTC lanes, the stock
//! HTTP API), and only swaps how KV events arrive:
//!
//! - `zmq-direct`: the stock ingress, one ZMQ SUB per worker on its raw vLLM
//!   event stream.
//! - `velo-relay`: one velo stream per worker from the pool relays in
//!   `select_service_bench --event-transport velo`, carrying router events.
//!
//! Once a second (`--stats-interval-ms`) it appends one JSON line to
//! `--stats-out`: CRTC lane queue depths, the ZMQ listener's ingest counters,
//! the relay consumers' counters and velo's metrics.

#[path = "gr_relay/common.rs"]
mod relay_common;
#[path = "gr_relay/selection_core.rs"]
mod relay_core;

use std::io::Write as _;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Context as _;
use clap::{Parser, ValueEnum};
use dynamo_kv_router::WorkerType;
use dynamo_kv_router::scheduling::config::try_kv_router_config_from_dynamo_env;
use dynamo_kv_router::services::indexer::listener::bench_counters;
use dynamo_kv_router::services::selection::{
    KvIndexSource, SelectionCacheConfig, SelectionServiceConfig, run_server,
};
use prometheus::proto::MetricType;
use serde_json::{Value, json};
use velo::VeloMetrics;

use relay_common::{VeloOptions, build_velo};
use relay_core::{BenchIngress, IngressMode, write_peer_file};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Ingress {
    ZmqDirect,
    VeloRelay,
}

#[derive(Debug, Parser)]
#[command(about = "Selection core with a selectable KV-event ingress, for the GR full-KV bench")]
struct Args {
    #[arg(long, default_value_t = 8092)]
    port: u16,
    /// CRTC indexer lanes, as `dynamo.select_service --threads`.
    #[arg(long, default_value_t = 4)]
    threads: usize,
    #[arg(long, value_enum)]
    ingress: Ingress,
    /// Velo listen address (the NIC address relays reach); `velo-relay` only.
    #[arg(long)]
    velo_bind: Option<SocketAddr>,
    #[arg(long, default_value_t = 4)]
    velo_lanes: u16,
    #[arg(long, default_value_t = 2)]
    velo_linger_ms: u64,
    /// Where the core writes its velo peer info for the relays.
    #[arg(long)]
    peer_file: Option<PathBuf>,
    #[arg(long)]
    stats_out: PathBuf,
    #[arg(long, default_value_t = 1000)]
    stats_interval_ms: u64,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let kv_router_config = try_kv_router_config_from_dynamo_env().map_err(anyhow::Error::msg)?;
    let config = SelectionServiceConfig {
        port: args.port,
        threads: args.threads,
        indexer_peers: Vec::new(),
        replica_sync_port: None,
        replica_sync_peers: Vec::new(),
        kv_router_config,
        selection_cache: SelectionCacheConfig::default(),
        session_affinity_ttl: None,
    };
    let mode = match args.ingress {
        Ingress::ZmqDirect => IngressMode::ZmqDirect,
        Ingress::VeloRelay => IngressMode::VeloRelay,
    };
    let ingress = BenchIngress::new(mode);

    // The CLI's runtime: tokio's default multi-thread runtime.
    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async move {
        let service = config
            .service_builder(
                WorkerType::Aggregated,
                dynamo_router_policies::default_registry(),
            )
            .kv_index(KvIndexSource::Owned(ingress.clone()))
            .build()
            .await?;

        let metrics_registry = prometheus::Registry::new();
        let _velo = if mode == IngressMode::VeloRelay {
            let bind = args
                .velo_bind
                .context("--velo-bind is required for velo-relay")?;
            let peer_file = args
                .peer_file
                .as_deref()
                .context("--peer-file is required for velo-relay")?;
            let metrics = Arc::new(VeloMetrics::register(&metrics_registry)?);
            let velo = build_velo(
                VeloOptions {
                    bind,
                    lanes: args.velo_lanes,
                    linger_ms: args.velo_linger_ms,
                },
                Some(metrics),
            )
            .await?;
            relay_core::register_relay_handler(&velo, Arc::clone(&ingress))?;
            write_peer_file(peer_file, &velo)?;
            tracing::info!(%bind, peer_file = %peer_file.display(), "velo relay ingress ready");
            Some(velo)
        } else {
            None
        };

        let stats_ingress = Arc::clone(&ingress);
        let stats_out = args.stats_out.clone();
        let interval = Duration::from_millis(args.stats_interval_ms.max(100));
        std::thread::Builder::new()
            .name("gr-stats".to_string())
            .spawn(move || {
                if let Err(error) =
                    write_stats(&stats_out, interval, &stats_ingress, &metrics_registry)
                {
                    tracing::error!(%error, "stats writer stopped");
                }
            })?;

        run_server(args.port, service).await
    })
}

fn write_stats(
    path: &std::path::Path,
    interval: Duration,
    ingress: &BenchIngress,
    registry: &prometheus::Registry,
) -> anyhow::Result<()> {
    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
    loop {
        std::thread::sleep(interval);
        let lanes = ingress.lane_depths();
        let line = json!({
            "t": SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64(),
            "lane_depths": lanes,
            "lane_backlog": lanes.iter().sum::<usize>(),
            "zmq": bench_counters::snapshot()
                .into_iter()
                .map(|(name, value)| (name.to_ascii_lowercase(), json!(value)))
                .collect::<serde_json::Map<_, _>>(),
            "relay": ingress.relay.snapshot(),
            "velo": gather(registry),
        });
        serde_json::to_writer(&mut file, &line)?;
        file.write_all(b"\n")?;
        file.flush()?;
    }
}

/// Each metric family summed over its label sets: counters and gauges by
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
