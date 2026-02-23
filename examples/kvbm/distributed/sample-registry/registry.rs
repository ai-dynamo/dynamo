// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Registry Hub Binary
//!
//! Runs the distributed object registry hub for KV cache block coordination.
//!
//! # Usage
//!
//! ```bash
//! # Build and run from this directory
//! cd examples/kvbm/distributed/sample-registry
//! cargo run --release
//!
//! # Or build from workspace root
//! cargo build --manifest-path examples/kvbm/distributed/sample-registry/Cargo.toml --release
//!
//! # Run with custom settings via environment variables
//! DYN_REGISTRY_HUB_CAPACITY=10000000 \
//! DYN_REGISTRY_HUB_QUERY_ADDR=tcp://*:6000 \
//! DYN_REGISTRY_HUB_REGISTER_ADDR=tcp://*:6001 \
//! cargo run --manifest-path examples/kvbm/distributed/sample-registry/Cargo.toml --release
//! ```
//!
//! # Environment Variables
//!
//! - `DYN_REGISTRY_HUB_CAPACITY`: Registry capacity (default: 1000000)
//! - `DYN_REGISTRY_HUB_QUERY_ADDR`: Query address (default: tcp://*:5555)
//! - `DYN_REGISTRY_HUB_REGISTER_ADDR`: Register address (default: tcp://*:5556)
//! - `DYN_REGISTRY_HUB_METRICS_ADDR`: Metrics HTTP address (default: 0.0.0.0:9108)
//! - `DYN_REGISTRY_HUB_PLUGIN_METRICS`: Enable core query/register metrics plugin (default: true)

use anyhow::Result;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::signal;
use tokio_util::sync::CancellationToken;
use tracing::info;
use tracing_subscriber::EnvFilter;

use dynamo_llm::block_manager::block::transfer::remote::RemoteKey;
use dynamo_llm::block_manager::distributed::registry::{
    BinaryCodec, NoMetadata, PositionalEviction, PositionalKey, RegistryHubConfig,
    RegistryMetricsSink, ZmqHub, ZmqHubServerConfig,
};

/// Type alias for the concrete hub type we use.
///
/// Uses `PositionalKey` for position-aware storage and `PositionalEviction`
/// which evicts from highest positions first (tail-first), optimizing for
/// prefix reuse in KV cache scenarios.
type G4RegistryHub = ZmqHub<
    PositionalKey,
    RemoteKey,
    NoMetadata,
    PositionalEviction<PositionalKey, RemoteKey>,
    BinaryCodec<PositionalKey, RemoteKey, NoMetadata>,
>;

struct MetricsState {
    started_at: Instant,
    capacity: u64,
    plugin_metrics: Option<Arc<RegistryPromMetrics>>,
}

const LATENCY_BUCKETS_SECS: [f64; 14] = [
    0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
];

#[derive(Default)]
struct AtomicHistogram {
    buckets: Vec<AtomicU64>,
    count: AtomicU64,
    sum_nanos: AtomicU64,
}

impl AtomicHistogram {
    fn new() -> Self {
        let mut buckets = Vec::with_capacity(LATENCY_BUCKETS_SECS.len());
        for _ in LATENCY_BUCKETS_SECS {
            buckets.push(AtomicU64::new(0));
        }
        Self {
            buckets,
            count: AtomicU64::new(0),
            sum_nanos: AtomicU64::new(0),
        }
    }

    fn observe_duration(&self, duration: std::time::Duration) {
        let secs = duration.as_secs_f64();
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum_nanos.fetch_add(
            duration.as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
        for (i, le) in LATENCY_BUCKETS_SECS.iter().enumerate() {
            if secs <= *le {
                self.buckets[i].fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn append_prometheus(&self, out: &mut String, name: &str, labels: &str) {
        for (idx, le) in LATENCY_BUCKETS_SECS.iter().enumerate() {
            let count = self.buckets[idx].load(Ordering::Relaxed);
            out.push_str(&format!("{name}_bucket{{{labels}le=\"{le}\"}} {count}\n"));
        }
        out.push_str(&format!(
            "{name}_bucket{{{labels}le=\"+Inf\"}} {}\n",
            self.count.load(Ordering::Relaxed)
        ));
        out.push_str(&format!(
            "{name}_sum{{{labels}}} {:.9}\n",
            self.sum_nanos.load(Ordering::Relaxed) as f64 / 1_000_000_000.0
        ));
        out.push_str(&format!(
            "{name}_count{{{labels}}} {}\n",
            self.count.load(Ordering::Relaxed)
        ));
    }
}

#[derive(Default)]
struct QueryMetricSet {
    requests_total: AtomicU64,
    keys_total: AtomicU64,
    latency: AtomicHistogram,
}

impl QueryMetricSet {
    fn new() -> Self {
        Self {
            requests_total: AtomicU64::new(0),
            keys_total: AtomicU64::new(0),
            latency: AtomicHistogram::new(),
        }
    }
}

#[derive(Default)]
struct RegistryPromMetrics {
    records_current: AtomicU64,
    query_decode_failures_total: AtomicU64,
    match_hits_total: AtomicU64,
    match_misses_total: AtomicU64,
    lease_active: AtomicU64,
    lease_grants_total: AtomicU64,
    lease_denied_total: AtomicU64,
    lease_released_total: AtomicU64,
    lease_expired_total: AtomicU64,
    can_offload: QueryMetricSet,
    match_query: QueryMetricSet,
    remove: QueryMetricSet,
    touch: QueryMetricSet,
    register_batches_total: AtomicU64,
    register_entries_total: AtomicU64,
    register_latency: AtomicHistogram,
}

impl RegistryPromMetrics {
    fn new(initial_records: usize) -> Self {
        Self {
            records_current: AtomicU64::new(initial_records as u64),
            query_decode_failures_total: AtomicU64::new(0),
            match_hits_total: AtomicU64::new(0),
            match_misses_total: AtomicU64::new(0),
            lease_active: AtomicU64::new(0),
            lease_grants_total: AtomicU64::new(0),
            lease_denied_total: AtomicU64::new(0),
            lease_released_total: AtomicU64::new(0),
            lease_expired_total: AtomicU64::new(0),
            can_offload: QueryMetricSet::new(),
            match_query: QueryMetricSet::new(),
            remove: QueryMetricSet::new(),
            touch: QueryMetricSet::new(),
            register_batches_total: AtomicU64::new(0),
            register_entries_total: AtomicU64::new(0),
            register_latency: AtomicHistogram::new(),
        }
    }

    fn query_set(&self, query_type: &str) -> Option<&QueryMetricSet> {
        match query_type {
            "can_offload" => Some(&self.can_offload),
            "match" => Some(&self.match_query),
            "remove" => Some(&self.remove),
            "touch" => Some(&self.touch),
            _ => None,
        }
    }

    fn append_query_metrics(&self, out: &mut String, query_type: &str, set: &QueryMetricSet) {
        let requests = set.requests_total.load(Ordering::Relaxed);
        let keys = set.keys_total.load(Ordering::Relaxed);
        out.push_str(&format!(
            "registry_query_requests_total{{query_type=\"{query_type}\"}} {requests}\n"
        ));
        out.push_str(&format!(
            "registry_query_keys_total{{query_type=\"{query_type}\"}} {keys}\n"
        ));
        let labels = format!("query_type=\"{query_type}\",");
        set.latency
            .append_prometheus(out, "registry_query_latency_seconds", &labels);
    }

    fn render_prometheus(&self) -> String {
        let mut out = String::new();
        out.push_str(
            "# HELP registry_records_current Current number of records in the registry.\n",
        );
        out.push_str("# TYPE registry_records_current gauge\n");
        out.push_str(&format!(
            "registry_records_current {}\n",
            self.records_current.load(Ordering::Relaxed)
        ));

        out.push_str(
            "# HELP registry_query_requests_total Total number of query requests processed.\n",
        );
        out.push_str("# TYPE registry_query_requests_total counter\n");
        out.push_str(
            "# HELP registry_query_keys_total Total number of keys processed by queries.\n",
        );
        out.push_str("# TYPE registry_query_keys_total counter\n");
        out.push_str("# HELP registry_query_latency_seconds Registry query latency in seconds.\n");
        out.push_str("# TYPE registry_query_latency_seconds histogram\n");
        self.append_query_metrics(&mut out, "can_offload", &self.can_offload);
        self.append_query_metrics(&mut out, "match", &self.match_query);
        self.append_query_metrics(&mut out, "remove", &self.remove);
        self.append_query_metrics(&mut out, "touch", &self.touch);

        out.push_str("# HELP registry_query_decode_failures_total Total query decode failures.\n");
        out.push_str("# TYPE registry_query_decode_failures_total counter\n");
        out.push_str(&format!(
            "registry_query_decode_failures_total {}\n",
            self.query_decode_failures_total.load(Ordering::Relaxed)
        ));

        out.push_str(
            "# HELP registry_query_match_hits_total Total keys that matched on `match` queries.\n",
        );
        out.push_str("# TYPE registry_query_match_hits_total counter\n");
        out.push_str(&format!(
            "registry_query_match_hits_total {}\n",
            self.match_hits_total.load(Ordering::Relaxed)
        ));

        out.push_str(
            "# HELP registry_query_match_misses_total Total keys that missed on `match` queries.\n",
        );
        out.push_str("# TYPE registry_query_match_misses_total counter\n");
        out.push_str(&format!(
            "registry_query_match_misses_total {}\n",
            self.match_misses_total.load(Ordering::Relaxed)
        ));

        out.push_str("# HELP registry_lease_active Current number of active leases.\n");
        out.push_str("# TYPE registry_lease_active gauge\n");
        out.push_str(&format!(
            "registry_lease_active {}\n",
            self.lease_active.load(Ordering::Relaxed)
        ));

        out.push_str("# HELP registry_lease_grants_total Total number of leases granted.\n");
        out.push_str("# TYPE registry_lease_grants_total counter\n");
        out.push_str(&format!(
            "registry_lease_grants_total {}\n",
            self.lease_grants_total.load(Ordering::Relaxed)
        ));

        out.push_str(
            "# HELP registry_lease_denied_total Total number of lease denials due to existing leases.\n",
        );
        out.push_str("# TYPE registry_lease_denied_total counter\n");
        out.push_str(&format!(
            "registry_lease_denied_total {}\n",
            self.lease_denied_total.load(Ordering::Relaxed)
        ));

        out.push_str("# HELP registry_lease_released_total Total number of leases released on register.\n");
        out.push_str("# TYPE registry_lease_released_total counter\n");
        out.push_str(&format!(
            "registry_lease_released_total {}\n",
            self.lease_released_total.load(Ordering::Relaxed)
        ));

        out.push_str("# HELP registry_lease_expired_total Total number of leases expired during cleanup.\n");
        out.push_str("# TYPE registry_lease_expired_total counter\n");
        out.push_str(&format!(
            "registry_lease_expired_total {}\n",
            self.lease_expired_total.load(Ordering::Relaxed)
        ));

        out.push_str(
            "# HELP registry_register_batches_total Total registration batches processed.\n",
        );
        out.push_str("# TYPE registry_register_batches_total counter\n");
        out.push_str(&format!(
            "registry_register_batches_total {}\n",
            self.register_batches_total.load(Ordering::Relaxed)
        ));

        out.push_str(
            "# HELP registry_register_entries_total Total registration entries processed.\n",
        );
        out.push_str("# TYPE registry_register_entries_total counter\n");
        out.push_str(&format!(
            "registry_register_entries_total {}\n",
            self.register_entries_total.load(Ordering::Relaxed)
        ));

        out.push_str(
            "# HELP registry_register_latency_seconds Registration batch latency in seconds.\n",
        );
        out.push_str("# TYPE registry_register_latency_seconds histogram\n");
        self.register_latency
            .append_prometheus(&mut out, "registry_register_latency_seconds", "");

        out
    }
}

impl RegistryMetricsSink for RegistryPromMetrics {
    fn on_query_decode_failure(&self) {
        self.query_decode_failures_total
            .fetch_add(1, Ordering::Relaxed);
    }

    fn on_query_processed(
        &self,
        query_type: &'static str,
        keys: usize,
        latency: std::time::Duration,
    ) {
        let Some(set) = self.query_set(query_type) else {
            return;
        };
        set.requests_total.fetch_add(1, Ordering::Relaxed);
        set.keys_total.fetch_add(keys as u64, Ordering::Relaxed);
        set.latency.observe_duration(latency);
    }

    fn on_match_result(&self, hits: usize, misses: usize) {
        self.match_hits_total
            .fetch_add(hits as u64, Ordering::Relaxed);
        self.match_misses_total
            .fetch_add(misses as u64, Ordering::Relaxed);
    }

    fn on_record_count_change(&self, records_current: usize) {
        self.records_current
            .store(records_current as u64, Ordering::Relaxed);
    }

    fn on_register_batch(
        &self,
        entries: usize,
        latency: std::time::Duration,
        records_current: usize,
    ) {
        self.register_batches_total.fetch_add(1, Ordering::Relaxed);
        self.register_entries_total
            .fetch_add(entries as u64, Ordering::Relaxed);
        self.register_latency.observe_duration(latency);
        self.records_current
            .store(records_current as u64, Ordering::Relaxed);
    }

    fn on_can_offload_result(
        &self,
        granted: usize,
        _already_stored: usize,
        leased: usize,
        active_leases: usize,
    ) {
        self.lease_grants_total
            .fetch_add(granted as u64, Ordering::Relaxed);
        self.lease_denied_total
            .fetch_add(leased as u64, Ordering::Relaxed);
        self.lease_active
            .store(active_leases as u64, Ordering::Relaxed);
    }

    fn on_leases_released(&self, released: usize, active_leases: usize) {
        self.lease_released_total
            .fetch_add(released as u64, Ordering::Relaxed);
        self.lease_active
            .store(active_leases as u64, Ordering::Relaxed);
    }

    fn on_leases_expired(&self, expired: usize, active_leases: usize) {
        self.lease_expired_total
            .fetch_add(expired as u64, Ordering::Relaxed);
        self.lease_active
            .store(active_leases as u64, Ordering::Relaxed);
    }
}

fn parse_bool_env(name: &str, default_value: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(default_value)
}

async fn run_metrics_server(
    addr: String,
    state: Arc<MetricsState>,
    cancel: CancellationToken,
) -> Result<()> {
    let listener = TcpListener::bind(&addr).await?;
    info!("Registry metrics endpoint listening on http://{addr}/metrics");

    loop {
        tokio::select! {
            _ = cancel.cancelled() => {
                return Ok(());
            }
            accepted = listener.accept() => {
                let (mut socket, _) = match accepted {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::warn!(error = %e, "registry metrics accept failed");
                        continue;
                    }
                };

                let state = state.clone();
                tokio::spawn(async move {
                    let mut buf = [0u8; 4096];
                    let n = match socket.read(&mut buf).await {
                        Ok(n) => n,
                        Err(e) => {
                            tracing::warn!(error = %e, "registry metrics read failed");
                            return;
                        }
                    };
                    if n == 0 {
                        return;
                    }

                    let req = String::from_utf8_lossy(&buf[..n]);
                    let first_line = req.lines().next().unwrap_or_default();

                    let (status, body, content_type) = if first_line.starts_with("GET /metrics") {
                        let uptime = state.started_at.elapsed().as_secs_f64();
                        let mut body = format!(
                            "# HELP registry_up Registry process liveness.\n\
                             # TYPE registry_up gauge\n\
                             registry_up 1\n\
                             # HELP registry_capacity_entries Configured registry capacity.\n\
                             # TYPE registry_capacity_entries gauge\n\
                             registry_capacity_entries {}\n\
                             # HELP registry_uptime_seconds Registry uptime in seconds.\n\
                             # TYPE registry_uptime_seconds gauge\n\
                             registry_uptime_seconds {:.3}\n",
                            state.capacity, uptime
                        );
                        if let Some(plugin_metrics) = &state.plugin_metrics {
                            body.push_str(&plugin_metrics.render_prometheus());
                        }
                        ("200 OK", body, "text/plain; version=0.0.4")
                    } else if first_line.starts_with("GET /healthz") {
                        ("200 OK", "ok\n".to_string(), "text/plain")
                    } else {
                        ("404 Not Found", "not found\n".to_string(), "text/plain")
                    };

                    let resp = format!(
                        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    if let Err(e) = socket.write_all(resp.as_bytes()).await {
                        tracing::warn!(error = %e, "registry metrics write failed");
                    }
                });
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_target(true)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .init();

    // Load config from environment
    let config = RegistryHubConfig::from_env();

    info!("╔══════════════════════════════════════════════════════════════╗");
    info!("║           Distributed Object Registry                        ║");
    info!("╠══════════════════════════════════════════════════════════════╣");
    info!(
        "║  Capacity:        {:<43}║",
        format!("{} entries", config.capacity)
    );
    info!("║  Query Addr:      {:<43}║", config.query_addr);
    info!("║  Register Addr:   {:<43}║", config.register_addr);
    info!(
        "║  Lease Timeout:   {:<43}║",
        format!("{} secs", config.lease_timeout.as_secs())
    );
    let metrics_addr = std::env::var("DYN_REGISTRY_HUB_METRICS_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:9108".to_string());
    let plugin_metrics_enabled = parse_bool_env("DYN_REGISTRY_HUB_PLUGIN_METRICS", true);
    info!("║  Metrics Addr:    {:<43}║", metrics_addr);
    info!(
        "║  Plugin Metrics:  {:<43}║",
        if plugin_metrics_enabled {
            "enabled"
        } else {
            "disabled"
        }
    );
    info!("╚══════════════════════════════════════════════════════════════╝");

    // Convert to ZmqHubServerConfig
    let zmq_config = ZmqHubServerConfig {
        query_addr: config.query_addr,
        pull_addr: config.register_addr,
        capacity: config.capacity,
        lease_ttl: config.lease_timeout,
        lease_cleanup_interval: std::time::Duration::from_secs(5),
    };

    // Create hub with positional eviction storage and codec
    // PositionalEviction evicts from highest positions first (tail-first),
    // which is optimal for KV cache prefix reuse.
    let storage = PositionalEviction::with_capacity(config.capacity as usize);
    let codec = BinaryCodec::new();
    let plugin_metrics = plugin_metrics_enabled.then(|| Arc::new(RegistryPromMetrics::new(0)));
    let hub: G4RegistryHub = if let Some(metrics) = &plugin_metrics {
        ZmqHub::with_metrics_sink(zmq_config, storage, codec, metrics.clone())
    } else {
        ZmqHub::new(zmq_config, storage, codec)
    };

    // Setup cancellation
    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();
    let metrics_state = Arc::new(MetricsState {
        started_at: Instant::now(),
        capacity: config.capacity,
        plugin_metrics,
    });

    let metrics_cancel = cancel.clone();
    tokio::spawn(async move {
        if let Err(e) = run_metrics_server(metrics_addr, metrics_state, metrics_cancel).await {
            tracing::error!(error = %e, "registry metrics server failed");
        }
    });

    // Handle Ctrl+C
    tokio::spawn(async move {
        match signal::ctrl_c().await {
            Ok(()) => {
                info!("Received Ctrl+C, initiating shutdown...");
                cancel_clone.cancel();
            }
            Err(e) => {
                tracing::error!("Failed to listen for Ctrl+C: {}", e);
            }
        }
    });

    // Run hub
    info!("Starting registry hub... Press Ctrl+C to stop.");
    hub.serve(cancel).await?;

    info!("Registry hub stopped.");
    Ok(())
}
