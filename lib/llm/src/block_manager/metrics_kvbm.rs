// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::Router;
use dynamo_runtime::metrics::prometheus_names::{
    kvbm::{
        DISK_CACHE_HIT_RATE, HOST_CACHE_HIT_RATE, MATCHED_TOKENS, OBJECT_CACHE_HIT_RATE,
        OFFLOAD_BLOCKS_D2D, OFFLOAD_BLOCKS_D2H, OFFLOAD_BLOCKS_D2O, OFFLOAD_BLOCKS_H2D,
        OFFLOAD_BYTES_OBJECT, ONBOARD_BLOCKS_D2D, ONBOARD_BLOCKS_H2D, ONBOARD_BLOCKS_O2D,
        ONBOARD_BYTES_OBJECT,
    },
    sanitize_prometheus_name,
};
use prometheus::{Gauge, IntCounter, Opts, Registry};
use std::{collections::HashMap, net::SocketAddr, sync::Arc, thread, time::Duration};
use tokio::{net::TcpListener, sync::Notify};

use crate::http::service::{RouteDoc, metrics::router};

#[derive(Clone, Debug)]
pub struct KvbmMetrics {
    // number of blocks offloaded from device to host
    pub offload_blocks_d2h: IntCounter,

    // number of blocks offloaded from host to disk
    pub offload_blocks_h2d: IntCounter,

    // number of blocks offloaded from device to disk (bypassing host memory)
    pub offload_blocks_d2d: IntCounter,

    // number of blocks offloaded from device to object storage
    pub offload_blocks_d2o: IntCounter,

    // number of blocks onboarded from host to device
    pub onboard_blocks_h2d: IntCounter,

    // number of blocks onboarded from disk to device
    pub onboard_blocks_d2d: IntCounter,

    // number of blocks onboarded from object storage to device
    pub onboard_blocks_o2d: IntCounter,

    // bytes transferred to object storage (offload)
    pub offload_bytes_object: IntCounter,

    // bytes transferred from object storage (onboard)
    pub onboard_bytes_object: IntCounter,

    // number of matched tokens from KVBM
    pub matched_tokens: IntCounter,

    // host cache hit rate (0.0-1.0) from the sliding window
    pub host_cache_hit_rate: Gauge,

    // disk cache hit rate (0.0-1.0) from the sliding window
    pub disk_cache_hit_rate: Gauge,

    // object storage cache hit rate (0.0-1.0) from the sliding window
    pub object_cache_hit_rate: Gauge,

    shutdown_notify: Option<Arc<Notify>>,
}

impl KvbmMetrics {
    /// Create raw metrics and (once per process) spawn an axum server exposing `/metrics` at metrics_port.
    /// Non-blocking: the HTTP server runs on a background task.
    pub fn new(mr: &KvbmMetricsRegistry, create_endpoint: bool, metrics_port: u16) -> Self {
        // 1) register kvbm metrics
        let offload_blocks_d2h = mr
            .create_intcounter(
                OFFLOAD_BLOCKS_D2H,
                "The number of offload blocks from device to host",
                &[],
            )
            .unwrap();
        let offload_blocks_h2d = mr
            .create_intcounter(
                OFFLOAD_BLOCKS_H2D,
                "The number of offload blocks from host to disk",
                &[],
            )
            .unwrap();
        let offload_blocks_d2d = mr
            .create_intcounter(
                OFFLOAD_BLOCKS_D2D,
                "The number of offload blocks from device to disk (bypassing host memory)",
                &[],
            )
            .unwrap();
        let offload_blocks_d2o = mr
            .create_intcounter(
                OFFLOAD_BLOCKS_D2O,
                "The number of offload blocks from device to object storage",
                &[],
            )
            .unwrap();
        let onboard_blocks_h2d = mr
            .create_intcounter(
                ONBOARD_BLOCKS_H2D,
                "The number of onboard blocks from host to device",
                &[],
            )
            .unwrap();
        let onboard_blocks_d2d = mr
            .create_intcounter(
                ONBOARD_BLOCKS_D2D,
                "The number of onboard blocks from disk to device",
                &[],
            )
            .unwrap();
        let onboard_blocks_o2d = mr
            .create_intcounter(
                ONBOARD_BLOCKS_O2D,
                "The number of onboard blocks from object storage to device",
                &[],
            )
            .unwrap();
        let offload_bytes_object = mr
            .create_intcounter(
                OFFLOAD_BYTES_OBJECT,
                "Bytes transferred to object storage (offload)",
                &[],
            )
            .unwrap();
        let onboard_bytes_object = mr
            .create_intcounter(
                ONBOARD_BYTES_OBJECT,
                "Bytes transferred from object storage (onboard)",
                &[],
            )
            .unwrap();
        let matched_tokens = mr
            .create_intcounter(MATCHED_TOKENS, "The number of matched tokens", &[])
            .unwrap();
        let host_cache_hit_rate = mr
            .create_gauge(
                HOST_CACHE_HIT_RATE,
                "Host cache hit rate (0.0-1.0) from the sliding window",
                &[],
            )
            .unwrap();
        let disk_cache_hit_rate = mr
            .create_gauge(
                DISK_CACHE_HIT_RATE,
                "Disk cache hit rate (0.0-1.0) from the sliding window",
                &[],
            )
            .unwrap();
        let object_cache_hit_rate = mr
            .create_gauge(
                OBJECT_CACHE_HIT_RATE,
                "Object storage cache hit rate (0.0-1.0) from the sliding window",
                &[],
            )
            .unwrap();

        // early return if no endpoint is needed
        if !create_endpoint {
            return Self {
                offload_blocks_d2h,
                offload_blocks_h2d,
                offload_blocks_d2d,
                offload_blocks_d2o,
                onboard_blocks_h2d,
                onboard_blocks_d2d,
                onboard_blocks_o2d,
                offload_bytes_object,
                onboard_bytes_object,
                matched_tokens,
                host_cache_hit_rate,
                disk_cache_hit_rate,
                object_cache_hit_rate,
                shutdown_notify: None,
            };
        }

        // 2) start HTTP server in background with graceful shutdown via Notify
        let registry = mr.inner(); // Arc<Registry>
        let notify = Arc::new(Notify::new());
        let notify_for_task = notify.clone();

        let addr = SocketAddr::from(([0, 0, 0, 0], metrics_port));
        let (_route_docs, app): (Vec<RouteDoc>, Router) = router(
            (*registry).clone(), // take owned Registry (Clone) for router to wrap in Arc
            None,                // or Some("/metrics".to_string()) to override the path
        );

        let run_server = async move {
            let listener = match TcpListener::bind(addr).await {
                Ok(listener) => listener,
                Err(err) => {
                    panic!("failed to bind metrics server to {addr}: {err}");
                }
            };

            if let Err(err) = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    // wait for shutdown signal
                    notify_for_task.notified().await;
                })
                .await
            {
                tracing::error!("[kvbm] metrics server error: {err}");
            }
        };

        // Spawn on existing runtime if present, otherwise start our own.
        if tokio::runtime::Handle::try_current().is_ok() {
            tokio::spawn(run_server);
        } else {
            thread::spawn(move || {
                let rt = tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .build()
                    .expect("build tokio runtime");
                rt.block_on(run_server);
            });
        }

        Self {
            offload_blocks_d2h,
            offload_blocks_h2d,
            offload_blocks_d2d,
            offload_blocks_d2o,
            onboard_blocks_h2d,
            onboard_blocks_d2d,
            onboard_blocks_o2d,
            offload_bytes_object,
            onboard_bytes_object,
            matched_tokens,
            host_cache_hit_rate,
            disk_cache_hit_rate,
            object_cache_hit_rate,
            shutdown_notify: Some(notify),
        }
    }

    /// Update cache hit rate metrics from a CacheStatsTracker
    pub fn update_cache_hit_rates(&self, host_rate: f32, disk_rate: f32, object_rate: f32) {
        self.host_cache_hit_rate.set(host_rate as f64);
        self.disk_cache_hit_rate.set(disk_rate as f64);
        self.object_cache_hit_rate.set(object_rate as f64);
    }

    /// Spawn a background task that periodically logs transfer stats summary.
    ///
    /// Logs offload/onboard block counts, bytes transferred, and throughput every `interval`.
    /// The task runs until the metrics instance is dropped.
    ///
    /// # Arguments
    /// * `interval` - How often to log stats (e.g., 30 seconds)
    pub fn spawn_periodic_stats_logger(&self, interval: Duration) {
        let metrics = self.clone();

        let stats_logger = async move {
            let mut ticker = tokio::time::interval(interval);

            // Track previous values to compute deltas
            let mut prev_offload_d2o: u64 = 0;
            let mut prev_onboard_o2d: u64 = 0;
            let mut prev_offload_bytes: u64 = 0;
            let mut prev_onboard_bytes: u64 = 0;

            loop {
                ticker.tick().await;

                // Get current counter values
                let curr_offload_d2o = metrics.offload_blocks_d2o.get();
                let curr_onboard_o2d = metrics.onboard_blocks_o2d.get();
                let curr_offload_bytes = metrics.offload_bytes_object.get();
                let curr_onboard_bytes = metrics.onboard_bytes_object.get();

                // Compute deltas for this period
                let delta_offload_blocks = curr_offload_d2o.saturating_sub(prev_offload_d2o);
                let delta_onboard_blocks = curr_onboard_o2d.saturating_sub(prev_onboard_o2d);
                let delta_offload_bytes = curr_offload_bytes.saturating_sub(prev_offload_bytes);
                let delta_onboard_bytes = curr_onboard_bytes.saturating_sub(prev_onboard_bytes);

                // Update previous values
                prev_offload_d2o = curr_offload_d2o;
                prev_onboard_o2d = curr_onboard_o2d;
                prev_offload_bytes = curr_offload_bytes;
                prev_onboard_bytes = curr_onboard_bytes;

                // Skip logging if no activity
                if delta_offload_blocks == 0 && delta_onboard_blocks == 0 {
                    continue;
                }

                // Compute MB and throughput
                let interval_secs = interval.as_secs_f64();
                let offload_mb = delta_offload_bytes as f64 / (1024.0 * 1024.0);
                let onboard_mb = delta_onboard_bytes as f64 / (1024.0 * 1024.0);
                let offload_mbps = if interval_secs > 0.0 { offload_mb / interval_secs } else { 0.0 };
                let onboard_mbps = if interval_secs > 0.0 { onboard_mb / interval_secs } else { 0.0 };

                tracing::info!(
                    target: "kvbm_stats",
                    interval_secs = interval_secs,
                    offload_blocks = delta_offload_blocks,
                    onboard_blocks = delta_onboard_blocks,
                    offload_mb = offload_mb,
                    onboard_mb = onboard_mb,
                    offload_mbps = offload_mbps,
                    onboard_mbps = onboard_mbps,
                    total_offload_blocks = curr_offload_d2o,
                    total_onboard_blocks = curr_onboard_o2d,
                    "TRANSFER_STATS ({:.0}s): offload={} blocks ({:.1} MB, {:.1} MB/s) | onboard={} blocks ({:.1} MB, {:.1} MB/s) | totals: offload={}, onboard={}",
                    interval_secs,
                    delta_offload_blocks,
                    offload_mb,
                    offload_mbps,
                    delta_onboard_blocks,
                    onboard_mb,
                    onboard_mbps,
                    curr_offload_d2o,
                    curr_onboard_o2d,
                );
            }
        };

        // Always spawn in dedicated thread with its own runtime for isolation
        thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_time()
                .build()
                .expect("build tokio runtime for stats logger");
            rt.block_on(stats_logger);
        });
    }
}

impl Drop for KvbmMetrics {
    fn drop(&mut self) {
        if let Some(n) = &self.shutdown_notify {
            // (all KvbmMetrics clones) + 1 (held by server task)
            // strong_count == 2 means this is the last metrics instance
            if Arc::strong_count(n) == 2 {
                n.notify_waiters();
            }
        }
    }
}

/// A raw, standalone Prometheus metrics registry implementation using the fixed prefix: `kvbm_`
#[derive(Debug, Clone)]
pub struct KvbmMetricsRegistry {
    registry: Arc<Registry>,
    prefix: String,
}

impl KvbmMetricsRegistry {
    pub fn new() -> Self {
        Self {
            registry: Arc::new(Registry::new()),
            prefix: "kvbm".to_string(),
        }
    }

    pub fn create_intcounter(
        &self,
        name: &str,
        description: &str,
        labels: &[(&str, &str)],
    ) -> anyhow::Result<IntCounter> {
        let metrics_name = sanitize_prometheus_name(&format!("{}_{}", self.prefix, name))?;
        let const_labels: HashMap<String, String> = labels
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let opts = Opts::new(metrics_name, description).const_labels(const_labels);
        let c = IntCounter::with_opts(opts)?;
        self.registry.register(Box::new(c.clone()))?;
        Ok(c)
    }

    pub fn create_gauge(
        &self,
        name: &str,
        description: &str,
        labels: &[(&str, &str)],
    ) -> anyhow::Result<Gauge> {
        let metrics_name = sanitize_prometheus_name(&format!("{}_{}", self.prefix, name))?;
        let const_labels: HashMap<String, String> = labels
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let opts = Opts::new(metrics_name, description).const_labels(const_labels);
        let g = Gauge::with_opts(opts)?;
        self.registry.register(Box::new(g.clone()))?;
        Ok(g)
    }

    pub fn inner(&self) -> Arc<Registry> {
        Arc::clone(&self.registry)
    }
}

impl Default for KvbmMetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}
