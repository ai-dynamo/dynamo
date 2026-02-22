// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::Router;
use dynamo_runtime::metrics::prometheus_names::{
    kvbm::{
        DISK_CACHE_HIT_RATE, HOST_CACHE_HIT_RATE, MATCHED_TOKENS, OFFLOAD_BLOCKS_D2D,
        OFFLOAD_BLOCKS_D2H, OFFLOAD_BLOCKS_D2R, OFFLOAD_BLOCKS_H2D, OFFLOAD_BYTES_REMOTE,
        ONBOARD_BLOCKS_D2D, ONBOARD_BLOCKS_H2D, ONBOARD_BLOCKS_R2D, ONBOARD_BYTES_REMOTE,
        REMOTE_CACHE_HIT_RATE, REMOTE_READ_FAILURES, REMOTE_TRANSFER_LATENCY_SECONDS,
        REMOTE_WRITE_FAILURES,
    },
    sanitize_prometheus_name,
};
use prometheus::{Gauge, HistogramOpts, HistogramVec, IntCounter, Opts, Registry};
use std::{collections::HashMap, net::SocketAddr, sync::Arc, thread};
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

    // number of blocks offloaded from device to remote storage
    pub offload_blocks_d2r: IntCounter,

    // number of blocks onboarded from host to device
    pub onboard_blocks_h2d: IntCounter,

    // number of blocks onboarded from disk to device
    pub onboard_blocks_d2d: IntCounter,

    // number of blocks onboarded from remote storage to device
    pub onboard_blocks_r2d: IntCounter,

    // number of matched tokens from KVBM
    pub matched_tokens: IntCounter,

    // host cache hit rate (0.0-1.0) from the sliding window
    pub host_cache_hit_rate: Gauge,

    // disk cache hit rate (0.0-1.0) from the sliding window
    pub disk_cache_hit_rate: Gauge,

    // remote cache hit rate (0.0-1.0) from the sliding window
    pub remote_cache_hit_rate: Gauge,

    // number of failed remote storage read operations (blocks)
    pub remote_read_failures: IntCounter,

    // number of failed remote storage write operations (blocks)
    pub remote_write_failures: IntCounter,

    // bytes transferred to remote storage (offload)
    pub offload_bytes_remote: IntCounter,

    // bytes transferred from remote storage (onboard)
    pub onboard_bytes_remote: IntCounter,

    // transfer latency histogram for remote storage operations
    pub remote_transfer_latency_seconds: HistogramVec,

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
        let offload_blocks_d2r = mr
            .create_intcounter(
                OFFLOAD_BLOCKS_D2R,
                "The number of offload blocks from device to remote storage",
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
        let onboard_blocks_r2d = mr
            .create_intcounter(
                ONBOARD_BLOCKS_R2D,
                "The number of onboard blocks from remote storage to device",
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
        let remote_cache_hit_rate = mr
            .create_gauge(
                REMOTE_CACHE_HIT_RATE,
                "Remote storage cache hit rate (0.0-1.0) from the sliding window",
                &[],
            )
            .unwrap();
        let remote_read_failures = mr
            .create_intcounter(
                REMOTE_READ_FAILURES,
                "The number of failed remote storage read operations (blocks)",
                &[],
            )
            .unwrap();
        let remote_write_failures = mr
            .create_intcounter(
                REMOTE_WRITE_FAILURES,
                "The number of failed remote storage write operations (blocks)",
                &[],
            )
            .unwrap();
        let offload_bytes_remote = mr
            .create_intcounter(
                OFFLOAD_BYTES_REMOTE,
                "The number of bytes offloaded from device to remote storage",
                &[],
            )
            .unwrap();
        let onboard_bytes_remote = mr
            .create_intcounter(
                ONBOARD_BYTES_REMOTE,
                "The number of bytes onboarded from remote storage to device",
                &[],
            )
            .unwrap();
        let remote_transfer_latency_seconds = mr
            .create_histogram_vec(
                REMOTE_TRANSFER_LATENCY_SECONDS,
                "Remote storage transfer latency in seconds",
                &["direction", "result", "backend"],
                vec![
                    0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
                    10.0, 30.0, 60.0,
                ],
            )
            .unwrap();
        // early return if no endpoint is needed
        if !create_endpoint {
            return Self {
                offload_blocks_d2h,
                offload_blocks_h2d,
                offload_blocks_d2d,
                offload_blocks_d2r,
                onboard_blocks_h2d,
                onboard_blocks_d2d,
                onboard_blocks_r2d,
                matched_tokens,
                host_cache_hit_rate,
                disk_cache_hit_rate,
                remote_cache_hit_rate,
                remote_read_failures,
                remote_write_failures,
                offload_bytes_remote,
                onboard_bytes_remote,
                remote_transfer_latency_seconds,
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
            offload_blocks_d2r,
            onboard_blocks_h2d,
            onboard_blocks_d2d,
            onboard_blocks_r2d,
            matched_tokens,
            host_cache_hit_rate,
            disk_cache_hit_rate,
            remote_cache_hit_rate,
            remote_read_failures,
            remote_write_failures,
            offload_bytes_remote,
            onboard_bytes_remote,
            remote_transfer_latency_seconds,
            shutdown_notify: Some(notify),
        }
    }

    /// Update cache hit rate metrics from a CacheStatsTracker
    pub fn update_cache_hit_rates(&self, host_rate: f32, disk_rate: f32, remote_rate: f32) {
        self.host_cache_hit_rate.set(host_rate as f64);
        self.disk_cache_hit_rate.set(disk_rate as f64);
        self.remote_cache_hit_rate.set(remote_rate as f64);
    }
    /// Record failed remote storage read operations
    pub fn record_remote_read_failure(&self, num_blocks: u64) {
        self.remote_read_failures.inc_by(num_blocks);
    }

    /// Record failed remote storage write operations
    pub fn record_remote_write_failure(&self, num_blocks: u64) {
        self.remote_write_failures.inc_by(num_blocks);
    }

    /// Record successful offload bytes to remote storage.
    pub fn record_remote_offload_bytes(&self, num_bytes: u64) {
        self.offload_bytes_remote.inc_by(num_bytes);
    }

    /// Record successful onboard bytes from remote storage.
    pub fn record_remote_onboard_bytes(&self, num_bytes: u64) {
        self.onboard_bytes_remote.inc_by(num_bytes);
    }

    /// Record remote transfer latency.
    pub fn record_remote_transfer_latency(
        &self,
        direction: &str,
        result: &str,
        backend: &str,
        seconds: f64,
    ) {
        self.remote_transfer_latency_seconds
            .with_label_values(&[direction, result, backend])
            .observe(seconds);
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

    pub fn create_histogram_vec(
        &self,
        name: &str,
        description: &str,
        labels: &[&str],
        buckets: Vec<f64>,
    ) -> anyhow::Result<HistogramVec> {
        let metrics_name = sanitize_prometheus_name(&format!("{}_{}", self.prefix, name))?;
        let opts = HistogramOpts::new(metrics_name, description).buckets(buckets);
        let h = HistogramVec::new(opts, labels)?;
        self.registry.register(Box::new(h.clone()))?;
        Ok(h)
    }
}

impl Default for KvbmMetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}
