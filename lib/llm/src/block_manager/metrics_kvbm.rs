// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::Router;
use dynamo_runtime::metrics::prometheus_names::{
    kvbm::{
        DISK_CACHE_HIT_RATE, HOST_CACHE_HIT_RATE, MATCHED_TOKENS, OBJECT_CACHE_HIT_RATE,
        OBJECT_READ_FAILURES, OBJECT_WRITE_FAILURES, OFFLOAD_BLOCKS_D2D, OFFLOAD_BLOCKS_D2H,
        OFFLOAD_BLOCKS_D2O, OFFLOAD_BLOCKS_H2D, ONBOARD_BLOCKS_D2D, ONBOARD_BLOCKS_H2D,
        ONBOARD_BLOCKS_O2D, QUEUE_D2D_ONBOARD_SECONDS, QUEUE_D2D_SECONDS, QUEUE_D2H_SECONDS,
        QUEUE_D2O_SECONDS, QUEUE_H2D_ONBOARD_SECONDS, QUEUE_H2D_SECONDS, QUEUE_O2D_SECONDS,
        TRANSFER_D2D_ONBOARD_SECONDS, TRANSFER_D2D_SECONDS, TRANSFER_D2H_SECONDS,
        TRANSFER_D2O_SECONDS, TRANSFER_H2D_ONBOARD_SECONDS, TRANSFER_H2D_SECONDS,
        TRANSFER_O2D_SECONDS,
    },
    sanitize_prometheus_name,
};
use prometheus::{Gauge, Histogram, HistogramOpts, HistogramVec, IntCounter, Opts, Registry};
use std::{collections::HashMap, net::SocketAddr, sync::Arc, sync::OnceLock, thread};
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

    // number of matched tokens from KVBM
    pub matched_tokens: IntCounter,

    // host cache hit rate (0.0-1.0) from the sliding window
    pub host_cache_hit_rate: Gauge,

    // disk cache hit rate (0.0-1.0) from the sliding window
    pub disk_cache_hit_rate: Gauge,

    // object cache hit rate (0.0-1.0) from the sliding window
    pub object_cache_hit_rate: Gauge,

    // number of failed object storage read operations (blocks)
    pub object_read_failures: IntCounter,

    // number of failed object storage write operations (blocks)
    pub object_write_failures: IntCounter,

    // Transfer latency histograms (HistogramVec with worker_id label)
    pub transfer_d2h: HistogramVec,
    pub transfer_h2d: HistogramVec,
    pub transfer_d2d: HistogramVec,
    pub transfer_d2o: HistogramVec,
    pub transfer_h2d_onboard: HistogramVec,
    pub transfer_d2d_onboard: HistogramVec,
    pub transfer_o2d: HistogramVec,

    // Queue wait histograms
    pub queue_d2h: Histogram,
    pub queue_h2d: Histogram,
    pub queue_d2d: Histogram,
    pub queue_d2o: Histogram,
    pub queue_h2d_onboard: Histogram,
    pub queue_d2d_onboard: Histogram,
    pub queue_o2d: Histogram,

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
        let object_read_failures = mr
            .create_intcounter(
                OBJECT_READ_FAILURES,
                "The number of failed object storage read operations (blocks)",
                &[],
            )
            .unwrap();
        let object_write_failures = mr
            .create_intcounter(
                OBJECT_WRITE_FAILURES,
                "The number of failed object storage write operations (blocks)",
                &[],
            )
            .unwrap();

        // Transfer latency histograms (HistogramVec with worker_id label)
        let transfer_d2h = mr
            .create_histogram_vec(TRANSFER_D2H_SECONDS, "Device to Host transfer time")
            .unwrap();
        let transfer_h2d = mr
            .create_histogram_vec(TRANSFER_H2D_SECONDS, "Host to Disk transfer time")
            .unwrap();
        let transfer_d2d = mr
            .create_histogram_vec(TRANSFER_D2D_SECONDS, "Device to Disk direct transfer time")
            .unwrap();
        let transfer_d2o = mr
            .create_histogram_vec(
                TRANSFER_D2O_SECONDS,
                "Device to Object Storage transfer time",
            )
            .unwrap();
        let transfer_h2d_onboard = mr
            .create_histogram_vec(
                TRANSFER_H2D_ONBOARD_SECONDS,
                "Host to Device onboard transfer time",
            )
            .unwrap();
        let transfer_d2d_onboard = mr
            .create_histogram_vec(
                TRANSFER_D2D_ONBOARD_SECONDS,
                "Disk to Device onboard transfer time",
            )
            .unwrap();
        let transfer_o2d = mr
            .create_histogram_vec(
                TRANSFER_O2D_SECONDS,
                "Object Storage to Device transfer time",
            )
            .unwrap();

        // Queue wait histograms
        let queue_d2h = mr
            .create_histogram(QUEUE_D2H_SECONDS, "D2H queue wait time")
            .unwrap();
        let queue_h2d = mr
            .create_histogram(QUEUE_H2D_SECONDS, "H2D (offload) queue wait time")
            .unwrap();
        let queue_d2d = mr
            .create_histogram(QUEUE_D2D_SECONDS, "D2D (offload) queue wait time")
            .unwrap();
        let queue_d2o = mr
            .create_histogram(QUEUE_D2O_SECONDS, "D2O queue wait time")
            .unwrap();
        let queue_h2d_onboard = mr
            .create_histogram(
                QUEUE_H2D_ONBOARD_SECONDS,
                "H2D (onboard) queue wait time",
            )
            .unwrap();
        let queue_d2d_onboard = mr
            .create_histogram(
                QUEUE_D2D_ONBOARD_SECONDS,
                "D2D (onboard) queue wait time",
            )
            .unwrap();
        let queue_o2d = mr
            .create_histogram(QUEUE_O2D_SECONDS, "O2D queue wait time")
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
                matched_tokens,
                host_cache_hit_rate,
                disk_cache_hit_rate,
                object_cache_hit_rate,
                object_read_failures,
                object_write_failures,
                transfer_d2h,
                transfer_h2d,
                transfer_d2d,
                transfer_d2o,
                transfer_h2d_onboard,
                transfer_d2d_onboard,
                transfer_o2d,
                queue_d2h,
                queue_h2d,
                queue_d2d,
                queue_d2o,
                queue_h2d_onboard,
                queue_d2d_onboard,
                queue_o2d,
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
            matched_tokens,
            host_cache_hit_rate,
            disk_cache_hit_rate,
            object_cache_hit_rate,
            object_read_failures,
            object_write_failures,
            transfer_d2h,
            transfer_h2d,
            transfer_d2d,
            transfer_d2o,
            transfer_h2d_onboard,
            transfer_d2d_onboard,
            transfer_o2d,
            queue_d2h,
            queue_h2d,
            queue_d2d,
            queue_d2o,
            queue_h2d_onboard,
            queue_d2d_onboard,
            queue_o2d,
            shutdown_notify: Some(notify),
        }
    }

    /// Update cache hit rate metrics from a CacheStatsTracker
    pub fn update_cache_hit_rates(&self, host_rate: f32, disk_rate: f32, object_rate: f32) {
        self.host_cache_hit_rate.set(host_rate as f64);
        self.disk_cache_hit_rate.set(disk_rate as f64);
        self.object_cache_hit_rate.set(object_rate as f64);
    }
    /// Record failed object storage read operations
    pub fn record_object_read_failure(&self, num_blocks: u64) {
        self.object_read_failures.inc_by(num_blocks);
    }

    /// Record failed object storage write operations
    pub fn record_object_write_failure(&self, num_blocks: u64) {
        self.object_write_failures.inc_by(num_blocks);
    }

    /// Store metrics in the process-global singleton. First call wins; subsequent calls are no-ops.
    pub fn init_global(metrics: KvbmMetrics) {
        let _ = GLOBAL_KVBM_METRICS.set(metrics);
    }

    /// Get a reference to the process-global metrics, if initialized.
    pub fn get_global() -> Option<&'static KvbmMetrics> {
        GLOBAL_KVBM_METRICS.get()
    }
}

static GLOBAL_KVBM_METRICS: OnceLock<KvbmMetrics> = OnceLock::new();

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

    pub fn create_histogram_vec(
        &self,
        name: &str,
        description: &str,
    ) -> anyhow::Result<HistogramVec> {
        let metrics_name = sanitize_prometheus_name(&format!("{}_{}", self.prefix, name))?;
        let buckets = vec![
            0.0, 0.0001, 0.00032, 0.001, 0.0032, 0.01, 0.032, 0.1, 0.32, 1.0, 3.2, 10.0,
        ];
        let opts = HistogramOpts::new(metrics_name, description).buckets(buckets);
        let h = HistogramVec::new(opts, &["worker_id"])?;
        self.registry.register(Box::new(h.clone()))?;
        Ok(h)
    }

    pub fn create_histogram(&self, name: &str, description: &str) -> anyhow::Result<Histogram> {
        let metrics_name = sanitize_prometheus_name(&format!("{}_{}", self.prefix, name))?;
        let buckets = vec![
            0.0, 0.0001, 0.00032, 0.001, 0.0032, 0.01, 0.032, 0.1, 0.32, 1.0, 3.2, 10.0,
        ];
        let opts = HistogramOpts::new(metrics_name, description).buckets(buckets);
        let h = Histogram::with_opts(opts)?;
        self.registry.register(Box::new(h.clone()))?;
        Ok(h)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_histogram_basic() {
        let mr = KvbmMetricsRegistry::new();
        let h = mr
            .create_histogram("test_latency", "A test histogram")
            .unwrap();
        h.observe(0.005);
        h.observe(0.1);
        assert_eq!(h.get_sample_count(), 2);
    }

    #[test]
    fn test_kvbm_metrics_has_transfer_histograms() {
        let mr = KvbmMetricsRegistry::new();
        let metrics = KvbmMetrics::new(&mr, false, 0);
        // Verify transfer histograms (HistogramVec) are functional
        metrics
            .transfer_d2h
            .with_label_values(&["test"])
            .observe(0.001);
        metrics.queue_d2h.observe(0.002);
        assert_eq!(
            metrics
                .transfer_d2h
                .with_label_values(&["test"])
                .get_sample_count(),
            1
        );
        assert_eq!(metrics.queue_d2h.get_sample_count(), 1);
    }
}
