// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::{
    Router,
    body::Body,
    extract::State,
    http::{HeaderValue, StatusCode, header},
    response::Response,
    routing::get,
};
use dynamo_runtime::metrics::{MetricsRegistry, prometheus_names::sanitize_prometheus_name};
use prometheus::{Encoder, IntCounter, Opts, Registry, TextEncoder};
use std::{collections::HashMap, net::SocketAddr, sync::Arc, thread};
use tokio::{net::TcpListener, sync::Notify};

#[derive(Clone, Debug)]
pub struct KvbmMetrics {
    // number of offload requests
    pub offload_requests: IntCounter,

    // number of blocks offloaded from device to host
    pub offload_blocks_d2h: IntCounter,

    // number of onboard requests
    pub onboard_requests: IntCounter,

    // number of blocks onboarded from host to device
    pub onboard_blocks_h2d: IntCounter,

    // number of blocks onboarded from disk to device
    pub onboard_blocks_d2d: IntCounter,

    // number of matched tokens from KVBM
    pub matched_tokens: IntCounter,

    shutdown_notify: Option<Arc<Notify>>,
}

impl KvbmMetrics {
    pub fn new(mr: &dyn MetricsRegistry) -> Self {
        let offload_requests = mr
            .create_intcounter("offload_requests", "The number of offload requests", &[])
            .unwrap();
        let offload_blocks_d2h = mr
            .create_intcounter(
                "offload_blocks_d2h",
                "The number of offload blocks from device to host",
                &[],
            )
            .unwrap();
        let onboard_requests = mr
            .create_intcounter("onboard_requests", "The number of onboard requests", &[])
            .unwrap();
        let onboard_blocks_h2d = mr
            .create_intcounter(
                "onboard_blocks_h2d",
                "The number of onboard blocks from host to device",
                &[],
            )
            .unwrap();
        let onboard_blocks_d2d = mr
            .create_intcounter(
                "onboard_blocks_d2d",
                "The number of onboard blocks from disk to device",
                &[],
            )
            .unwrap();
        let matched_tokens = mr
            .create_intcounter("matched_tokens", "The number of matched tokens", &[])
            .unwrap();
        Self {
            offload_requests,
            offload_blocks_d2h,
            onboard_requests,
            onboard_blocks_h2d,
            onboard_blocks_d2d,
            matched_tokens,
            shutdown_notify: None,
        }
    }

    /// Create raw metrics and (once per process) spawn an axum server exposing `/metrics` at metrics_port.
    /// Non-blocking: the HTTP server runs on a background task.
    pub fn new_with_standalone(mr: &KvbmMetricsRegistry, metrics_port: u16) -> Self {
        let offload_requests = mr
            .create_intcounter("offload_requests", "The number of offload requests", &[])
            .unwrap();
        let offload_blocks_d2h = mr
            .create_intcounter(
                "offload_blocks_d2h",
                "The number of offload blocks from device to host",
                &[],
            )
            .unwrap();
        let onboard_requests = mr
            .create_intcounter("onboard_requests", "The number of onboard requests", &[])
            .unwrap();
        let onboard_blocks_h2d = mr
            .create_intcounter(
                "onboard_blocks_h2d",
                "The number of onboard blocks from host to device",
                &[],
            )
            .unwrap();
        let onboard_blocks_d2d = mr
            .create_intcounter(
                "onboard_blocks_d2d",
                "The number of onboard blocks from disk to device",
                &[],
            )
            .unwrap();
        let matched_tokens = mr
            .create_intcounter("matched_tokens", "The number of matched tokens", &[])
            .unwrap();

        // 2) start HTTP server in background with graceful shutdown via Notify
        let registry = mr.inner(); // Arc<Registry>
        let notify = Arc::new(Notify::new());
        let notify_for_task = notify.clone();

        let addr = SocketAddr::from(([0, 0, 0, 0], metrics_port));
        let app = Router::new()
            .route("/metrics", get(metrics_handler))
            .with_state(registry);

        let run_server = async move {
            let listener = TcpListener::bind(addr).await.expect("bind metrics addr");

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
            offload_requests,
            offload_blocks_d2h,
            onboard_requests,
            onboard_blocks_h2d,
            onboard_blocks_d2d,
            matched_tokens,
            shutdown_notify: Some(notify),
        }
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

/// GET /metrics
async fn metrics_handler(State(registry): State<Arc<Registry>>) -> Response {
    let metric_families = registry.gather();
    let encoder = TextEncoder::new();

    let mut buf = Vec::new();
    if let Err(e) = encoder.encode(&metric_families, &mut buf) {
        let mut resp = Response::new(Body::from(format!("encode error: {e}")));
        *resp.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
        resp.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/plain; charset=utf-8"),
        );
        return resp;
    }

    let mut resp = Response::new(Body::from(buf));
    *resp.status_mut() = StatusCode::OK;
    resp.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
    );
    resp
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

    pub fn inner(&self) -> Arc<Registry> {
        Arc::clone(&self.registry)
    }
}

impl Default for KvbmMetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}
