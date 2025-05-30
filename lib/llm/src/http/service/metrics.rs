// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::{extract::State, http::StatusCode, response::IntoResponse, routing::get, Router};
use prometheus::{Encoder, HistogramOpts, HistogramVec, IntCounterVec, IntGaugeVec, Opts};
use std::{sync::Arc, time::Instant};

pub use prometheus::Registry;

use super::RouteDoc;

/// Value for the `status` label in the request counter for successful requests
pub const REQUEST_STATUS_SUCCESS: &str = "success";

/// Value for the `status` label in the request counter if the request failed
pub const REQUEST_STATUS_ERROR: &str = "error";

/// Partial value for the `type` label in the request counter for streaming requests
pub const REQUEST_TYPE_STREAM: &str = "stream";

/// Partial value for the `type` label in the request counter for unary requests
pub const REQUEST_TYPE_UNARY: &str = "unary";

pub struct Metrics {
    request_counter: IntCounterVec,
    inflight_gauge: IntGaugeVec,
    request_duration: HistogramVec,
}

/// RAII object for inflight gauge and request counters
/// If this object is dropped without calling `mark_ok`, then the request will increment
/// the request counter with the `status` label with [`REQUEST_STATUS_ERROR`]; otherwise, it will increment
/// the counter with `status` label [`REQUEST_STATUS_SUCCESS`]
pub struct InflightGuard {
    metrics: Arc<Metrics>,
    model: String,
    endpoint: Endpoint,
    request_type: RequestType,
    status: Status,
    timer: Instant,
}

/// Requests will be logged by the type of endpoint hit
/// This will include llamastack in the future
pub enum Endpoint {
    /// OAI Completions
    Completions,

    /// OAI Chat Completions
    ChatCompletions,

    /// OAI Embeddings
    Embeddings,
}

/// Metrics for the HTTP service
pub enum RequestType {
    /// SingleIn / SingleOut
    Unary,

    /// SingleIn / ManyOut
    Stream,
}

/// Status
pub enum Status {
    Success,
    Error,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new("nv_llm")
    }
}

impl Metrics {
    /// Create Metrics with the given prefix
    /// The following metrics will be created:
    /// - `{prefix}_http_service_requests_total` - IntCounterVec for the total number of requests processed
    /// - `{prefix}_http_service_inflight_requests` - IntGaugeVec for the number of inflight requests
    /// - `{prefix}_http_service_request_duration_seconds` - HistogramVec for the duration of requests
    pub fn new(prefix: &str) -> Self {
        let request_counter = IntCounterVec::new(
            Opts::new(
                format!("{}_http_service_requests_total", prefix),
                "Total number of LLM requests processed",
            ),
            &["model", "endpoint", "request_type", "status"],
        )
        .unwrap();

        let inflight_gauge = IntGaugeVec::new(
            Opts::new(
                format!("{}_http_service_inflight_requests", prefix),
                "Number of inflight requests",
            ),
            &["model"],
        )
        .unwrap();

        let buckets = vec![0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0];

        let request_duration = HistogramVec::new(
            HistogramOpts::new(
                format!("{}_http_service_request_duration_seconds", prefix),
                "Duration of LLM requests",
            )
            .buckets(buckets),
            &["model"],
        )
        .unwrap();

        Metrics {
            request_counter,
            inflight_gauge,
            request_duration,
        }
    }

    /// Get the number of successful requests for the given dimensions:
    /// - model
    /// - endpoint (completions/chat_completions)
    /// - request type (unary/stream)
    /// - status (success/error)
    pub fn get_request_counter(
        &self,
        model: &str,
        endpoint: &Endpoint,
        request_type: &RequestType,
        status: &Status,
    ) -> u64 {
        self.request_counter
            .with_label_values(&[
                model,
                endpoint.as_str(),
                request_type.as_str(),
                status.as_str(),
            ])
            .get()
    }

    /// Increment the counter for requests for the given dimensions:
    /// - model
    /// - endpoint (completions/chat_completions)
    /// - request type (unary/stream)
    /// - status (success/error)
    fn inc_request_counter(
        &self,
        model: &str,
        endpoint: &Endpoint,
        request_type: &RequestType,
        status: &Status,
    ) {
        self.request_counter
            .with_label_values(&[
                model,
                endpoint.as_str(),
                request_type.as_str(),
                status.as_str(),
            ])
            .inc()
    }

    /// Get the number if inflight requests for the given model
    pub fn get_inflight_count(&self, model: &str) -> i64 {
        self.inflight_gauge.with_label_values(&[model]).get()
    }

    fn inc_inflight_gauge(&self, model: &str) {
        self.inflight_gauge.with_label_values(&[model]).inc()
    }

    fn dec_inflight_gauge(&self, model: &str) {
        self.inflight_gauge.with_label_values(&[model]).dec()
    }

    pub fn register(&self, registry: &Registry) -> Result<(), prometheus::Error> {
        registry.register(Box::new(self.request_counter.clone()))?;
        registry.register(Box::new(self.inflight_gauge.clone()))?;
        registry.register(Box::new(self.request_duration.clone()))?;
        Ok(())
    }

    /// Create a new [`InflightGuard`] for the given model and annotate if its a streaming request,
    /// and the kind of endpoint that was hit
    ///
    /// The [`InflightGuard`] is an RAII object will handle incrementing the inflight gauge and
    /// request counters.
    pub fn create_inflight_guard(
        self: Arc<Self>,
        model: &str,
        endpoint: Endpoint,
        streaming: bool,
    ) -> InflightGuard {
        let request_type = if streaming {
            RequestType::Stream
        } else {
            RequestType::Unary
        };

        InflightGuard::new(self.clone(), model.to_string(), endpoint, request_type)
    }
}

impl InflightGuard {
    fn new(
        metrics: Arc<Metrics>,
        model: String,
        endpoint: Endpoint,
        request_type: RequestType,
    ) -> Self {
        // Start the timer
        let timer = Instant::now();

        // Increment the inflight gauge when the guard is created
        metrics.inc_inflight_gauge(&model);

        // Return the RAII Guard
        InflightGuard {
            metrics,
            model,
            endpoint,
            request_type,
            status: Status::Error,
            timer,
        }
    }

    pub(crate) fn mark_ok(&mut self) {
        self.status = Status::Success;
    }
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        // Decrement the gauge when the guard is dropped
        self.metrics.dec_inflight_gauge(&self.model);

        // the frequency on incrementing the full request counter is relatively low
        // if we were incrementing the counter on every forward pass, we'd use static CounterVec or
        // discrete counter object without the more costly lookup required for the following calls
        self.metrics.inc_request_counter(
            &self.model,
            &self.endpoint,
            &self.request_type,
            &self.status,
        );

        // Record the duration of the request
        self.metrics
            .request_duration
            .with_label_values(&[&self.model])
            .observe(self.timer.elapsed().as_secs_f64());
    }
}

impl std::fmt::Display for Endpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Endpoint::Completions => write!(f, "completions"),
            Endpoint::ChatCompletions => write!(f, "chat_completions"),
            Endpoint::Embeddings => write!(f, "embeddings"),
        }
    }
}

impl Endpoint {
    pub fn as_str(&self) -> &'static str {
        match self {
            Endpoint::Completions => "completions",
            Endpoint::ChatCompletions => "chat_completions",
            Endpoint::Embeddings => "embeddings",
        }
    }
}

impl RequestType {
    pub fn as_str(&self) -> &'static str {
        match self {
            RequestType::Unary => REQUEST_TYPE_UNARY,
            RequestType::Stream => REQUEST_TYPE_STREAM,
        }
    }
}

impl Status {
    pub fn as_str(&self) -> &'static str {
        match self {
            Status::Success => REQUEST_STATUS_SUCCESS,
            Status::Error => REQUEST_STATUS_ERROR,
        }
    }
}

/// Create a new router with the given path
pub fn router(registry: Registry, path: Option<String>) -> (Vec<RouteDoc>, Router) {
    let registry = Arc::new(registry);
    let path = path.unwrap_or_else(|| "/metrics".to_string());
    let doc = RouteDoc::new(axum::http::Method::GET, &path);
    let route = Router::new()
        .route(&path, get(handler_metrics))
        .with_state(registry);
    (vec![doc], route)
}

/// Metrics Handler
async fn handler_metrics(State(registry): State<Arc<Registry>>) -> impl IntoResponse {
    let encoder = prometheus::TextEncoder::new();
    let metric_families = registry.gather();
    let mut buffer = vec![];
    if encoder.encode(&metric_families, &mut buffer).is_err() {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to encode metrics",
        )
            .into_response();
    }

    let metrics = match String::from_utf8(buffer) {
        Ok(metrics) => metrics,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to encode metrics",
            )
                .into_response()
        }
    };

    (StatusCode::OK, metrics).into_response()
}
