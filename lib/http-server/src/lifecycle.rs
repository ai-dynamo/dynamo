// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::time::Duration;

use axum::body::Body;
use axum::http::Response;
use futures::StreamExt;
use tokio::sync::Notify;

/// Middleware that echoes `x-request-id` from request to response headers.
pub async fn echo_request_id_header(
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let x_request_id = request.headers().get("x-request-id").cloned();
    let mut response = next.run(request).await;
    if let Some(value) = x_request_id {
        response.headers_mut().insert("x-request-id", value);
    }
    response
}

/// Lifecycle stage for the HTTP frontend.
///
/// The stage gates readiness and request admission separately from the runtime
/// cancellation token so the frontend can stop accepting new requests before
/// tearing down discovery and transport state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ServiceStage {
    /// The frontend is ready to admit new inference requests.
    Ready = 0,
    /// The frontend is rejecting new requests while admitted responses drain.
    Draining = 1,
    /// The frontend is cancelling runtime state and shutting down.
    Stopping = 2,
}

impl ServiceStage {
    fn as_u8(self) -> u8 {
        self as u8
    }

    fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Ready,
            1 => Self::Draining,
            _ => Self::Stopping,
        }
    }
}

impl std::fmt::Display for ServiceStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ready => f.write_str("ready"),
            Self::Draining => f.write_str("draining"),
            Self::Stopping => f.write_str("stopping"),
        }
    }
}

/// Shared HTTP frontend lifecycle and inflight request tracker.
///
/// `ServiceObserver` is shared by HTTP handlers, health endpoints, and the
/// shutdown path. It lets shutdown first mark the frontend as draining, then
/// wait for admitted inference response bodies to complete before cancelling
/// runtime state.
#[derive(Debug)]
pub struct ServiceObserver {
    stage: AtomicU8,
    inflight_inference: AtomicU64,
    inflight_zero: Notify,
}

impl Default for ServiceObserver {
    fn default() -> Self {
        Self {
            stage: AtomicU8::new(ServiceStage::Ready.as_u8()),
            inflight_inference: AtomicU64::new(0),
            inflight_zero: Notify::new(),
        }
    }
}

impl ServiceObserver {
    /// Return the current frontend lifecycle stage.
    pub fn stage(&self) -> ServiceStage {
        ServiceStage::from_u8(self.stage.load(Ordering::Acquire))
    }

    /// Return true when the frontend should admit new inference requests.
    pub fn is_ready(&self) -> bool {
        self.stage() == ServiceStage::Ready
    }

    /// Mark the frontend as draining.
    ///
    /// Draining makes readiness fail and causes request admission checks to
    /// reject new inference requests while existing response bodies continue.
    pub fn start_draining(&self) {
        tracing::info!(
            previous_stage = ?self.stage(),
            inflight_requests = self.inflight_count(),
            "frontend service entering draining stage"
        );
        self.stage
            .store(ServiceStage::Draining.as_u8(), Ordering::Release);
    }

    /// Mark the frontend as stopping.
    ///
    /// Stopping is entered after inflight requests drain or the graceful
    /// shutdown timeout expires.
    pub fn start_stopping(&self) {
        tracing::info!(
            previous_stage = ?self.stage(),
            inflight_requests = self.inflight_count(),
            "frontend service entering stopping stage"
        );
        self.stage
            .store(ServiceStage::Stopping.as_u8(), Ordering::Release);
    }

    /// Track one admitted inference response body.
    ///
    /// The returned permit must live for the full HTTP response body lifetime,
    /// including streaming responses. Dropping the permit decrements the
    /// inflight count and wakes shutdown waiters when the count reaches zero.
    pub fn acquire_inflight(self: &Arc<Self>) -> InflightPermit {
        self.inflight_inference.fetch_add(1, Ordering::Relaxed);
        InflightPermit {
            observer: self.clone(),
        }
    }

    /// Return the number of admitted inference requests still in flight.
    pub fn inflight_count(&self) -> u64 {
        self.inflight_inference.load(Ordering::Acquire)
    }

    /// Wait until all admitted inference requests drain or `timeout` expires.
    ///
    /// Returns `true` when inflight work drained before the timeout and `false`
    /// when shutdown should proceed because the timeout expired.
    pub async fn wait_inflight_zero_or_timeout(&self, timeout: Duration) -> bool {
        tokio::time::timeout(timeout, async {
            loop {
                let notified = self.inflight_zero.notified();
                tokio::pin!(notified);
                // Register before reading the count so a final permit drop
                // cannot notify between the count check and the await.
                notified.as_mut().enable();
                if self.inflight_count() == 0 {
                    break;
                }
                notified.as_mut().await;
            }
        })
        .await
        .is_ok()
    }
}

/// RAII guard for one admitted inference response.
///
/// This permit is held by a response-body wrapper so it is released only when
/// the client response body finishes or is dropped.
pub struct InflightPermit {
    observer: Arc<ServiceObserver>,
}

impl Drop for InflightPermit {
    fn drop(&mut self) {
        if self
            .observer
            .inflight_inference
            .fetch_sub(1, Ordering::AcqRel)
            == 1
            && self.observer.stage() != ServiceStage::Ready
        {
            self.observer.inflight_zero.notify_waiters();
        }
    }
}

/// Hold an inflight permit until the complete HTTP body is sent or dropped.
pub fn track_inflight_response(response: Response<Body>, permit: InflightPermit) -> Response<Body> {
    let (parts, body) = response.into_parts();
    // Keep the permit alive until the full response body, including streams,
    // finishes or is dropped.
    let stream = body.into_data_stream().map(move |result| {
        let _permit = &permit;
        result
    });
    Response::from_parts(parts, Body::from_stream(stream))
}
