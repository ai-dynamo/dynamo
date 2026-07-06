// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend metric snapshots published on the event plane.
//!
//! All fields in this first release are required. Fields added later must use
//! `#[serde(default)]`, and `Option` when absence does not mean zero.

use std::{sync::Arc, time::Duration};

use dynamo_runtime::{DistributedRuntime, transports::event_plane::EventPublisher};
use tokio_util::sync::CancellationToken;

use crate::discovery::ModelManager;
use crate::utils::retry::{Backoff, FailureStreak};
use dynamo_http_server::metrics::Metrics;

pub(crate) const FRONTEND_LOAD_TOPIC: &str = "frontend_load";
pub(crate) const FRONTEND_LOAD_PUBLISH_INTERVAL: Duration = Duration::from_secs(1);
const PUBLISHER_RETRY_INITIAL: Duration = Duration::from_millis(100);
const PUBLISHER_RETRY_MAX: Duration = Duration::from_secs(5);

/// Exact configured namespace, else `DYN_NAMESPACE`, else `dynamo`.
pub(crate) fn frontend_load_namespace(configured: Option<String>) -> String {
    configured
        .or_else(|| std::env::var("DYN_NAMESPACE").ok())
        .unwrap_or_else(|| "dynamo".to_string())
}

pub(crate) use dynamo_http_server::load::{
    FrontendLoadFrame, FrontendModelLoad, RequestGauges, RequestTotals,
};

/// Start periodic publishing after the caller checks `DYN_SERVING_LOAD_PUBLISH`.
pub(crate) fn start_frontend_load_publisher(
    runtime: DistributedRuntime,
    namespace: String,
    manager: Arc<ModelManager>,
    metrics: Arc<Metrics>,
    is_ready: impl Fn() -> bool + Send + 'static,
    cancel: CancellationToken,
) {
    runtime.runtime().secondary().spawn(async move {
        let namespace_handle = match runtime.namespace(namespace.clone()) {
            Ok(namespace) => namespace,
            Err(error) => {
                tracing::error!(%namespace, %error, "cannot create frontend load event namespace");
                return;
            }
        };
        let frontend_instance_id = runtime.connection_id();
        // Publish on the first tick, matching the scheduler-load publisher.
        let mut interval = tokio::time::interval(FRONTEND_LOAD_PUBLISH_INTERVAL);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut retry = Backoff::new(PUBLISHER_RETRY_INITIAL, PUBLISHER_RETRY_MAX);
        let mut failures = FailureStreak::default();

        // Reconnect loop: a publish failure drops the publisher and builds a new one.
        loop {
            let publisher = tokio::select! {
                _ = cancel.cancelled() => return,
                result = EventPublisher::for_namespace(&namespace_handle, FRONTEND_LOAD_TOPIC) => result,
            };
            let error = match publisher {
                Ok(publisher) => loop {
                    tokio::select! {
                        _ = cancel.cancelled() => return,
                        _ = interval.tick() => {}
                    }
                    let frame = metrics.frontend_load(
                        frontend_instance_id,
                        is_ready(),
                        manager.committed_model_views().into_iter().map(Into::into).collect(),
                    );
                    let result = tokio::select! {
                        biased;
                        _ = cancel.cancelled() => return,
                        result = publisher.publish(&frame) => result,
                    };
                    match result {
                        Ok(()) => {
                            retry.reset();
                            if let Some(failures) = failures.recover() {
                                tracing::info!(%namespace, failures, "frontend load publishing recovered");
                            }
                        }
                        Err(error) => break error,
                    }
                },
                Err(error) => error,
            };
            let delay = retry.next_delay();
            if failures.fail() {
                tracing::warn!(%namespace, %error, retry_ms = delay.as_millis(), "frontend load publishing failed; reconnecting");
            } else {
                tracing::debug!(%namespace, %error, failures = failures.failures(), retry_ms = delay.as_millis(), "frontend load publishing still failing");
            }
            tokio::select! {
                _ = cancel.cancelled() => return,
                _ = tokio::time::sleep(delay) => {}
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A Relay decodes frames from newer frontends that carry fields it does not
    /// know, but never invents identity or zero load for a frame missing them.
    #[test]
    fn frames_decode_across_versions() {
        use dynamo_runtime::transports::event_plane::Codec;
        use serde_json::json;

        let codec = Codec::default();
        let decode = |frame: serde_json::Value| {
            codec.decode_payload::<FrontendLoadFrame>(&codec.encode_payload(&frame).unwrap())
        };
        let gauges = json!({
            "requests_awaiting_first_token": 1, "requests_generating": 3,
            "awaiting_first_token_input_tokens": 5, "inflight_input_tokens": 7,
            "added_gauge": 1,
        });
        let totals = json!({
            "requests_started": 4, "requests_completed": 1, "requests_failed": 0,
            "requests_cancelled": 0, "input_tokens": 9, "output_tokens": 2,
            "added_total": 1,
        });

        let newer = decode(json!({
            "frontend_instance_id": 10, "incarnation": 1, "sequence": 2,
            "is_serving_ready": true, "added_frame_field": 1,
            "models": [{
                "model": "a", "is_ready": true, "aliases": [], "added_model_field": "x",
                "gauges": gauges, "totals": totals,
            }],
        }))
        .unwrap();
        assert!(newer.is_serving_ready);
        assert_eq!(newer.models[0].gauges.requests_generating, 3);
        assert_eq!(newer.models[0].totals.requests_started, 4);

        assert!(
            decode(json!({
                "incarnation": 1, "sequence": 2, "is_serving_ready": true, "models": [],
            }))
            .is_err()
        );
        assert!(
            decode(json!({
                "frontend_instance_id": 10, "incarnation": 1, "sequence": 2,
                "is_serving_ready": true,
                "models": [{"model": "a", "is_ready": true, "aliases": [], "totals": totals}],
            }))
            .is_err()
        );
    }
}
